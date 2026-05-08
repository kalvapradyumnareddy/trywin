import asyncio
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from rag_engine import RAGEngine, SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

rag: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logger.info("Initializing RAG engine...")
    rag = RAGEngine()
    # Auto-ingest any documents already present on startup
    existing = list(Path(settings.documents_dir).rglob("*"))
    if any(p.suffix.lower() in SUPPORTED_EXTENSIONS for p in existing):
        logger.info("Auto-ingesting existing documents from %s", settings.documents_dir)
        rag.ingest_directory(settings.documents_dir)
    logger.info("RAG engine ready. Vector store has %d chunks.", rag.collection_count())
    asyncio.create_task(rag.prewarm())
    yield
    logger.info("Shutting down RAG engine.")


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation over your PDF, DOCX, and TXT documents",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    chunks = rag.collection_count() if rag else 0
    return {"status": "ok", "vector_store_chunks": chunks}


@app.get("/documents")
def list_documents():
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")
    return {"documents": rag.list_documents()}


@app.post("/ingest/upload")
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    dest = Path(settings.documents_dir) / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = rag.ingest_file(str(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {"filename": file.filename, "chunks_ingested": chunks}


@app.post("/ingest/directory")
def ingest_directory():
    """Re-ingest all documents already present in the documents directory."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")
    results = rag.ingest_directory(settings.documents_dir)
    return {"results": results}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag.query(request.question)
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(answer=result["answer"], sources=result["sources"])


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints — lets OpenWebUI treat this as a model
# ---------------------------------------------------------------------------

RAG_MODEL_ID = "rag-documents"


class _Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = RAG_MODEL_ID
    messages: List[_Message]
    stream: Optional[bool] = False


def _build_answer_with_sources(result: dict) -> str:
    answer = result["answer"]
    sources = result.get("sources", [])
    if not sources:
        return answer
    seen: set = set()
    unique = []
    for s in sources:
        key = s.get("source", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(Path(key).name)
    if unique:
        answer += "\n\n**Sources:** " + ", ".join(unique)
    return answer


def _sources_text(sources: list) -> str:
    seen: set = set()
    names = []
    for s in sources:
        src = s.get("source", "")
        if src and src not in seen:
            seen.add(src)
            names.append(Path(src).name)
    return ("\n\n**Sources:** " + ", ".join(names)) if names else ""


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": RAG_MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "rag-api",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    question = user_messages[-1].content.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:
        try:
            sources, token_gen = await rag.astream_query(question)
        except Exception as exc:
            logger.error("RAG stream query failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

        async def generate():
            yield "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": RAG_MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }) + "\n\n"
            async for token in token_gen:
                yield "data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": RAG_MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }) + "\n\n"
            footer = _sources_text(sources)
            if footer:
                yield "data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": RAG_MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": footer}, "finish_reason": None}],
                }) + "\n\n"
            yield "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": RAG_MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming: run blocking call in thread pool so event loop stays healthy
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, rag.query, question)
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    content = _build_answer_with_sources(result)
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": RAG_MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
