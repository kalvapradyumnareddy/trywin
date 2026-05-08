# =============================================================================
# main.py — FastAPI application entry point
#
# What is FastAPI?
#   FastAPI is a modern Python web framework for building APIs. It is:
#   - FAST: built on top of Starlette (async web server) and Pydantic (validation).
#   - AUTO-DOCUMENTED: visit /docs in a browser to get a live interactive UI
#     for every endpoint, auto-generated from your code.
#   - TYPE-SAFE: uses Python type hints to validate request/response data
#     automatically — no manual if/else checks needed.
#
# What does this file do?
#   It defines all the HTTP endpoints (routes) the RAG API exposes:
#     GET  /health                  — liveness/readiness check
#     GET  /documents               — list uploaded files
#     POST /ingest/upload           — upload and index a new document
#     POST /ingest/directory        — re-index all documents on disk
#     POST /query                   — ask a question, get a full answer
#     GET  /v1/models               — OpenAI-compatible model list
#     POST /v1/chat/completions     — OpenAI-compatible chat (used by OpenWebUI)
# =============================================================================

import asyncio        # Async task scheduling (create_task, run_in_executor)
import json           # Serialize Python dicts to JSON strings for SSE events
import logging        # Write structured log messages
import shutil         # Copy file-like objects (used when saving uploaded files)
import time           # Get current Unix timestamp for response metadata
import uuid           # Generate unique IDs for each chat completion response
from contextlib import asynccontextmanager  # Decorator to write async setup/teardown
from pathlib import Path                    # Clean cross-platform file path handling
from typing import List, Optional           # Type hint helpers

# --- FastAPI imports ----------------------------------------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
#   FastAPI      — the main application class
#   File         — marker that tells FastAPI a parameter is an uploaded file field
#   UploadFile   — the type for an uploaded file; gives access to filename, content type, etc.
#   HTTPException — raise this to return an HTTP error response (e.g. 404, 400, 500)
#   BackgroundTasks — FastAPI's built-in way to run work AFTER the response is sent

from fastapi.responses import StreamingResponse
# ↑ A special response type that streams data to the client incrementally
#   instead of sending the whole body at once. Used for token streaming.

from pydantic import BaseModel
# ↑ Pydantic is FastAPI's data validation library.
#   Any class that extends BaseModel automatically:
#   - Validates incoming JSON against the field types
#   - Returns a clear 422 error if the data is wrong
#   - Serializes Python objects back to JSON

from config import settings                        # App-wide configuration values
from rag_engine import RAGEngine, SUPPORTED_EXTENSIONS  # Our RAG brain

# Set up logging. Every log line will look like:
#   2026-05-09 10:00:00,123 INFO Initializing RAG engine...
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Global variable that holds the single RAGEngine instance.
# It's None at import time and gets assigned inside lifespan() at startup.
# "| None" is Python 3.10+ union type syntax meaning: RAGEngine or None.
rag: RAGEngine | None = None


# =============================================================================
# FASTAPI CONCEPT: Lifespan (startup / shutdown hooks)
# =============================================================================
# Before FastAPI starts handling requests, we need to:
#   1. Initialize the RAGEngine (load embeddings model, connect to ChromaDB).
#   2. Pre-warm the LLM so the first user query isn't slow.
#
# @asynccontextmanager turns a function with `yield` into a context manager.
# Code BEFORE `yield` runs at startup. Code AFTER `yield` runs at shutdown.
# The `app` argument is the FastAPI instance (we don't use it here but FastAPI
# requires it as a parameter).

async def _startup_ingest() -> None:
    """
    Runs in the background after startup to index any documents already on disk.

    Why background? Ingesting documents can take 30–60 seconds for large files.
    If we blocked startup on this, FastAPI wouldn't start accepting requests until
    it finished — causing Kubernetes liveness probes to fail and restart the pod.
    By running as a background task, the app becomes ready immediately and ingests
    while serving traffic.

    Skip if ChromaDB already has data — no need to re-embed on every restart.
    """
    if rag.collection_count() > 0:
        logger.info("Vector store has %d chunks, skipping startup ingest", rag.collection_count())
        return
    existing = list(Path(settings.documents_dir).rglob("*"))
    if any(p.suffix.lower() in SUPPORTED_EXTENSIONS for p in existing):
        logger.info("Vector store empty — ingesting existing documents...")
        # run_in_executor: ingest_directory is a blocking (sync) function.
        # Calling a blocking function directly in async code would freeze the
        # event loop and stop FastAPI from handling other requests.
        # run_in_executor runs it in a thread pool instead — non-blocking.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, rag.ingest_directory, settings.documents_dir)
        logger.info("Startup ingest done. Vector store has %d chunks.", rag.collection_count())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: runs setup before the server starts, teardown after it stops.

    Startup (before yield):
      - Create the RAGEngine (loads models, connects to ChromaDB).
      - Schedule background tasks for startup ingest and model pre-warm.

    Shutdown (after yield):
      - Close the HTTP client so TCP connections to Ollama are released cleanly.

    asyncio.create_task() schedules a coroutine to run concurrently in the
    background. The current function doesn't wait for it — it just fires and
    continues. This is what lets startup complete in ~6 seconds even if ingestion
    takes 30 seconds.
    """
    global rag  # Tell Python we're assigning to the module-level variable, not a local one
    logger.info("Initializing RAG engine...")
    rag = RAGEngine()
    logger.info("RAG engine ready. Vector store has %d chunks.", rag.collection_count())
    asyncio.create_task(_startup_ingest())  # Index documents in background (non-blocking)
    asyncio.create_task(rag.prewarm())      # Load LLM into Ollama's RAM in background
    yield  # ← Server is running and handling requests between startup and shutdown
    await rag.aclose()  # Cleanup: close persistent HTTP connections
    logger.info("Shutting down RAG engine.")


# =============================================================================
# FASTAPI CONCEPT: The App Instance
# =============================================================================
# FastAPI() creates the WSGI/ASGI application. All decorators (@app.get, etc.)
# register routes on this object.
#
# title/description/version: show up in the auto-generated /docs Swagger UI.
# lifespan: the startup/shutdown handler we defined above.

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation over your PDF, DOCX, and TXT documents",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# FASTAPI CONCEPT: Pydantic Models (Request & Response Schemas)
# =============================================================================
# These classes define the shape of JSON bodies for requests and responses.
# FastAPI automatically:
#   - Parses incoming JSON and validates field types.
#   - Returns HTTP 422 Unprocessable Entity with a clear error message if
#     required fields are missing or have the wrong type.
#   - Serializes response objects back to JSON.
#
# Example: POST /query with body {"question": "who is prajjumna"}
#   FastAPI reads the JSON, creates a QueryRequest(question="who is prajjumna"),
#   and passes it to the endpoint function — no manual json.loads() needed.

class QueryRequest(BaseModel):
    question: str  # Required field — FastAPI will reject requests without it


class QueryResponse(BaseModel):
    answer: str    # The LLM's generated answer
    sources: list  # List of source documents used to generate the answer


# =============================================================================
# FASTAPI CONCEPT: Route Decorators
# =============================================================================
# @app.get("/path") registers a GET endpoint.
# @app.post("/path") registers a POST endpoint.
# The decorated function is called the "route handler" or "endpoint function".
# FastAPI calls it automatically whenever a matching HTTP request arrives.


@app.get("/health")
def health():
    """
    Health check endpoint. Kubernetes calls this every 30–60 seconds to decide
    if the pod is alive and ready to serve traffic.

    Returns 200 OK with a JSON body — Kubernetes only checks the status code.
    We also include vector_store_chunks so we can quickly see if documents
    are indexed when debugging without exec-ing into the pod.
    """
    chunks = rag.collection_count() if rag else 0
    return {"status": "ok", "vector_store_chunks": chunks}


@app.get("/documents")
def list_documents():
    """
    Returns the list of documents currently stored on disk.
    Useful for checking what's been uploaded without SSHing into the server.
    """
    if rag is None:
        # HTTPException tells FastAPI to stop processing and return an error
        # response immediately. 503 = Service Unavailable — used when the
        # server is up but a dependency (our RAG engine) isn't ready yet.
        raise HTTPException(status_code=503, detail="RAG engine not ready")
    return {"documents": rag.list_documents()}


# =============================================================================
# FASTAPI CONCEPT: Background Tasks
# =============================================================================
# FastAPI's BackgroundTasks lets you run a function AFTER the HTTP response
# has already been sent to the client. Perfect for slow work like indexing.
#
# Without this, the client would wait 30–60 seconds staring at a loading
# spinner while the document is being embedded. With BackgroundTasks, the
# client gets an instant "uploaded" confirmation and indexing happens quietly.

async def _ingest_background(file_path: str) -> None:
    """
    Runs after the upload response is sent. Embeds the document and stores
    chunks in ChromaDB. Errors here are logged but don't affect the client
    since the response was already sent.
    """
    loop = asyncio.get_running_loop()
    try:
        n = await loop.run_in_executor(None, rag.ingest_file, file_path)
        logger.info("Background ingest done: %d chunks from %s", n, file_path)
    except Exception as exc:
        logger.error("Background ingest failed for %s: %s", file_path, exc)


# =============================================================================
# FASTAPI CONCEPT: File Uploads
# =============================================================================
# UploadFile is FastAPI's type for multipart file uploads (like HTML form uploads).
# It gives you access to:
#   file.filename  — original filename from the client
#   file.file      — a file-like object you can read bytes from
#   file.content_type — MIME type (e.g. "application/pdf")
#
# File(...) means the field is required. The ... (Ellipsis) is Python's way
# of saying "no default value — this MUST be provided".

@app.post("/ingest/upload")
async def upload_and_ingest(
    background_tasks: BackgroundTasks,   # FastAPI injects this automatically
    file: UploadFile = File(...),        # The uploaded file from the multipart form
):
    """
    Accepts a file upload, saves it to disk, then indexes it in the background.

    The response is sent immediately after saving — the client doesn't wait
    for embedding/indexing to complete (that happens via BackgroundTasks).
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    # Validate file extension before saving anything to disk.
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        # 400 Bad Request — the client sent something we can't handle.
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Save the uploaded file to the documents directory.
    # shutil.copyfileobj copies from a file-like source to a file-like destination
    # in chunks — memory-efficient even for large files.
    dest = Path(settings.documents_dir) / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Register the ingest as a background task. FastAPI will call this after
    # the return statement below sends the HTTP response.
    background_tasks.add_task(_ingest_background, str(dest))

    # Return immediately — the client gets this response right away.
    return {"filename": file.filename, "status": "uploaded", "message": "Indexing in background"}


@app.post("/ingest/directory")
def ingest_directory():
    """
    Re-indexes all documents already present on disk.
    Useful after manually copying files into the documents directory,
    or to force a full re-index if something went wrong.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")
    results = rag.ingest_directory(settings.documents_dir)
    return {"results": results}


# =============================================================================
# FASTAPI CONCEPT: response_model
# =============================================================================
# response_model=QueryResponse tells FastAPI two things:
#   1. Serialize the return value using the QueryResponse Pydantic model.
#   2. Show this schema in /docs so API users know what to expect back.
#
# If the function returns extra fields not in QueryResponse, FastAPI strips them.
# If required fields are missing, FastAPI raises an internal error.

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Synchronous question-answering endpoint.
    Blocks until the LLM generates the full answer, then returns it.
    Use this for programmatic access where streaming isn't needed.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")
    if not request.question.strip():
        # .strip() removes leading/trailing whitespace. An all-spaces question
        # would waste an LLM call, so we reject it early with a 400 error.
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag.query(request.question)
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        # 500 Internal Server Error — something unexpected went wrong on our side.
        raise HTTPException(status_code=500, detail=str(exc))

    # FastAPI automatically serializes this Pydantic model to JSON.
    return QueryResponse(answer=result["answer"], sources=result["sources"])


# =============================================================================
# OpenAI-compatible endpoints
# =============================================================================
# OpenWebUI (the chat UI) expects the backend to behave like OpenAI's API.
# By implementing /v1/models and /v1/chat/completions with the same request/
# response format as OpenAI, OpenWebUI treats our RAG API as if it were GPT-4.
# No changes needed to OpenWebUI — it just works.

RAG_MODEL_ID = "rag-documents"  # The model name OpenWebUI will display


class _Message(BaseModel):
    """One message in a conversation. role is "user", "assistant", or "system"."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """
    Mirrors the OpenAI /v1/chat/completions request body.
    stream=False by default — OpenWebUI sends stream=True to get token-by-token output.
    """
    model: str = RAG_MODEL_ID          # Which model to use (we only have one)
    messages: List[_Message]           # Full conversation history
    stream: Optional[bool] = False     # True = stream tokens, False = return full answer


def _build_answer_with_sources(result: dict) -> str:
    """
    Appends a "Sources: filename.pdf" footer to the answer for non-streaming responses.
    Deduplicates source files so the same file isn't listed twice.
    """
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
            unique.append(Path(key).name)  # Just the filename, not the full path
    if unique:
        answer += "\n\n**Sources:** " + ", ".join(unique)
    return answer


def _sources_text(sources: list) -> str:
    """
    Builds the sources footer string for the streaming response.
    Same logic as above but returns just the text, not appended to anything.
    """
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
    """
    OpenAI-compatible model list endpoint.
    OpenWebUI calls this to discover available models and show them in the UI.
    We return a single entry: "rag-documents".
    """
    return {
        "object": "list",
        "data": [{
            "id": RAG_MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "rag-api",
        }],
    }


# =============================================================================
# FASTAPI CONCEPT: StreamingResponse + Server-Sent Events (SSE)
# =============================================================================
# Normally an HTTP response sends the full body at once and closes.
# StreamingResponse keeps the connection open and sends data in pieces.
#
# Server-Sent Events (SSE) is a protocol where the server pushes lines of text
# to the browser in real time over a single HTTP connection. Each event looks like:
#   data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n
#
# OpenWebUI reads these events and appends each token to the chat bubble,
# creating the "typing" effect. The format exactly mirrors OpenAI's streaming API.

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    The main endpoint OpenWebUI calls for every chat message.

    Two modes depending on request.stream:

    STREAMING (stream=True) — used by OpenWebUI:
      1. Get the last user message from the conversation.
      2. Call rag.astream_query() which returns (sources, async_token_generator).
      3. Wrap each token in an OpenAI-format SSE JSON chunk.
      4. Return a StreamingResponse — FastAPI keeps the HTTP connection open
         and sends each chunk as the generator yields it.

    NON-STREAMING (stream=False):
      1. Run rag.query() in a thread pool (it's a blocking function).
      2. Return the full answer as a single JSON object.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    # Extract the latest user message from the conversation history.
    # OpenWebUI sends the full chat history on every request so the model
    # has context. We only use the last user message for RAG retrieval.
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    question = user_messages[-1].content.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Unique ID for this completion — OpenWebUI uses it to match streaming
    # chunks together. Format mirrors what OpenAI returns.
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())  # Unix timestamp in seconds

    # -------------------------------------------------------------------------
    # STREAMING PATH
    # -------------------------------------------------------------------------
    if request.stream:
        try:
            # astream_query returns immediately with (sources, generator).
            # The generator hasn't produced any tokens yet — it's lazy.
            sources, token_gen = await rag.astream_query(question)
        except Exception as exc:
            logger.error("RAG stream query failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

        async def generate():
            """
            Async generator that produces SSE-formatted chunks one at a time.

            FastAPI's StreamingResponse iterates this generator and writes
            each yielded string to the HTTP response body as it arrives.

            SSE format rules:
              - Each event starts with "data: "
              - The data is a JSON string
              - Each event ends with two newlines: \n\n
              - The stream ends with "data: [DONE]\n\n"
            """
            # First chunk: announce the assistant is starting to respond.
            # delta={"role": "assistant"} is the OpenAI convention for the first chunk.
            yield "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": RAG_MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }) + "\n\n"

            # Stream each token from Ollama as it's generated.
            # `async for` works with async generators — it awaits each yield.
            async for token in token_gen:
                yield "data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": RAG_MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }) + "\n\n"

            # After all tokens, append the sources footer as a final chunk.
            footer = _sources_text(sources)
            if footer:
                yield "data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": RAG_MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": footer}, "finish_reason": None}],
                }) + "\n\n"

            # Final chunk: finish_reason="stop" signals the response is complete.
            yield "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": RAG_MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n"

            # SSE termination signal — tells the client the stream is closed.
            yield "data: [DONE]\n\n"

        # media_type="text/event-stream" is the MIME type for SSE.
        # It tells the browser/client to read the response as a stream,
        # not wait for the full body.
        return StreamingResponse(generate(), media_type="text/event-stream")

    # -------------------------------------------------------------------------
    # NON-STREAMING PATH
    # -------------------------------------------------------------------------
    # rag.query() is a synchronous (blocking) function. Calling it directly
    # in an async function would freeze FastAPI's event loop, preventing it
    # from handling any other requests while the LLM is generating.
    # run_in_executor runs it in a thread pool so the event loop stays free.
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, rag.query, question)
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Build the full response in OpenAI's chat completion format.
    # OpenWebUI reads the "content" field inside choices[0].message.
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
        # We don't track token counts — set to 0 to satisfy OpenAI schema.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
