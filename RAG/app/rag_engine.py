import asyncio
import hashlib
import json
import os
import logging
from pathlib import Path
from typing import AsyncGenerator, List, Tuple

import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

PROMPT_TEMPLATE = """Use the context below to answer the question. Be direct and helpful. Summarize the relevant information clearly.

Context:
{context}

Question: {question}
Answer:"""


def _get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext in {".docx", ".doc"}:
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    return None


class RAGEngine:
    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        os.makedirs(settings.documents_dir, exist_ok=True)

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=5),
        )

        self._embeddings = FastEmbedEmbeddings(
            model_name=settings.embedding_model,
            cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
        )
        self._vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self._embeddings,
            collection_name="rag_documents",
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,
            num_ctx=1024,
            num_predict=200,
        )
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retriever_top_k},
        )
        self._qa_chain = self._build_chain()

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def prewarm(self) -> None:
        try:
            url = settings.ollama_base_url.rstrip("/") + "/api/generate"
            await self._http_client.post(url, json={
                "model": settings.ollama_model,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1},
            })
            logger.info("Model pre-warm complete")
        except Exception as exc:
            logger.warning("Model pre-warm failed (non-fatal): %s", exc)

    def _build_chain(self) -> RetrievalQA:
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        return RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=self._retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def ingest_file(self, file_path: str) -> int:
        loader = _get_loader(file_path)
        if loader is None:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

        docs = loader.load()
        chunks = self._splitter.split_documents(docs)

        try:
            self._vectorstore._collection.delete(where={"source": file_path})
        except Exception:
            pass

        base = hashlib.md5(file_path.encode()).hexdigest()
        ids = [f"{base}_{i}" for i in range(len(chunks))]
        self._vectorstore.add_documents(chunks, ids=ids)

        logger.info("Ingested %d chunks from %s", len(chunks), file_path)
        return len(chunks)

    def ingest_directory(self, directory: str) -> dict:
        results = {}
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    count = self.ingest_file(str(path))
                    results[path.name] = {"chunks": count, "status": "ok"}
                except Exception as exc:
                    logger.error("Failed to ingest %s: %s", path, exc)
                    results[path.name] = {"status": "error", "error": str(exc)}
        return results

    def query(self, question: str) -> dict:
        result = self._qa_chain.invoke({"query": question})
        seen: set = set()
        unique_sources: list = []
        for doc in result.get("source_documents", []):
            src = (doc.metadata.get("source", ""), doc.metadata.get("page"))
            if src not in seen:
                seen.add(src)
                unique_sources.append({"source": src[0], "page": src[1]})
        return {"answer": result["result"], "sources": unique_sources}

    async def astream_query(self, question: str) -> Tuple[list, AsyncGenerator[str, None]]:
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, self._retriever.invoke, question)

        context = "\n\n".join(" ".join(d.page_content.split()) for d in docs)
        prompt_text = PROMPT_TEMPLATE.format(context=context, question=question)

        seen: set = set()
        sources: list = []
        for d in docs:
            src = (d.metadata.get("source", ""), d.metadata.get("page"))
            if src not in seen:
                seen.add(src)
                sources.append({"source": src[0], "page": src[1]})

        async def _token_gen() -> AsyncGenerator[str, None]:
            url = settings.ollama_base_url.rstrip("/") + "/api/generate"
            payload = {
                "model": settings.ollama_model,
                "prompt": prompt_text,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 1024,
                    "num_predict": 200,
                    "num_batch": 512,
                },
            }
            async with self._http_client.stream("POST", url, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

        return sources, _token_gen()

    def list_documents(self) -> List[str]:
        docs = []
        for path in Path(settings.documents_dir).rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.append(path.name)
        return docs

    def collection_count(self) -> int:
        return self._vectorstore._collection.count()
