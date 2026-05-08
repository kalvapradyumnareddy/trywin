import asyncio
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

PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, say that you don't know — do not make up an answer.

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
            num_predict=256,
        )
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retriever_top_k},
        )
        self._qa_chain = self._build_chain()

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
        self._vectorstore.add_documents(chunks)
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

    async def astream_query(self, question: str) -> Tuple[list, AsyncGenerator[str, None]]:
        """
        Returns (sources, token_async_generator).
        Retrieval is done up-front; tokens stream from Ollama in real time via httpx.
        """
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, self._retriever.invoke, question)

        context = "\n\n".join(d.page_content for d in docs)
        prompt_text = PROMPT_TEMPLATE.format(context=context, question=question)

        seen: set = set()
        sources: list = []
        for d in docs:
            key = (d.metadata.get("source", ""), d.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": d.metadata.get("source", ""),
                    "page": d.metadata.get("page"),
                })

        async def _token_gen() -> AsyncGenerator[str, None]:
            url = settings.ollama_base_url.rstrip("/") + "/api/generate"
            payload = {
                "model": settings.ollama_model,
                "prompt": prompt_text,
                "stream": True,
                "options": {"temperature": 0.1, "num_ctx": 1024, "num_predict": 256},
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                async with client.stream("POST", url, json=payload) as resp:
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

    def query(self, question: str) -> dict:
        result = self._qa_chain.invoke({"query": question})
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
            }
            for doc in result.get("source_documents", [])
        ]
        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s["source"], s["page"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {"answer": result["result"], "sources": unique_sources}

    def list_documents(self) -> List[str]:
        docs = []
        for path in Path(settings.documents_dir).rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.append(path.name)
        return docs

    def collection_count(self) -> int:
        return self._vectorstore._collection.count()
