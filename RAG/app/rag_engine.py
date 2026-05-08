# =============================================================================
# rag_engine.py — The brain of the RAG (Retrieval-Augmented Generation) system
#
# What is RAG?
#   Instead of asking an AI model a question from scratch (which it may not
#   know the answer to), RAG first *retrieves* relevant chunks from your own
#   documents, then *augments* the AI's prompt with that context, so the model
#   answers using your data — not just its training knowledge.
#
# High-level flow:
#   1. User uploads a PDF/DOCX/TXT file  →  ingest_file() splits it into
#      small chunks and stores vector embeddings in ChromaDB.
#   2. User asks a question  →  the retriever finds the most similar chunks.
#   3. Those chunks are injected into the LLM prompt as "context".
#   4. The LLM (phi3.5 running in Ollama) reads the context and answers.
# =============================================================================

import asyncio        # Python's built-in library for writing async (non-blocking) code
import hashlib        # Used to generate a unique fingerprint (MD5 hash) for file paths
import json           # Parse JSON responses from the Ollama API
import os             # Interact with the operating system (create dirs, read env vars)
import logging        # Write log messages so we can monitor what the app is doing
from pathlib import Path                        # A cleaner way to work with file paths
from typing import AsyncGenerator, List, Tuple  # Type hints — help with code readability

import httpx  # An async HTTP client — used to call the Ollama API directly

# LangChain is a framework that makes it easier to build LLM-powered applications.
# It provides ready-made components for splitting documents, storing embeddings,
# and chaining retrieval + generation together.

from langchain.text_splitter import RecursiveCharacterTextSplitter
# ↑ Splits a long document into smaller overlapping chunks so they fit in the
#   LLM's context window and can be embedded individually.

from langchain_community.document_loaders import (
    PyPDFLoader,    # Reads text out of PDF files page by page
    Docx2txtLoader, # Reads text out of Word (.docx/.doc) files
    TextLoader,     # Reads plain .txt files
)

from langchain_community.embeddings import FastEmbedEmbeddings
# ↑ Converts text into a list of numbers (a "vector") that captures its meaning.
#   Similar sentences produce similar vectors. Runs locally via ONNX — no internet needed.

from langchain_chroma import Chroma
# ↑ ChromaDB is a vector database. It stores the embeddings produced above and
#   lets us search for the most semantically similar chunks to a given query.

from langchain_ollama import OllamaLLM
# ↑ A LangChain wrapper around Ollama's local LLM server (phi3.5 in our case).
#   Used for the non-streaming /query path.

from langchain.chains import RetrievalQA
# ↑ A pre-built LangChain "chain" that wires together: retriever → prompt → LLM.
#   When invoked, it automatically fetches relevant chunks and feeds them to the model.

from langchain.prompts import PromptTemplate
# ↑ A template string with named placeholders ({context}, {question}) that LangChain
#   fills in before sending to the LLM.

from config import settings  # Our app's configuration (model names, paths, chunk sizes, etc.)

# Set up a logger for this module. Log messages will appear in `kubectl logs`.
logger = logging.getLogger(__name__)

# The file types this app knows how to read and index.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# This is the instruction we give to the LLM for every question.
# {context} will be replaced with the retrieved document chunks.
# {question} will be replaced with the user's actual question.
# Keeping the instruction simple and direct works best with smaller models.
PROMPT_TEMPLATE = """Use the context below to answer the question. Be direct and helpful. Summarize the relevant information clearly.

Context:
{context}

Question: {question}
Answer:"""


def _get_loader(file_path: str):
    """
    Returns the right LangChain document loader based on the file extension.

    Why do we need different loaders?
    - PDFs store text in a binary format with page metadata.
    - DOCX files are ZIP archives containing XML.
    - TXT files are just plain text.
    Each loader knows how to extract raw text from its specific format.

    Returns None if the file type is not supported.
    """
    ext = Path(file_path).suffix.lower()  # e.g. ".pdf", ".docx", ".txt"

    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext in {".docx", ".doc"}:
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")

    return None  # Unsupported type — caller handles this


class RAGEngine:
    """
    The central class that manages the entire RAG pipeline.

    Responsibilities:
    - Hold all the shared components (embeddings model, vector store, LLM, retriever).
    - Ingest documents (split → embed → store in ChromaDB).
    - Answer questions (retrieve relevant chunks → build prompt → call LLM).
    """

    def __init__(self):
        """
        Called once when the app starts up. Creates and connects all components.
        Think of this as the "wiring" phase — nothing is processed yet.
        """

        # Create the storage directories on disk if they don't already exist.
        # exist_ok=True means: don't raise an error if the folder is already there.
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        os.makedirs(settings.documents_dir, exist_ok=True)

        # --- HTTP Client -------------------------------------------------
        # httpx.AsyncClient is a persistent HTTP connection pool.
        # Instead of opening a new TCP connection to Ollama for every request
        # (which adds ~100ms overhead), we reuse the same connection.
        # timeout=600s: LLM responses on CPU can be slow — don't give up early.
        # max_keepalive_connections=2: keep at most 2 idle connections alive.
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=5),
        )

        # --- Embeddings Model --------------------------------------------
        # FastEmbedEmbeddings runs a small ONNX model (BAAI/bge-small-en-v1.5)
        # locally inside the pod to convert text → vectors.
        # No GPU needed. ~100MB model loaded once at startup.
        # cache_dir: where to store the downloaded ONNX model files on disk.
        self._embeddings = FastEmbedEmbeddings(
            model_name=settings.embedding_model,
            cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
        )

        # --- Vector Store (ChromaDB) -------------------------------------
        # Chroma is our vector database. It saves embeddings to disk so they
        # survive pod restarts. On startup it loads any previously stored data.
        #
        # persist_directory: where ChromaDB stores its files on the hostPath volume.
        # embedding_function: the model above — Chroma calls it when you add documents.
        # collection_name: think of this like a table name inside ChromaDB.
        self._vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self._embeddings,
            collection_name="rag_documents",
        )

        # --- Text Splitter -----------------------------------------------
        # LLMs have a limited context window (can only read so many tokens at once).
        # We can't feed an entire 50-page PDF to the model, so we split it into
        # small overlapping chunks first.
        #
        # chunk_size=300:   each chunk is at most 300 characters long.
        # chunk_overlap=50: the last 50 chars of one chunk appear at the start
        #                   of the next. This avoids cutting a sentence in half
        #                   at a chunk boundary and losing context.
        #
        # "Recursive" means it tries to split on paragraph breaks first,
        # then sentences, then words — preferring natural boundaries.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # --- LLM (Language Model) ----------------------------------------
        # OllamaLLM connects to the Ollama server running in the ollama pod.
        # This is used for the non-streaming /query endpoint via LangChain's chain.
        #
        # temperature=0.1: controls randomness. 0 = deterministic, 1 = very random.
        #                  Low temperature means the model sticks to factual answers.
        # num_ctx=2048:    the context window size in tokens. Larger = more text fits
        #                  in the prompt, but generation gets slower.
        # num_predict=200: stop generating after 200 tokens (~150 words). Keeps
        #                  responses short and fast.
        self._llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,
            num_ctx=2048,
            num_predict=200,
        )

        # --- Retriever ---------------------------------------------------
        # The retriever is how we search ChromaDB. Given a question, it embeds
        # the question and finds the k most similar document chunks.
        #
        # search_type="similarity": use cosine similarity between vectors.
        # k=retriever_top_k: how many chunks to retrieve (currently 3).
        #   More chunks = more context for the LLM, but slower generation.
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retriever_top_k},
        )

        # Build the LangChain QA chain once and reuse it for every non-streaming query.
        self._qa_chain = self._build_chain()

    # -------------------------------------------------------------------------
    # Lifecycle helpers
    # -------------------------------------------------------------------------

    async def aclose(self) -> None:
        """
        Cleanly close the HTTP client when the app shuts down.
        Called from FastAPI's lifespan teardown. Releases the TCP connections
        to Ollama so the OS doesn't leave sockets in TIME_WAIT state.
        """
        await self._http_client.aclose()

    async def prewarm(self) -> None:
        """
        Sends a tiny dummy request to Ollama right after startup.

        Why? Ollama loads the model into RAM only on the first request.
        Loading phi3.5 (~2.2GB) takes 10–20 seconds. By sending a 1-token
        request at startup, we pay this cost once so the first real user
        query feels instant instead of timing out.
        """
        try:
            url = settings.ollama_base_url.rstrip("/") + "/api/generate"
            await self._http_client.post(url, json={
                "model": settings.ollama_model,
                "prompt": "hi",
                "stream": False,
                # num_predict=1: generate just 1 token — enough to load the model.
                # num_ctx=2048: match the context size we use for real queries so
                #               Ollama doesn't have to reload the model later.
                "options": {"num_predict": 1, "num_ctx": 2048},
            })
            logger.info("Model pre-warm complete")
        except Exception as exc:
            # Non-fatal: if Ollama isn't ready yet the first user query will
            # just be slower. Don't crash the app over this.
            logger.warning("Model pre-warm failed (non-fatal): %s", exc)

    # -------------------------------------------------------------------------
    # Chain builder
    # -------------------------------------------------------------------------

    def _build_chain(self) -> RetrievalQA:
        """
        Assembles the LangChain RetrievalQA chain used by the /query endpoint.

        What is a "chain"?
        A chain is a sequence of steps that run in order. This one does:
          Step 1 — Retriever fetches the top-k relevant chunks from ChromaDB.
          Step 2 — PromptTemplate fills {context} and {question} with real values.
          Step 3 — LLM reads the filled-in prompt and generates an answer.

        chain_type="stuff": the simplest strategy — all retrieved chunks are
        "stuffed" into a single prompt. Good when chunks are small (as ours are).

        return_source_documents=True: the chain also returns which chunks it
        used, so we can show the user "Sources: resume.pdf page 2".
        """
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

    # -------------------------------------------------------------------------
    # Document ingestion
    # -------------------------------------------------------------------------

    def ingest_file(self, file_path: str) -> int:
        """
        Reads a document, splits it into chunks, embeds each chunk, and stores
        them in ChromaDB. Returns the number of chunks created.

        This is the pipeline:
          File on disk
            → loader extracts raw text
            → splitter cuts it into small overlapping chunks
            → ChromaDB embeds each chunk and stores (text + vector + metadata)

        Idempotent: safe to call multiple times on the same file.
        We delete the old chunks first so re-uploading a file doesn't
        create duplicates in the database.
        """
        # Pick the right loader for this file type.
        loader = _get_loader(file_path)
        if loader is None:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

        # loader.load() reads the file and returns a list of LangChain Document
        # objects. Each Document has .page_content (the text) and .metadata
        # (e.g. {"source": "/data/documents/resume.pdf", "page": 0}).
        docs = loader.load()

        # Split each Document into smaller chunks. A 10-page PDF might become
        # 60+ chunks depending on chunk_size.
        chunks = self._splitter.split_documents(docs)

        # Delete any existing chunks for this file so re-uploading is clean.
        # The "source" metadata field stores the original file path, which is
        # what we use to identify which chunks belong to which file.
        try:
            self._vectorstore._collection.delete(where={"source": file_path})
        except Exception:
            pass  # If there were no existing chunks, the delete is a no-op

        # Generate deterministic (stable) IDs for each chunk.
        # Why? ChromaDB needs a unique ID per chunk. If we used random IDs,
        # re-uploading the same file would create duplicate entries.
        # Using md5(file_path) + chunk_index means the same file always produces
        # the same IDs, so ChromaDB upserts instead of appending.
        base = hashlib.md5(file_path.encode()).hexdigest()
        ids = [f"{base}_{i}" for i in range(len(chunks))]

        # Store all chunks in ChromaDB. Chroma automatically calls the
        # embedding function on each chunk's text and saves the vector.
        self._vectorstore.add_documents(chunks, ids=ids)

        logger.info("Ingested %d chunks from %s", len(chunks), file_path)
        return len(chunks)

    def ingest_directory(self, directory: str) -> dict:
        """
        Ingests every supported document found anywhere inside `directory`.
        rglob("*") walks all subdirectories recursively.
        Returns a dict mapping filename → result so the caller knows what happened.
        """
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

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Answers a question using the LangChain RetrievalQA chain (non-streaming).

        Used by the /query REST endpoint and the non-streaming OpenAI path.
        Blocks until the full answer is generated — suitable for simple API calls
        where the client is happy to wait.

        Returns: {"answer": "...", "sources": [{"source": "path", "page": 0}, ...]}
        """
        # self._qa_chain.invoke() runs all three steps: retrieve → prompt → LLM.
        # The input key "query" is what LangChain's RetrievalQA expects.
        result = self._qa_chain.invoke({"query": question})

        # Deduplicate sources: the retriever may return multiple chunks from the
        # same page of the same file. We only want to show each page once.
        seen: set = set()
        unique_sources: list = []
        for doc in result.get("source_documents", []):
            src = (doc.metadata.get("source", ""), doc.metadata.get("page"))
            if src not in seen:
                seen.add(src)
                unique_sources.append({"source": src[0], "page": src[1]})

        return {"answer": result["result"], "sources": unique_sources}

    async def astream_query(self, question: str) -> Tuple[list, AsyncGenerator[str, None]]:
        """
        Answers a question with token-by-token streaming (used by OpenWebUI).

        Why streaming?
        Generating a full answer on CPU takes 20–60 seconds. Without streaming,
        the user stares at a blank screen until it's done. With streaming, they
        see each word as it's generated — much better user experience.

        How it works:
          1. Run the retriever synchronously in a thread pool (run_in_executor)
             because LangChain's retriever is not async-native.
          2. Build the prompt string manually with the retrieved context.
          3. Return (sources, async_generator). The caller iterates the generator
             and sends each token to the client as a Server-Sent Event (SSE).

        Returns a tuple of:
          - sources: list of {"source": path, "page": number}
          - async generator that yields one token string at a time
        """

        # run_in_executor runs a blocking (sync) function in a background thread
        # so it doesn't block the async event loop. The event loop can handle
        # other requests while the retriever is doing its work.
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, self._retriever.invoke, question)

        # Build the context string from retrieved chunks.
        # " ".join(d.page_content.split()) collapses all whitespace (newlines,
        # multiple spaces) inside each chunk into single spaces, saving tokens.
        context = "\n\n".join(" ".join(d.page_content.split()) for d in docs)

        # Fill the prompt template with real context and question.
        prompt_text = PROMPT_TEMPLATE.format(context=context, question=question)

        # Collect unique source files from the retrieved chunks (same dedup logic
        # as in query() above).
        seen: set = set()
        sources: list = []
        for d in docs:
            src = (d.metadata.get("source", ""), d.metadata.get("page"))
            if src not in seen:
                seen.add(src)
                sources.append({"source": src[0], "page": src[1]})

        # Define the async generator as a nested function so it can close over
        # prompt_text and sources without needing extra arguments.
        async def _token_gen() -> AsyncGenerator[str, None]:
            """
            Streams tokens from Ollama's /api/generate endpoint one at a time.

            Ollama's streaming API sends newline-delimited JSON, one object per line:
              {"model":"phi3.5","response":"Hello","done":false}
              {"model":"phi3.5","response":" world","done":false}
              {"model":"phi3.5","response":"","done":true,"eval_count":42}

            We parse each line, yield the "response" field (one or more characters),
            and stop when "done" is true.
            """
            url = settings.ollama_base_url.rstrip("/") + "/api/generate"
            payload = {
                "model": settings.ollama_model,
                "prompt": prompt_text,
                "stream": True,       # Tell Ollama to stream tokens as they're generated
                "options": {
                    "temperature": 0.1,   # Low = more focused, less creative
                    "num_ctx": 2048,      # Context window in tokens
                    "num_predict": 200,   # Max tokens to generate (~150 words)
                    "num_batch": 512,     # Tokens processed in parallel on CPU (throughput)
                },
            }
            logger.info("stream_query model=%s prompt_len=%d", settings.ollama_model, len(prompt_text))
            first_token = True

            try:
                # self._http_client.stream() opens the HTTP connection and keeps
                # it open while we read the response line by line.
                # This is different from a normal POST where you wait for the
                # full response body — here we read it incrementally.
                async with self._http_client.stream("POST", url, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue  # Skip empty lines (Ollama sometimes sends them)

                        # Each line is a JSON object. Parse it safely.
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # Skip any malformed lines

                        token = data.get("response", "")
                        if token:
                            if first_token:
                                # Log the first token so we can verify in kubectl logs
                                # that the model is actually generating useful content.
                                logger.info("stream_query first_token=%r", token)
                                first_token = False
                            # `yield` is what makes this an async generator.
                            # Each yield sends one token back to the caller immediately.
                            yield token

                        if data.get("done"):
                            # Ollama signals the end of the response with done=true.
                            logger.info("stream_query done eval_count=%s", data.get("eval_count"))
                            break

            except Exception as exc:
                # Ollama sometimes closes the connection early (TransferEncodingError).
                # We catch it here so whatever tokens already streamed are kept —
                # the user gets a partial answer instead of an error page.
                logger.warning("stream_query connection error (partial response delivered): %s", exc)

        # Return sources up-front (we know them before generation starts)
        # and the generator object. The caller (main.py) will iterate the generator.
        return sources, _token_gen()

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    def list_documents(self) -> List[str]:
        """
        Returns the filenames of all documents currently stored on disk.
        Used by the GET /documents endpoint to show the user what's been uploaded.
        """
        docs = []
        for path in Path(settings.documents_dir).rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.append(path.name)
        return docs

    def collection_count(self) -> int:
        """
        Returns the total number of chunks stored in ChromaDB.
        Used by the /health endpoint so we can confirm documents are indexed.
        A count of 0 means no documents have been ingested yet.
        """
        return self._vectorstore._collection.count()
