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
#   1. User uploads a PDF/DOCX/TXT  →  ingest_file() splits it into chunks
#      and stores vector embeddings in ChromaDB.
#   2. User asks a question  →  retriever finds the most similar chunks.
#   3. Those chunks are injected into the LLM prompt as "context".
#   4. The LLM (phi3.5 via Ollama) reads the context and writes an answer.
#
# ─────────────────────────────────────────────────────────────────────────────
# PYTHON BASICS REFRESHER (read this before the code)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IS A FUNCTION?
#   A function is a reusable block of code with a name. You write the logic
#   once and can call (run) it as many times as you want from anywhere.
#
#   def greet(name):          ← define it
#       return "Hello " + name
#
#   greet("Prajjumna")        ← call it → returns "Hello Prajjumna"
#   greet("World")            ← call it again → returns "Hello World"
#
#   Without functions you'd copy-paste the same code everywhere and fixing
#   a bug would mean changing it in 20 places.
#
# WHAT IS A CLASS?
#   A class is a blueprint for creating objects that bundle related data
#   (attributes) and functions (methods) together.
#
#   class Car:
#       def __init__(self, brand):   ← runs when you create an object
#           self.brand = brand       ← store data on the object
#       def drive(self):             ← a method (function on an object)
#           print(self.brand + " is moving")
#
#   my_car = Car("Toyota")   ← create an object (instance) from the blueprint
#   my_car.drive()           ← call a method → "Toyota is moving"
#
#   `self` always refers to the specific object the method is called on.
#   RAGEngine is a class — `rag = RAGEngine()` creates one object that holds
#   all the components (ChromaDB, LLM, embeddings) for the lifetime of the app.
#
# WHAT IS `async` / `await`?
#   Normal Python code is sequential: line 1 runs, finishes, then line 2 runs.
#   If line 1 is "wait for the LLM to respond" that could take 30 seconds —
#   during that time your program is frozen and can't serve other users.
#
#   `async def` marks a function as asynchronous — it can pause and let other
#   code run while waiting for slow I/O (network calls, disk reads).
#   `await` is the pause point: "start this, let other things run, come back
#   when it's done."
#
#   async def get_answer():
#       result = await ask_ollama("hello")  ← pause here, don't block
#       return result
#
#   FastAPI is built entirely on async — every request is handled concurrently.
#
# WHAT IS `yield` / A GENERATOR?
#   A normal function runs to completion and returns ONE value with `return`.
#   A generator function uses `yield` to produce a SEQUENCE of values one at
#   a time, pausing between each one.
#
#   def count_up():
#       yield 1    ← produce 1, pause
#       yield 2    ← produce 2, pause
#       yield 3    ← produce 3, done
#
#   for n in count_up():   → prints 1, 2, 3 one at a time
#       print(n)
#
#   We use `async def` + `yield` for streaming tokens: each token from Ollama
#   is yielded immediately to the browser without waiting for the full answer.
#
# WHAT IS `self`?
#   Inside a class method, `self` is the object itself. It lets you access
#   the object's data and other methods.
#   self._llm means "the _llm attribute stored on THIS RAGEngine instance".
#   The underscore prefix (_llm, _retriever) is a Python convention meaning
#   "internal — don't touch this from outside the class".
#
# WHAT IS A TYPE HINT?
#   Python doesn't enforce types at runtime, but you can annotate them for
#   readability and IDE autocomplete.
#   def ingest_file(self, file_path: str) -> int:
#   means: ingest_file takes a string and returns an int. Not enforced, but
#   documents the intent.
# =============================================================================

import asyncio        # Built-in async task scheduling — run_in_executor, create_task
import hashlib        # Produce a consistent fingerprint (MD5) from a string
import json           # Convert between Python dicts and JSON strings
import os             # OS operations: create directories, read environment variables
import logging        # Structured logging — messages appear in `kubectl logs`
from pathlib import Path                        # Clean file-path manipulation
from typing import AsyncGenerator, List, Tuple  # Type hints for complex return types

import httpx  # Async HTTP client — used to stream tokens directly from Ollama's REST API

# ─────────────────────────────────────────────────────────────────────────────
# LangChain imports
# LangChain is a framework that provides ready-made building blocks for LLM apps.
# Think of it like Lego pieces: splitter, embeddings, vector store, chain.
# ─────────────────────────────────────────────────────────────────────────────

from langchain.text_splitter import RecursiveCharacterTextSplitter
# Splits a long document into smaller overlapping pieces (chunks).
# "Recursive" = tries paragraph breaks → sentence breaks → word breaks,
# always preferring natural language boundaries over arbitrary character counts.

from langchain_community.document_loaders import (
    PyPDFLoader,     # Extracts text + page numbers from PDF files
    Docx2txtLoader,  # Extracts text from Word .docx / .doc files
    TextLoader,      # Reads plain .txt files with a specified encoding
)

from langchain_community.embeddings import FastEmbedEmbeddings
# Runs a small local ONNX model (BAAI/bge-small-en-v1.5, ~100MB) to convert
# text into a list of ~384 numbers called a "vector" or "embedding".
# Vectors capture MEANING — "dog" and "puppy" will have similar vectors.
# Runs fully offline inside the pod; no API key needed.

from langchain_chroma import Chroma
# ChromaDB is our vector database.
# It stores (text, vector, metadata) for every chunk so we can later search
# for the chunks most similar to a user's question using vector math.

from langchain_ollama import OllamaLLM
# A LangChain wrapper that lets us use Ollama as the LLM backend.
# Used in the non-streaming /query path via the RetrievalQA chain.

from langchain.chains import RetrievalQA
# A pre-built LangChain "chain" — a fixed sequence of steps:
#   1. Retriever fetches relevant chunks from ChromaDB.
#   2. PromptTemplate inserts those chunks into the prompt.
#   3. LLM reads the filled prompt and generates an answer.

from langchain.prompts import PromptTemplate
# A string template with {placeholders} that LangChain fills in at query time.
# Separating the template from the code makes it easy to tune without touching logic.

from config import settings  # Our app's settings loaded from environment variables

# logging.getLogger(__name__) creates a logger named after this file ("rag_engine").
# Every logger.info("...") call writes a timestamped line to stdout,
# visible via `kubectl logs -n ai-stack deploy/rag-api`.
logger = logging.getLogger(__name__)

# A Python SET of file extensions we can handle.
# Sets use {} and are unordered — optimised for fast "is X in this set?" checks.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# Triple-quoted strings (""") span multiple lines.
# {context} and {question} are placeholders — str.format() replaces them later.
# This is the exact text sent to the LLM before every answer.
PROMPT_TEMPLATE = """Use the context below to answer the question. Be direct and helpful. Summarize the relevant information clearly.

Context:
{context}

Question: {question}
Answer:"""


# =============================================================================
# FUNCTION: _get_loader
# =============================================================================
# A FUNCTION is a named, reusable block of code.
# `def` starts the definition. `_get_loader` is the name.
# `file_path: str` is a parameter (input). `: str` is a type hint (it's a string).
# The function runs when you CALL it: _get_loader("/data/docs/resume.pdf")
#
# The leading underscore (_) means "private helper — only used inside this file".
# =============================================================================

def _get_loader(file_path: str):
    """
    Given a file path, return the correct LangChain loader for that file type.

    Why do we need different loaders?
      PDFs are binary files with internal page structure.
      DOCX files are ZIP archives of XML — you can't just read the bytes.
      TXT files are plain text — the simplest case.
    Each loader knows the internal format of its file type and extracts clean text.

    Returns None for unsupported types — the caller decides what to do.
    """
    # Path(file_path).suffix gives the file extension including the dot.
    # .lower() normalises "PDF", "pdf", "Pdf" → all become ".pdf".
    ext = Path(file_path).suffix.lower()

    # if / elif / else: check conditions in order, run the first matching block.
    if ext == ".pdf":
        return PyPDFLoader(file_path)      # Returns a loader object, not text yet
    elif ext in {".docx", ".doc"}:
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")

    return None  # None is Python's "nothing" value — means no loader found


# =============================================================================
# CLASS: RAGEngine
# =============================================================================
# A CLASS is a blueprint for objects. `class RAGEngine:` defines the blueprint.
# `rag = RAGEngine()` creates one concrete object from that blueprint.
#
# Why a class instead of plain functions?
#   We need to keep several expensive objects alive for the whole app lifetime:
#   the embedding model, the ChromaDB connection, the HTTP client, the LLM.
#   A class groups them together under one variable (rag) and lets every
#   method share them via `self`.
#
# All methods (functions inside a class) receive `self` as their first argument.
# `self` is the object itself — it's how methods access each other's data.
# =============================================================================

class RAGEngine:

    # =========================================================================
    # METHOD: __init__  (constructor)
    # =========================================================================
    # __init__ is a SPECIAL METHOD — Python calls it automatically when you
    # write `RAGEngine()`. It runs once at startup to set everything up.
    # Think of it as the wiring phase: connect all the components before
    # any user traffic arrives.
    #
    # Everything assigned to `self.something` here persists for the entire
    # lifetime of the object and is accessible from every other method.
    # =========================================================================

    def __init__(self):
        """One-time setup: creates directories, connects to ChromaDB, loads models."""

        # os.makedirs creates a folder (and any missing parent folders).
        # exist_ok=True: don't raise an error if the folder already exists.
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        os.makedirs(settings.documents_dir, exist_ok=True)

        # ── HTTP Client ──────────────────────────────────────────────────────
        # httpx.AsyncClient is a persistent HTTP connection pool.
        #
        # WHY PERSISTENT?
        # Opening a TCP connection involves a "handshake" (~100ms).
        # If we created a new client for every Ollama request, we'd waste 100ms
        # every single time. A persistent client reuses the same connection.
        #
        # timeout=600s: LLM responses on CPU can be very slow — don't give up.
        # max_keepalive_connections=2: keep at most 2 idle connections warm.
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=5),
        )

        # ── Embeddings Model ─────────────────────────────────────────────────
        # This loads a ~100MB ONNX model into RAM once.
        # Every time we ingest a chunk or search for similar chunks, this model
        # converts the text → a list of 384 numbers (the embedding vector).
        # cache_dir tells it where to store the model files on disk.
        self._embeddings = FastEmbedEmbeddings(
            model_name=settings.embedding_model,
            cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
        )

        # ── Vector Store (ChromaDB) ──────────────────────────────────────────
        # ChromaDB opens (or creates) a database at persist_directory.
        # All previously ingested chunks are loaded from disk automatically.
        #
        # persist_directory: the folder on the hostPath volume (/home/pradyumna/rag/chroma)
        # embedding_function: ChromaDB calls _embeddings.embed() internally
        #                     when you add or query documents.
        # collection_name: like a table name — groups related documents together.
        self._vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self._embeddings,
            collection_name="rag_documents",
        )

        # ── Text Splitter ────────────────────────────────────────────────────
        # We cannot feed an entire PDF to the LLM — its context window has limits.
        # This splitter cuts documents into small overlapping pieces.
        #
        # chunk_size=300:   max characters per chunk (~60-70 words)
        # chunk_overlap=50: last 50 chars of chunk N also appear at the start
        #                   of chunk N+1, so no sentence is cut off at a boundary
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # ── LLM (Language Model via Ollama) ──────────────────────────────────
        # OllamaLLM wraps the HTTP calls to Ollama's /api/generate endpoint.
        # This is used ONLY for the non-streaming /query path through LangChain.
        # The streaming path calls Ollama directly via httpx (see astream_query).
        #
        # temperature=0.1: near-zero randomness → consistent, factual answers
        # num_ctx=2048:     how many tokens the model can "see" at once
        # num_predict=200:  stop after 200 output tokens (~150 words) to keep it fast
        self._llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,
            num_ctx=2048,
            num_predict=200,
        )

        # ── Retriever ────────────────────────────────────────────────────────
        # as_retriever() wraps ChromaDB in LangChain's retriever interface.
        # When called with a question, it embeds the question and finds the
        # k most similar chunks using cosine similarity.
        #
        # search_type="similarity": compare vectors using cosine distance
        # k=retriever_top_k:        how many chunks to return (currently 3)
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retriever_top_k},
        )

        # Build the LangChain QA chain once and reuse it for every /query call.
        # _build_chain() is another method on this class — called via self.
        self._qa_chain = self._build_chain()

    # =========================================================================
    # METHOD: aclose  (async)
    # =========================================================================
    # `async def` = this is an asynchronous method. You must `await` it.
    # Called by FastAPI's lifespan teardown when the app is shutting down.
    # =========================================================================

    async def aclose(self) -> None:
        """
        Gracefully close the persistent HTTP client on shutdown.

        `-> None` means this function returns nothing (like `void` in Java/C).
        Without this, the OS would see open TCP connections and wait for them
        to time out (up to several minutes) before the pod fully terminates.
        """
        await self._http_client.aclose()

    # =========================================================================
    # METHOD: prewarm  (async)
    # =========================================================================
    # `async def` = must be awaited. Uses `await` internally when making the
    # HTTP call to Ollama so it doesn't block the event loop.
    # =========================================================================

    async def prewarm(self) -> None:
        """
        Sends a 1-token dummy request to Ollama at startup.

        WHY?
        Ollama only loads a model into RAM when the FIRST request arrives.
        Loading phi3.5 (2.2GB) takes 10–20 seconds. Without pre-warming, the
        first real user request would time out or feel very slow.
        Pre-warming pays that cost at startup, invisibly to the user.

        try / except:
          `try` runs the code. If any error occurs, Python jumps to `except`.
          We use it here because if Ollama isn't ready yet (still starting up),
          we don't want to crash the whole app — just log a warning and move on.
        """
        try:
            url = settings.ollama_base_url.rstrip("/") + "/api/generate"
            # await pauses THIS coroutine until the HTTP response comes back,
            # but lets other FastAPI requests run in the meantime.
            await self._http_client.post(url, json={
                "model": settings.ollama_model,
                "prompt": "hi",
                "stream": False,
                "options": {
                    "num_predict": 1,    # Generate only 1 token — just enough to load the model
                    "num_ctx": 2048,     # Use the same context size as real queries so Ollama
                                         # doesn't have to reload the model later
                },
            })
            logger.info("Model pre-warm complete")
        except Exception as exc:
            # `exc` holds the error object. `%s` formats it as a string in the log.
            logger.warning("Model pre-warm failed (non-fatal): %s", exc)

    # =========================================================================
    # METHOD: _build_chain  (private helper)
    # =========================================================================
    # Regular `def` (not async) — runs synchronously and returns immediately.
    # Called once in __init__, result stored in self._qa_chain.
    # Return type hint `-> RetrievalQA` documents what the function returns.
    # =========================================================================

    def _build_chain(self) -> RetrievalQA:
        """
        Builds and returns the LangChain RetrievalQA chain.

        WHAT IS A CHAIN?
        A chain is a pipeline where the output of one step becomes the input
        of the next. This chain has three steps:
          Step 1 — self._retriever  → takes a question, returns top-k chunks
          Step 2 — PromptTemplate   → inserts chunks into the prompt template
          Step 3 — self._llm        → reads the filled prompt, returns an answer

        chain_type="stuff": all chunks are "stuffed" into one prompt string.
        This is the simplest and fastest strategy — works well for small chunks.

        return_source_documents=True: the chain also returns WHICH chunks it used
        so we can display "Sources: resume.pdf, page 2" to the user.
        """
        # PromptTemplate compiles the PROMPT_TEMPLATE string and registers
        # which variable names ({context}, {question}) need to be filled in.
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        # from_chain_type is a class method (factory) that constructs the chain.
        # We pass all three components: LLM, retriever, and prompt.
        return RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=self._retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    # =========================================================================
    # METHOD: ingest_file
    # =========================================================================
    # Regular synchronous method. Blocking — takes several seconds for large
    # files. Called in a background thread (run_in_executor) from async code
    # so it doesn't freeze the event loop.
    # =========================================================================

    def ingest_file(self, file_path: str) -> int:
        """
        Full ingestion pipeline for one file. Returns the number of chunks stored.

        Steps:
          1. Pick the right loader for the file type.
          2. Load raw text from the file.
          3. Split text into small overlapping chunks.
          4. Delete any existing chunks for this file (handles re-uploads).
          5. Generate deterministic IDs for each chunk.
          6. Embed and store all chunks in ChromaDB.

        IDEMPOTENT means: safe to call multiple times with the same input —
        the result is always the same, no duplicates created.
        """
        # Call the helper function defined at the top of this file.
        # If it returns None the file type is unsupported → raise an error.
        loader = _get_loader(file_path)
        if loader is None:
            # `raise` throws an exception that stops execution and bubbles up
            # to the caller, which can catch it with try/except.
            # ValueError is a built-in Python exception for invalid input.
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

        # loader.load() reads the file and returns a Python LIST of LangChain
        # Document objects. Each Document has two fields:
        #   .page_content  — the extracted text
        #   .metadata      — dict with info like {"source": "/data/...", "page": 0}
        docs = loader.load()

        # Split each Document into chunks.
        # Input:  [Document(page_content="very long text...")]
        # Output: [Document(page_content="chunk 1..."),
        #          Document(page_content="chunk 2..."), ...]
        chunks = self._splitter.split_documents(docs)

        # DELETE old chunks for this file before adding new ones.
        # WHY? If we re-upload a file, we'd get duplicate entries in ChromaDB
        # without this step. The "source" metadata field is how we identify
        # which chunks belong to which file.
        # try/except: if no chunks exist yet for this file, delete is a no-op.
        try:
            self._vectorstore._collection.delete(where={"source": file_path})
        except Exception:
            pass  # `pass` means "do nothing" — ignore the error and continue

        # DETERMINISTIC IDs: generate stable, predictable IDs for every chunk.
        # hashlib.md5(file_path.encode()).hexdigest() produces a 32-char hex
        # string like "a1b2c3..." that is ALWAYS the same for the same file path.
        # We append _0, _1, _2... for each chunk index.
        # Result: ["a1b2c3_0", "a1b2c3_1", "a1b2c3_2", ...]
        #
        # WHY DETERMINISTIC?
        # ChromaDB upserts when the same ID is inserted twice (update, not duplicate).
        # Random IDs would create duplicates every time the file is re-ingested.
        base = hashlib.md5(file_path.encode()).hexdigest()
        # List comprehension: a compact way to build a list.
        # [f"{base}_{i}" for i in range(len(chunks))] is the same as:
        #   ids = []
        #   for i in range(len(chunks)):
        #       ids.append(f"{base}_{i}")
        ids = [f"{base}_{i}" for i in range(len(chunks))]

        # add_documents stores each chunk: it calls the embedding model on each
        # chunk's text, then saves (text, vector, metadata, id) to ChromaDB.
        self._vectorstore.add_documents(chunks, ids=ids)

        # %d and %s are format placeholders for the logger — faster than f-strings.
        logger.info("Ingested %d chunks from %s", len(chunks), file_path)

        # `return` sends a value back to whoever called this function.
        # The caller (ingest_directory, background task) uses it to report results.
        return len(chunks)

    # =========================================================================
    # METHOD: ingest_directory
    # =========================================================================
    # Loops over every file in a folder and calls ingest_file() on each one.
    # Returns a dict so the caller knows which files succeeded or failed.
    # =========================================================================

    def ingest_directory(self, directory: str) -> dict:
        """
        Ingest every supported file found anywhere inside `directory`.

        rglob("*") walks the entire folder tree recursively (like `find` in bash).
        Returns: {"resume.pdf": {"chunks": 31, "status": "ok"}, ...}
        """
        # {} creates an empty Python DICTIONARY — key/value pairs.
        results = {}

        for path in Path(directory).rglob("*"):
            # path.suffix gives the extension, .lower() normalises case
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    # str(path) converts the Path object to a plain string
                    count = self.ingest_file(str(path))
                    # path.name gives just the filename, not the full path
                    results[path.name] = {"chunks": count, "status": "ok"}
                except Exception as exc:
                    logger.error("Failed to ingest %s: %s", path, exc)
                    results[path.name] = {"status": "error", "error": str(exc)}

        return results

    # =========================================================================
    # METHOD: query  (synchronous)
    # =========================================================================
    # Regular `def` — blocks until the LLM finishes generating the full answer.
    # Used by the /query REST endpoint. In async contexts it must be wrapped
    # with run_in_executor to avoid freezing the event loop.
    # =========================================================================

    def query(self, question: str) -> dict:
        """
        Ask a question and get a complete answer (no streaming).

        Internally runs the full chain: retriever → prompt → LLM → answer.
        Returns a dict: {"answer": "...", "sources": [...]}
        """
        # .invoke() runs the chain end-to-end.
        # Input: {"query": "who is prajjumna"}
        # Output: {"result": "Prajjumna is a DevOps engineer...",
        #          "source_documents": [Document(...), ...]}
        result = self._qa_chain.invoke({"query": question})

        # Deduplicate sources — the retriever may return 3 chunks all from
        # page 1 of the same PDF. We only want to show that page once.
        # A SET automatically ignores duplicate values — we use it as a "seen" tracker.
        seen: set = set()
        unique_sources: list = []

        for doc in result.get("source_documents", []):
            # .get("key", default) safely reads from a dict — returns default if missing
            src = (doc.metadata.get("source", ""), doc.metadata.get("page"))
            if src not in seen:
                seen.add(src)   # Mark this (file, page) pair as already processed
                unique_sources.append({"source": src[0], "page": src[1]})

        # Return a plain Python dict — FastAPI serialises it to JSON automatically.
        return {"answer": result["result"], "sources": unique_sources}

    # =========================================================================
    # METHOD: astream_query  (async)
    # =========================================================================
    # `async def` means you must `await` it. It returns TWO things as a Tuple:
    #   1. sources  — list of source dicts (known before generation starts)
    #   2. _token_gen — an async generator that yields tokens one at a time
    #
    # Tuple[list, AsyncGenerator[str, None]] is the type hint:
    #   Tuple = a fixed-length collection of values (like an immutable list)
    #   AsyncGenerator[str, None] = an async generator that yields strings
    # =========================================================================

    async def astream_query(self, question: str) -> Tuple[list, AsyncGenerator[str, None]]:
        """
        Ask a question and stream the answer token by token.

        WHY STREAMING?
        phi3.5 on CPU takes 20–60 seconds to generate a full answer.
        Without streaming, the user sees a blank screen for a minute.
        With streaming, words appear one at a time as they're generated —
        much better UX (same as how ChatGPT works).

        HOW IT WORKS:
          1. Retriever finds relevant chunks (blocking → run in thread pool).
          2. We build the prompt string manually.
          3. We return (sources, async_generator).
          4. The caller (main.py) iterates the generator and forwards each
             token to the browser as a Server-Sent Event (SSE).
        """

        # ── Step 1: Retrieve relevant chunks ─────────────────────────────────
        # self._retriever.invoke() is a SYNCHRONOUS (blocking) function.
        # Calling it directly inside an async function would FREEZE the event
        # loop — no other requests could be served while it runs.
        #
        # SOLUTION: run_in_executor runs it in a background thread pool.
        # `await` pauses this coroutine until the thread finishes, but the
        # event loop is free to handle other requests during that time.
        #
        # asyncio.get_running_loop() returns the current event loop object.
        # None as the first argument means "use the default thread pool".
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, self._retriever.invoke, question)

        # ── Step 2: Build the prompt ──────────────────────────────────────────
        # For each retrieved Document, collapse internal whitespace and join them.
        # " ".join(d.page_content.split()) replaces any run of whitespace
        # (tabs, newlines, multiple spaces) with a single space — saves tokens.
        context = "\n\n".join(" ".join(d.page_content.split()) for d in docs)

        # str.format() replaces {context} and {question} in the template string.
        prompt_text = PROMPT_TEMPLATE.format(context=context, question=question)

        # ── Step 3: Collect sources ───────────────────────────────────────────
        # We know the sources NOW (before generation starts), so we collect them
        # here and return them alongside the generator.
        seen: set = set()
        sources: list = []
        for d in docs:
            src = (d.metadata.get("source", ""), d.metadata.get("page"))
            if src not in seen:
                seen.add(src)
                sources.append({"source": src[0], "page": src[1]})

        # ── Step 4: Define the token streaming generator ─────────────────────
        # A NESTED FUNCTION is a function defined inside another function.
        # _token_gen is defined inside astream_query so it can automatically
        # ACCESS prompt_text and sources from the outer scope — this is called
        # a "closure". No need to pass them as arguments.
        #
        # `async def` + `yield` = ASYNC GENERATOR.
        # An async generator is a function that yields values asynchronously.
        # Each `yield token` pauses the function and hands a token to the caller.
        # The caller does `async for token in _token_gen()` to receive them.

        async def _token_gen() -> AsyncGenerator[str, None]:
            """
            Streams raw tokens from Ollama's /api/generate endpoint.

            Ollama sends newline-delimited JSON when stream=True:
              {"response": "Prajjumna", "done": false}
              {"response": " is",       "done": false}
              {"response": " a",        "done": false}
              {"response": "",          "done": true, "eval_count": 17}

            We yield each "response" value immediately so the browser sees
            words appear in real time.
            """
            url = settings.ollama_base_url.rstrip("/") + "/api/generate"
            # A dict literal {} creates a Python dictionary.
            payload = {
                "model": settings.ollama_model,
                "prompt": prompt_text,
                "stream": True,         # Ask Ollama to send tokens as they're generated
                "options": {
                    "temperature": 0.1,     # Low randomness = consistent factual answers
                    "num_ctx": 2048,        # Context window in tokens
                    "num_predict": 200,     # Stop after 200 output tokens (~150 words)
                    "num_batch": 512,       # Process 512 tokens at once (CPU throughput)
                },
            }
            logger.info("stream_query model=%s prompt_len=%d", settings.ollama_model, len(prompt_text))
            first_token = True  # Boolean flag to log only the very first token

            # try/except: catches errors during streaming (e.g. Ollama drops the
            # connection before sending "done"). Without this, a mid-stream error
            # would crash the whole generator and show an error page instead of
            # delivering the partial answer the user already received.
            try:
                # `async with` is the async version of `with`.
                # self._http_client.stream() opens an HTTP connection that stays
                # OPEN while we read the response line by line.
                # `as resp` binds the response object to the name `resp`.
                async with self._http_client.stream("POST", url, json=payload) as resp:

                    # `async for` iterates an async iterable — each iteration
                    # awaits the next value. resp.aiter_lines() reads one line
                    # at a time from the HTTP response body as it arrives.
                    async for line in resp.aiter_lines():

                        if not line:
                            continue  # `continue` skips to the next loop iteration

                        # json.loads() parses a JSON string into a Python dict.
                        # If the line isn't valid JSON, skip it.
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # .get("key", default) reads from the dict safely.
                        # If "response" key is missing, return "" (empty string).
                        token = data.get("response", "")

                        if token:
                            if first_token:
                                logger.info("stream_query first_token=%r", token)
                                first_token = False

                            # ── `yield` ──────────────────────────────────────
                            # This is what makes _token_gen an ASYNC GENERATOR.
                            # `yield token` immediately sends this token to whoever
                            # is iterating the generator (`async for token in gen`),
                            # then PAUSES here until the caller asks for the next value.
                            # No return statement — the function can yield many times.
                            yield token

                        if data.get("done"):
                            logger.info("stream_query done eval_count=%s", data.get("eval_count"))
                            break  # `break` exits the for loop immediately

            except Exception as exc:
                # Log the error but don't re-raise it.
                # The generator simply stops, and whatever tokens were already
                # yielded are kept — the user sees a partial answer.
                logger.warning("stream_query connection error (partial response delivered): %s", exc)

        # Return both values as a TUPLE.
        # The () around two values creates a tuple: (sources, _token_gen())
        # Note: _token_gen() creates the generator object but does NOT start
        # running it yet — it only starts when the caller does `async for`.
        return sources, _token_gen()

    # =========================================================================
    # METHOD: list_documents
    # =========================================================================

    def list_documents(self) -> List[str]:
        """
        Returns filenames of all documents on disk (not from ChromaDB).
        List[str] type hint means: returns a list where every item is a string.
        """
        docs = []  # Start with an empty list
        for path in Path(settings.documents_dir).rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.append(path.name)  # .append() adds an item to the end of a list
        return docs

    # =========================================================================
    # METHOD: collection_count
    # =========================================================================

    def collection_count(self) -> int:
        """
        Returns the number of chunks currently stored in ChromaDB.
        `-> int` means this returns a whole number (integer).
        0 = no documents indexed yet.
        Used by /health and the startup ingest check.
        """
        return self._vectorstore._collection.count()
