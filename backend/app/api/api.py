"""
FastAPI backend for the RAG pipeline project.
Provides endpoints for document upload, document listing, and health checks.
Reuses functions from pipeline.py and agentic_rag.py.

Required pip install packages:
# pip install fastapi uvicorn python-multipart langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf docx2txt groq python-dotenv langgraph
"""

import os
import json
import logging
import time
import uuid
from datetime import datetime
from typing import List, Optional, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv

from pipeline import (
    load_document,
    clean_documents,
    chunk_documents,
    get_embedding_model,
    build_or_load_vectorstore,
)

from agentic_rag import run_agentic_rag

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

# File handler -> app.log
file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# FastAPI app & CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Persistent state (in-memory; survives within a single server process)
# ---------------------------------------------------------------------------

DOCS_REGISTRY_PATH = "documents_registry.json"
SAMPLE_DOCS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
INDEX_PATH = "faiss_index"

# Ensure sample_docs folder exists
os.makedirs(SAMPLE_DOCS_FOLDER, exist_ok=True)

# Load registry at startup
if os.path.exists(DOCS_REGISTRY_PATH):
    with open(DOCS_REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry: List[dict] = json.load(f)
else:
    registry = []
    with open(DOCS_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

# Embedding model and vectorstore – initialise once at startup
_embeddings = get_embedding_model()
_vectorstore = build_or_load_vectorstore([], _embeddings, index_path=INDEX_PATH)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    message: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_timestamp: str
    file_type: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


# Chat request and response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[dict]
    confidence: float
    session_id: str
    retries_used: int


class ChatHistoryEntry(BaseModel):
    question: str
    answer: str
    citations: List[dict]
    timestamp: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _generate_doc_id() -> str:
    """Generate a simple unique document ID."""
    return f"doc_{int(time.time() * 1000)}"


def _save_registry():
    """Persist the current registry to disk."""
    with open(DOCS_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _log_request(endpoint: str, method: str):
    """Log incoming request."""
    logger.info(f"Request: {method} {endpoint}")


def _log_error(error: Exception):
    """Log an error occurrence."""
    logger.error(f"Error: {error}")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    _log_error(exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error: " + str(exc)},
    )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, TXT, or DOCX file, process it through the pipeline,
    and add it to the combined vectorstore."""
    _log_request("/documents/upload", "POST")

    # Validate file type
    allowed_extensions = [".pdf", ".txt", ".docx"]
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""

    if not file_ext or file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: .pdf, .txt, .docx",
        )

    # Validate file is not empty
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    # Save uploaded file to sample_docs folder
    safe_filename = file.filename.replace(" ", "_")
    file_path = os.path.join(SAMPLE_DOCS_FOLDER, safe_filename)

    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        _log_error(e)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Log request details
    logger.info(f"Saved uploaded file: {file_path} (type: {file_ext})")

    # Call pipeline functions to process the new document
    try:
        # 1. Load document
        docs = load_document(file_path)
        logger.info(f"Loaded {len(docs)} document page(s) from '{safe_filename}'")

        # 2. Clean documents
        cleaned_docs = clean_documents(docs)
        logger.info(f"After cleaning: {len(cleaned_docs)} document(s) remain.")

        # 3. Chunk documents
        chunks = chunk_documents(cleaned_docs, chunk_size=500, chunk_overlap=50)
        logger.info(f"Created {len(chunks)} chunks for '{safe_filename}'.")

        # 4. Add chunks to the existing combined vectorstore
        _vectorstore.add_documents(chunks)
        logger.info(
            f"Added {len(chunks)} chunks to the combined vectorstore "
            f"(total vectors: {_vectorstore.index.ntotal})"
        )

        # Save the updated vectorstore
        _vectorstore.save_local(INDEX_PATH)
        logger.info(f"Saved updated combined vectorstore to '{INDEX_PATH}'.")

        # 5. Update registry
        doc_id = _generate_doc_id()
        registry_entry = {
            "id": doc_id,
            "filename": safe_filename,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "file_type": file_ext,
        }
        registry.append(registry_entry)
        _save_registry()
        logger.info(f"Updated documents registry with entry: {registry_entry}")

    except ValueError as ve:
        _log_error(ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        _log_error(e)
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {e}"
        )

    return DocumentUploadResponse(
        id=doc_id,
        filename=safe_filename,
        message="Document uploaded and processed successfully",
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """Return the list of all documents currently tracked in the registry."""
    _log_request("/documents", "GET")
    return DocumentListResponse(documents=registry)


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the registry.

    Looks up the document id in documents_registry.json. If not found, returns 404.
    If found, removes the entry from the registry and deletes the original file
    from the sample_docs folder if it still exists.

    Note: FAISS does not support easy single-document deletion. This endpoint removes
    the document from tracking/registry only; the vector data may remain in the FAISS
    index but will no longer be listed as an active document. This is an acceptable
    simplification for this assignment.
    """
    _log_request(f"/documents/{doc_id}", "DELETE")

    # Find the entry in the registry
    entry_index = None
    for i, entry in enumerate(registry):
        if entry["id"] == doc_id:
            entry_index = i
            break

    if entry_index is None:
        raise HTTPException(status_code=404, detail=f"Document with id '{doc_id}' not found in registry.")

    # Remove from registry
    removed_entry = registry.pop(entry_index)
    _save_registry()
    logger.info(f"Removed document entry from registry: {removed_entry}")

    # Delete the original file from sample_docs if it still exists
    safe_filename = removed_entry.get("filename", "")
    file_path = os.path.join(SAMPLE_DOCS_FOLDER, safe_filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted original file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete file {file_path}: {e}")

    return {"message": f"Document '{doc_id}' deleted successfully."}


@app.get("/health/chat")
async def health_check():
    """Simple health check endpoint."""
    _log_request("/health/chat", "GET")
    return {"status": "ok", "service": "chat"}


# Chat history file path
CHAT_HISTORY_PATH = "chat_history.json"


def _load_chat_history():
    """Load chat history from JSON file."""
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_chat_history(history):
    """Persist chat history to disk."""
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the agentic RAG workflow.

    Accepts a message and optional session_id. If no session_id is provided,
    a new UUID is generated. The message is processed via run_agentic_rag() and
    the exchange is stored in chat_history.json organized by session_id.
    """
    _log_request("/chat", "POST")

    # Generate or use provided session_id
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # Load chat history
    history = _load_chat_history()

    try:
        # Call the existing run_agentic_rag function
        result = run_agentic_rag(request.message)

        # Build the exchange entry
        exchange = {
            "question": request.message,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store in history organized by session_id
        if session_id not in history:
            history[session_id] = []
        history[session_id].append(exchange)
        _save_chat_history(history)

        return ChatResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            confidence=result.get("relevance_score", 0.0),
            session_id=session_id,
            retries_used=result.get("retry_count", 0),
        )

    except Exception as e:
        _log_error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {e}",
        )


@app.get("/chat/history/{session_id}", response_model=List[ChatHistoryEntry])
async def chat_history(session_id: str):
    """Return all chat messages/exchanges for a given session_id."""
    _log_request(f"/chat/history/{session_id}", "GET")

    history = _load_chat_history()

    # Return the entries for this session_id, or empty array if not found
    entries = history.get(session_id, [])
    return entries


# To start the server:
# uvicorn api:app --host 0.0.0.0 --port 8000
# Or with reload: uvicorn api:app --reload --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------------------
# Testing the new endpoints
# ---------------------------------------------------------------------------

# 1. DELETE /documents/{id}
#   - Run the server: uvicorn api:app --host 0.0.0.0 --port 8000
#   - First upload a document via POST /documents/upload
#   - Note the document id from the response or from GET /documents
#   - Delete it: DELETE /documents/{id} (replace {id} with the document id)
#   - Verify it's removed via GET /documents
#   - To also delete the physical file, ensure the file name matches what was uploaded

# 2. POST /chat
#   - Run the server and upload a document first
#   - Send a POST request to /chat with JSON body:
#     {
#       "message": "Your question here",
#       "session_id": "optional-session-id"  # omit to auto-generate a UUID
#     }
#   - Response includes: answer, citations, confidence, session_id, retries_used
#   - Chat history is saved to chat_history.json organized by session_id

# 3. GET /chat/history/{session_id}
#   - Retrieve chat history for a specific session:
#   GET /chat/history/{session_id} (replace {session_id} with the actual id)
#   - Returns a JSON array of exchanges (question, answer, citations, timestamp)
#   - Returns an empty array if no history exists for that session_id

# Full API documentation is available at http://localhost:8000/docs after starting the server