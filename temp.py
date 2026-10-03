import os
import shutil
import logging
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.search import RAGSearch
from src.data_loader import load_all_documents
from src.embeddings import EmbeddingPipeline
from src.observability import setup_observability
from src.vector_store import (
    index_documents,
    get_qdrant_client,
    collection_exists,
    DEFAULT_COLLECTION,
)

logger = logging.getLogger("DocuMind.App")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="DocuMind RAG",
    description="Production-grade RAG system with Qdrant Vector DB, Redis Caching, Logfire Observability, and LLM Gateway.",
    version="0.2.0",
)

# Initialize Logfire OpenTelemetry instrumentation
setup_observability(service_name="documind-rag", app=app)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize RAG pipeline
rag_search = RAGSearch()


def reindex_documents() -> dict:
    """Reads all documents from DATA_DIR, chunks them, and indexes them into Qdrant."""
    docs = load_all_documents(DATA_DIR)
    if not docs:
        return {"indexed": False, "message": "No documents found in data folder.", "count": 0}

    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(docs)

    result = index_documents(chunks, collection_name=DEFAULT_COLLECTION, recreate=True)
    # Clear cache on reindexing so outdated answers are never served
    rag_search.cache.clear()
    return result


class SearchRequest(BaseModel):
    query: str
    top_k: int = 2


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health")
async def health_check():
    """Health check endpoint displaying Qdrant, Cache, Observability, and LLM Gateway config."""
    client = get_qdrant_client()
    has_collection = collection_exists(client, DEFAULT_COLLECTION)
    cache_stats = rag_search.cache.get_stats()
    
    return {
        "status": "healthy",
        "vector_db": "Qdrant",
        "collection": DEFAULT_COLLECTION,
        "collection_exists": has_collection,
        "cache": cache_stats,
        "observability": "Logfire",
        "primary_llm": rag_search.gateway.groq_model,
        "fallback_llm": rag_search.gateway.openai_model,
        "guardrails_enabled": True,
    }


@app.get("/api/files")
async def list_files():
    files = []
    for entry in os.scandir(DATA_DIR):
        if entry.is_file():
            files.append({
                "name": entry.name,
                "size": entry.stat().st_size,
            })
    files.sort(key=lambda f: f["name"])
    return {"files": files}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    safe_name = os.path.basename(file.filename)
    file_path = os.path.join(DATA_DIR, safe_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    index_result = reindex_documents()
    return {
        "filename": safe_name,
        "message": f"Uploaded {safe_name} successfully.",
        "index": index_result,
    }


@app.post("/api/search")
async def search(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    top_k = max(1, min(request.top_k, 10))
    result = rag_search.search_and_summarize_with_meta(query, top_k=top_k)

    return {
        "query": query,
        "answer": result["answer"],
        "guardrails": result.get("guardrails", {}),
        "gateway": result.get("gateway", {}),
        "sources": result.get("sources", []),
        "cached": result.get("cached", False),
        "cache_backend": result.get("cache_backend", "none"),
    }


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DATA_DIR, safe_name)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    os.remove(file_path)
    index_result = reindex_documents()
    return {
        "message": f"Deleted {safe_name}.",
        "index": index_result,
    }


@app.post("/api/cache/clear")
async def clear_cache():
    rag_search.cache.clear()
    return {"message": "Cache cleared successfully."}
