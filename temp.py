from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import os
import typesense
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Typesense as TypesenseVectorStore

from src.search import RAGSearch
from src.data_loader import load_all_documents

app = FastAPI(title="DocuMind RAG")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

COLLECTION_NAME = "rag2"
TYPESENSE_CONFIG = {
    "nodes": [{
        "host": "ustvlr6epf5jmh3yp-1.a1.typesense.net",
        "port": "443",
        "protocol": "https",
    }],
    "api_key": "f6tWPrTVl1N0iZBlBg7F8BlrDvZXLa1B",
    "connection_timeout_seconds": 2,
}

rag_search = RAGSearch()


def get_typesense_client():
    return typesense.Client(TYPESENSE_CONFIG)


def collection_exists(client, collection_name: str) -> bool:
    try:
        client.collections[collection_name].retrieve()
        return True
    except Exception:
        return False


def reindex_documents() -> dict:
    client = get_typesense_client()

    if collection_exists(client, COLLECTION_NAME):
        client.collections[COLLECTION_NAME].delete()

    docs = load_all_documents(DATA_DIR)
    if not docs:
        return {"indexed": False, "message": "No documents found in data folder.", "count": 0}

    embeddings = HuggingFaceEmbeddings()
    TypesenseVectorStore.from_documents(
        docs,
        embeddings,
        typesense_client_params={
            "host": TYPESENSE_CONFIG["nodes"][0]["host"],
            "port": TYPESENSE_CONFIG["nodes"][0]["port"],
            "protocol": TYPESENSE_CONFIG["nodes"][0]["protocol"],
            "typesense_api_key": TYPESENSE_CONFIG["api_key"],
            "typesense_collection_name": COLLECTION_NAME,
        },
    )
    return {"indexed": True, "message": f"Indexed {len(docs)} document chunks.", "count": len(docs)}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 2


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


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
    answer = rag_search.search_and_summarize(query, top_k=top_k)
    return {"query": query, "answer": answer}


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
