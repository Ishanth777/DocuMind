"""
DocuMind Qdrant Vector Store Module
Replaces Typesense with Qdrant for vector storage and semantic retrieval.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
logger = logging.getLogger("DocuMind.VectorStore")
logging.basicConfig(level=logging.INFO)

DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "documind_docs")
STORAGE_DIR = os.getenv("QDRANT_STORAGE_DIR", str(Path(__file__).parent.parent / "qdrant_storage"))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "384"))

_client_instance: Optional[QdrantClient] = None
_embedding_instance: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_instance
    if _embedding_instance is None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"[VectorStore] Initializing embedding model: {model_name}")
        try:
            _embedding_instance = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            _embedding_instance = HuggingFaceEmbeddings(model_name=model_name)
    return _embedding_instance


def get_qdrant_client() -> QdrantClient:
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if url and url.strip():
        logger.info(f"[VectorStore] Connecting to remote Qdrant at: {url.strip()}")
        _client_instance = QdrantClient(url=url.strip(), api_key=api_key.strip() if api_key else None, timeout=15)
    else:
        os.makedirs(STORAGE_DIR, exist_ok=True)
        logger.info(f"[VectorStore] Initializing local Qdrant at: {STORAGE_DIR}")
        _client_instance = QdrantClient(path=STORAGE_DIR)
    return _client_instance


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        return client.collection_exists(collection_name)
    except Exception as e:
        logger.error(f"[VectorStore] Error checking collection: {e}")
        return False


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = VECTOR_DIMENSION):
    if not collection_exists(client, collection_name):
        logger.info(f"[VectorStore] Creating collection '{collection_name}' (dim={vector_size})...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE),
        )


def get_vector_store(collection_name: str = DEFAULT_COLLECTION) -> QdrantVectorStore:
    client = get_qdrant_client()
    embeddings = get_embedding_model()
    ensure_collection(client, collection_name)
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)


def index_documents(docs: List[Document], collection_name: str = DEFAULT_COLLECTION, recreate: bool = False) -> dict:
    if not docs:
        return {"indexed": False, "message": "No documents provided.", "count": 0}
    client = get_qdrant_client()
    if recreate and collection_exists(client, collection_name):
        logger.info(f"[VectorStore] Deleting collection '{collection_name}' for reindex...")
        client.delete_collection(collection_name)
    store = get_vector_store(collection_name)
    logger.info(f"[VectorStore] Indexing {len(docs)} chunks into '{collection_name}'...")
    store.add_documents(documents=docs)
    logger.info(f"[VectorStore] Indexed {len(docs)} chunks successfully.")
    return {"indexed": True, "message": f"Indexed {len(docs)} document chunks.", "count": len(docs)}


def similarity_search(query: str, top_k: int = 2, collection_name: str = DEFAULT_COLLECTION) -> List[Document]:
    client = get_qdrant_client()
    if not collection_exists(client, collection_name):
        logger.warning(f"[VectorStore] Collection '{collection_name}' does not exist.")
        return []
    store = get_vector_store(collection_name)
    return store.similarity_search(query, k=top_k)


def vector_store_example(query: str, top_k: int = 2) -> List[Document]:
    client = get_qdrant_client()
    collection_name = DEFAULT_COLLECTION
    if not collection_exists(client, collection_name):
        logger.info(f"[VectorStore] Collection not found. Indexing from data/...")
        from src.data_loader import load_all_documents
        from src.embeddings import EmbeddingPipeline
        docs = load_all_documents("data")
        if docs:
            pipeline = EmbeddingPipeline()
            chunks = pipeline.chunk_documents(docs)
            index_documents(chunks, collection_name=collection_name, recreate=True)
        else:
            logger.warning("[VectorStore] No documents found in data/.")
            return []
    return similarity_search(query, top_k=top_k, collection_name=collection_name)
