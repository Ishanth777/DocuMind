from src.data_loader import load_all_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Typesense

import typesense

def collection_exists(client, collection_name):
    try:
        client.collections[collection_name].retrieve()
        return True
    except Exception:
        return False

def vector_store_example(query):
    client = typesense.Client({
        "nodes": [{
            "host": "ustvlr6epf5jmh3yp-1.a1.typesense.net",
            "port": "443",
            "protocol": "https"
        }],
        "api_key": "f6tWPrTVl1N0iZBlBg7F8BlrDvZXLa1B",
        "connection_timeout_seconds": 2
    })

    collection_name = "rag2"

   
    if not collection_exists(client, collection_name):
        print("Creating collection and indexing...")

        docs = load_all_documents("data")
        embeddings = HuggingFaceEmbeddings()

        docsearch = Typesense.from_documents(
            docs,
            embeddings,
            typesense_client_params={
                "host": "ustvlr6epf5jmh3yp-1.a1.typesense.net",
                "port": "443",
                "protocol": "https",
                "typesense_api_key": "f6tWPrTVl1N0iZBlBg7F8BlrDvZXLa1B",
                "typesense_collection_name": collection_name
            },
        )

        return docsearch.similarity_search(query)

    else:
        print("Using existing collection...")
        client = typesense.Client({
            "nodes": [{
                "host": "ustvlr6epf5jmh3yp-1.a1.typesense.net",
                "port": "443",
                "protocol": "https"
            }],
            "api_key": "f6tWPrTVl1N0iZBlBg7F8BlrDvZXLa1B",
            "connection_timeout_seconds": 2
        })

        store = Typesense(
            typesense_client=client,
            embedding=HuggingFaceEmbeddings(),
            typesense_collection_name=collection_name
        )

        return store.similarity_search(query)