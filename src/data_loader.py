from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader


def load_all_documents(data_dir: str) -> List[Any]:
    data_path = Path(__file__).parent.parent / data_dir
    data_path = data_path.resolve()

    print(f"[DEBUG] Data path: {data_path}")
    print(f"[DEBUG] Exists: {data_path.exists()}")
    print(f"[DEBUG] Contents: {list(data_path.glob('*'))}")

    documents = []

    pdf_files = list(data_path.glob('**/*.[pP][dD][fF]'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# if __name__ == "__main__":
#     docs = load_all_documents("new_data")
#     print(f"Loaded {len(docs)} documents.")
#     print("Example document:", docs[0] if docs else None)