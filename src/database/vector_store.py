from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from config import settings


def setup_vector_db(
    chunks,
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str | None = None,
):
    """Initializes ChromaDB and indexes chunks only if the collection is empty."""
    persist_dir = persist_directory or settings.CHROMA_PERSIST_DIR
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Check if the collection already has data to avoid duplicates
    existing_data = vector_db.get()
    if not existing_data["ids"]:
        print("Indexing documents for the first time...")
        vector_db.add_documents(chunks)
    else:
        print(f"Using existing collection with {len(existing_data['ids'])} vectors.")

    return vector_db
