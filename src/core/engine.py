from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from embeddings import LocalPyTorchEmbeddings
from database import setup_vector_db
from retrieval import get_reranked_retriever
from generation import get_rag_chain
from config import settings


class RAGSystem:
    """Core RAG System orchestrator."""

    def __init__(self, pdf_path: str | None = None, collection_name: str | None = None):
        self.pdf_path = pdf_path or settings.DEFAULT_PDF_PATH
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.embeddings = LocalPyTorchEmbeddings()
        self.vector_db = None
        self.retriever = None
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    def ingest_and_process(self):
        """Step 1: Document ingestion and chunking."""
        print(f"Loading PDF from: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"Total pages loaded: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=150, add_start_index=True
        )
        self.chunks = text_splitter.split_documents(documents)
        print(f"Created {len(self.chunks)} chunks.")

    def setup(self):
        """Initialize the vector database and retriever."""
        self.vector_db = setup_vector_db(
            self.chunks, self.embeddings, self.collection_name
        )
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

    def similarity_search(self, query: str) -> List[Any]:
        """Step 4: Similarity search implementation."""
        if not self.vector_db:
            raise ValueError("Vector DB not initialized.")
        return self.vector_db.similarity_search(query, k=3)

    def ask(self, query: str, use_rerank: bool = False) -> Dict[str, Any]:
        """Orchestrate the retrieval and generation process."""
        retriever = (
            get_reranked_retriever(self.retriever) if use_rerank else self.retriever
        )
        chain = get_rag_chain(retriever, self.llm)

        # Get documents for source metadata
        docs = retriever.invoke(query)
        response = chain.invoke(query)

        sources = [
            {
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:100] + "...",
            }
            for doc in docs
        ]

        return {
            "query": query,
            "answer": response,
            "sources": sources,
            "reranked_used": use_rerank,
        }
