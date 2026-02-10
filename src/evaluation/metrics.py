from pathlib import Path
from core import RAGSystem
from config import settings


def run_evaluation(pdf_path: str):
    """Run basic evaluation on the RAG system."""
    rag = RAGSystem(pdf_path)
    rag.ingest_and_process()
    rag.setup()

    # Golden Dataset: 5 questions and answers based on the document
    golden_dataset = [
        {
            "question": "¿Cuál es el objetivo principal de la actividad?",
            "expected_contains": "objetivo",
        },
        {
            "question": "¿Qué herramientas se deben utilizar?",
            "expected_contains": "langchain",
        },
        {"question": "¿Cómo se evalúa el sistema?", "expected_contains": "hit rate"},
        {
            "question": "¿Qué modelo de embeddings se utiliza?",
            "expected_contains": "text-embedding-3",
        },
        {
            "question": "¿Qué base de datos vectorial se recomienda?",
            "expected_contains": "Chroma",
        },
    ]

    hits = 0

    print("\n--- Starting Basic Evaluation (Hit Rate @ 3) ---")

    for item in golden_dataset:
        query = item["question"]
        docs = rag.similarity_search(query)

        # Hit rate: check if expected keyword is in the top 3 documents
        found = any(
            item["expected_contains"].lower() in doc.page_content.lower()
            for doc in docs
        )

        if found:
            hits += 1
            status = "HIT"
        else:
            status = "MISS"

        print(f"Question: {query} -> {status}")

    hit_rate = (hits / len(golden_dataset)) * 100
    print(f"\nHit Rate @ 3: {hit_rate}%")

    # LLM-as-a-judge evaluation (Faithfulness)
    print("\n--- Evaluation with LLM as Judge (Faithfulness) ---")
    question = golden_dataset[0]["question"]
    result = rag.ask(question)
    print(f"Generated Answer: {result['answer']}")
    print(
        "The system is prepared to integrate RAGAS using the 'answer' and 'sources' outputs."
    )


if __name__ == "__main__":
    pdf_path = Path(settings.DEFAULT_PDF_PATH)
    if pdf_path.exists():
        run_evaluation(str(pdf_path))
    else:
        print(f"File not found: {pdf_path}")
