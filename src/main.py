from pathlib import Path
from core import RAGSystem
from config import settings

if __name__ == "__main__":
    pdf_path = Path(settings.DEFAULT_PDF_PATH)

    if pdf_path.exists():
        rag = RAGSystem(str(pdf_path))
        print(f"Ingesting PDF from: {pdf_path}")
        rag.ingest_and_process()
        print("Setting up Vector DB...")
        rag.setup()

        print("\n--- RAG Chat Initialized ---")
        print("Type 'salir' to exit or 'rerank' to toggle re-ranking.")

        use_rerank = False
        while True:
            try:
                query = input("\n> ").strip()

                if not query:
                    continue
                if query.lower() in ["salir", "exit", "quit"]:
                    break
                if query.lower() == "rerank":
                    use_rerank = not use_rerank
                    print(f"Re-ranking: {'ON' if use_rerank else 'OFF'}")
                    continue

                print("Thinking...")
                result = rag.ask(query, use_rerank=use_rerank)
                print(f"\nAnswer: {result['answer']}")

                print("\nSources:")
                for s in result["sources"]:
                    print(f"- [Page {s['page']}]: {s['content']}")

            except KeyboardInterrupt:
                break

        print("\nGoodbye!")
    else:
        print(f"File not found: {pdf_path}")
