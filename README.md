# Professional RAG System & Workshop

A modular, production-ready RAG (Retrieval-Augmented Generation) system built with Python, LangChain, and Groq. This repository is designed as both a professional architecture reference and a comprehensive 15-module workshop curriculum for students.

## 🚀 Features

- **Professional Architecture**: Clean `src/` layout with isolated modules for Core, Database, Embeddings, Generation, Retrieval, and Evaluation.
- **Local Intelligence**: Local embeddings using `Sentence-Transformers` and `PyTorch` (running locally to save costs and ensure privacy).
- **Advanced Retrieval**: Two-stage retrieval process with **FlashRank Re-ranking** for high-precision context delivery.
- **Type-Safe Configuration**: Centralized settings management using `Pydantic Settings`.
- **Observability**: Built-in support for **LangSmith** tracing and monitoring.
- **Evaluation Suite**: Automated "Hit Rate @ K" metrics and LLM-as-a-judge faithfulness checks.
- **Modern Tooling**: Managed with `uv` for fast, reproducible environments.

## 📚 Workshop Curriculum

The repository includes a [WORKSHOP.md](docs/WORKSHOP.md) guide featuring 15 educational modules:

1. **Implementation (1-10)**: Build the core system from scratch, covering chunking, local embeddings, vector persistence, and re-ranking.
2. **Advanced Research (11-15)**: Investigate Hybrid Search, Semantic Caching, RAG Guardrails, Agentic RAG, and Multi-modal extraction.

## 🛠️ Tech Stack

- **Framework**: [LangChain](https://www.langchain.com/)
- **LLM**: [Groq](https://groq.com/) (Llama-3.3-70b)
- **Vector DB**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via PyTorch
- **Package Manager**: [uv](https://github.com/astral-sh/uv)

## ⚙️ Setup

1.  **Install dependencies**:
    ```bash
    uv sync
    ```
2.  **Environment Variables**: Create a `.env` file at the root:
    ```env
    GROQ_API_KEY=your_key_here
    LANGSMITH_API_KEY=optional_key
    ```
3.  **Run Interactive Chat**:
    ```bash
    $env:PYTHONPATH="src"; uv run python src/main.py
    ```
4.  **Run Evaluation**:
    ```bash
    $env:PYTHONPATH="src"; uv run python src/evaluation/__init__.py
    ```

---

_Created for educational excellence in NLP and AI Architecture._
