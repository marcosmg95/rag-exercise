# Professional RAG workshop

In this session, we will create a RAG system from scratch, making it a professional, modular, and "production-ready" system.

The src directory will be uploaded to the repo after the class. And it will be a reference "architecture solution".

---

## Module 1: The Foundation - Environment & uv

**Context**: Professional Python development starts with reliable package management.

- **Task**: Initialize a project using `uv`.
- **Challenge**: Set up your `.env` file and ensure it's ignored by git.
- **Level Up**: Use `uv sync` to manage a locked environment across different machines.

## Module 2: Configuration - Pydantic Settings

**Context**: Hardcoding paths and API keys is a security risk.

- **Task**: Implement `config.py` using `BaseSettings`.
- **Challenge**: Automatically resolve paths relative to the project root regardless of where the script is executed.
- **Level Up**: Implement "Validation": ensure `GROQ_API_KEY` is present before the app even starts.

## Module 3: Document Ingestion - Chunking Strategies

**Context**: LLMs have limited context windows; how you split text determines how well they "read".

- **Task**: Use `RecursiveCharacterTextSplitter`.
- **Challenge**: Add `start_index` to your metadata to track exactly where a chunk came from.
- **Level Up**: Compare `chunk_size` 500 vs 1000. How does it affect the retrieval quality?

## Module 4: Local Intelligence - The Embedding Layer

**Context**: Privacy and cost are key. We will run embeddings locally.

- **Task**: Implement a custom `Embeddings` class using `SentenceTransformer` and `Torch`.
- **Challenge**: Ensure the model runs on GPU if available, fallback to CPU.
- **Level Up**: Cache the model weights locally so it doesn't re-download on every run.

## Module 5: Vector Persistence - ChromaDB

**Context**: Speed matters. You shouldn't re-index your PDF every time you ask a question.

- **Task**: Initialize `Chroma` with a persistent directory.
- **Challenge**: Implement an "Idempotent Load": check if the collection is already populated before adding documents.
- **Level Up**: Multi-tenancy: How would you handle multiple PDFs in different collections?

## Module 6: The Retrieval Engine

**Context**: Retrieval is the "R" in RAG.

- **Task**: Convert the Vector Store into a `Retriever`.
- **Challenge**: Implement a standalone `similarity_search` utility for debugging.
- **Level Up**: Filters. How would you retrieve only from specific "pages" if asked?

## Module 7: Advanced Retrieval - FlashRank Re-ranking

**Context**: Basic similarity (cosine) isn't always enough. Similarity != Relevance.

- **Task**: Implement a two-stage retriever using `ContextualCompressionRetriever`.
- **Challenge**: Retrieve 10 chunks, but use `FlashrankRerank` to only send the best 3 to the LLM.
- **Level Up**: Measure the latency. Is the extra time worth the accuracy gain?

## Module 8: Generation - Chains & Prompt Engineering

**Context**: The LLM needs strict instructions to prevent "Hallucinations".

- **Task**: Create a `ChatPromptTemplate` with a "Truthfulness" constraint.
- **Challenge**: Use LCEL (LangChain Expression Language) to pipe the retriever into the LLM.
- **Level Up**: Create a "System Message" that gives the AI a persona (e.g., "Analyst").

## Module 9: Observability - LangSmith

**Context**: If it breaks in production, you need to know why.

- **Task**: Connect your chain to LangSmith.
- **Challenge**: Tag your traces with the project name and version.
- **Level Up**: Identify the most expensive or slowest step in your chain using the Tracing UI.

## Module 10: Evaluation - The Golden Dataset

**Context**: "It feels okay" isn't a metric.

- **Task**: Create a Golden Dataset (Q&A pairs).
- **Challenge**: Calculate the **Hit Rate @ 3** (automated).
- **Level Up**: Implement a "Faithfulness" check using an LLM-as-a-judge.

---

## Module 11: Use CBOW for embeddings, that you trained on previous class

**Context**: Custom domain-specific embeddings can sometimes outperform generic pre-trained ones.

- **Task**: Load your custom CBOW model (e.g., from a `.bin` or `.model` file) and wrap it in a LangChain `Embeddings` class.
- **Level Up**: Compare the retrieval accuracy between your custom CBOW model and the professional `SentenceTransformer` model. Which one generalizes better?

## Module 12: Hybrid Search - Vector + Keyword

**Context**: Vectors are great for "meaning," but sometimes you need exact keyword matches (e.g., specific ID numbers or acronyms).

- **Task**: Research how to combine ChromaDB vector search with BM25 (keyword search).
- **Challenge**: Implement a "Reciprocal Rank Fusion" (RRF) to merge results from both methods.
- **Level Up**: When does keyword search perform better than vector search in your PDF?

## Module 13: Performance - Semantic Caching

**Context**: LLMs are expensive and slow. If two users ask the same thing, don't ask the LLM twice.

- **Task**: Implement a local "Semantic Cache" (using your embeddings).
- **Challenge**: If a new query is >95% similar to a previous one, return the cached answer.
- **Level Up**: What happens if the PDF content changes? How do you invalidate the cache?

## Module 14: Security - RAG Guardrails

**Context**: LLMs can "hallucinate" or be manipulated via prompt injection.

- **Task**: Implement a "input guardrail" to check if the user query is relevant to the document before searching.
- **Challenge**: Implement an "output guardrail" that double-checks if the LLM's answer is actually supported by the context.
- **Level Up**: Can you trick your own RAG system into talking about something unrelated?

## Module 15: Expansion - Multi-modal RAG

**Context**: Real documents have images, tables, and charts.

- **Task**: Investigate how to extract and describe images within the PDF using a Vision Model (e.g., Llama-3.2-Vision).
- **Challenge**: Index the "Image Descriptions" alongside the text chunks.
- **Level Up**: Ask a question about a chart in the PDF. Does the system find it?

---
