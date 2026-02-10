def get_reranked_retriever(base_retriever, k_initial: int = 10, k_final: int = 3):
    """
    Wrap a retriever with FlashRank re-ranking.
    Initializes a ContextualCompressionRetriever using FlashrankRerank.
    """
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors.flashrank_rerank import (
        FlashrankRerank,
    )

    base_retriever.search_kwargs["k"] = k_initial
    compressor = FlashrankRerank(top_n=k_final)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
