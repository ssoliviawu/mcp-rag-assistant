from embedding.model import BGEEmbeddingModel
from db.repository import (
    search_chunks,
    search_chunks_keyword,
)
from retrieval.reranker import rerank

_model = BGEEmbeddingModel()


# ============================================================
# Vector Search
# ============================================================

def vector_search(
    query: str,
    limit: int = 5,
):
    """
    Search the knowledge base using
    semantic similarity.

    This is the original vector search.
    """

    query_embedding = _model.embed_query(
        query
    )

    return search_chunks(
        query_embedding,
        limit=limit,
    )


# ============================================================
# Keyword Search
# ============================================================

def keyword_search(
    query: str,
    limit: int = 5,
):
    """
    Search the knowledge base using
    PostgreSQL Full-Text Search.
    """

    return search_chunks_keyword(
        query,
        limit=limit,
    )


# ============================================================
# Source-level Deduplication
# ============================================================

def deduplicate_by_source(
    results,
    limit: int = 5,
):
    """
    Remove duplicate chunks from the same source URL.

    The retrieval system works at chunk level, but multiple
    chunks may belong to the same document/source.

    Example:

        Input:

            A
            A
            A
            B
            C
            B
            D

        Output:

            A
            B
            C
            D

    Since results are already ranked, the first occurrence
    of each source is kept because it has the best score.

    Parameters
    ----------
    results:
        Ranked retrieval results.

    limit:
        Maximum number of unique sources to return.

    Returns
    -------
    list:
        Deduplicated ranked results.
    """

    seen_sources = set()

    output = []

    for result in results:

        source = result.get(
            "url"
        )

        # ----------------------------------------------------
        # Fallback
        #
        # If URL is missing, use chunk ID so that the result
        # is not accidentally removed.
        # ----------------------------------------------------

        if not source:

            source = result.get(
                "id"
            )

        if not source:

            # Extremely defensive fallback.
            # Keep the result if no identifier exists.
            output.append(
                result
            )

            if len(output) >= limit:
                break

            continue

        if source in seen_sources:
            continue

        seen_sources.add(
            source
        )

        output.append(
            result
        )

        if len(output) >= limit:
            break

    return output

def vector_rerank_dedup(
    query: str,
    limit: int = 10,
    candidate_limit: int = 50,
):
    """
    Vector retrieval → ONNX reranking → source deduplication.
    """

    candidates = vector_search(
        query,
        limit=candidate_limit,
    )

    if not candidates:
        return []

    reranked = rerank(
        query,
        candidates,
    )

    return deduplicate_by_source(
        reranked,
        limit=limit,
    )

# ============================================================
# Diverse Vector Search
# ============================================================

def vector_search_diverse(
    query: str,
    limit: int = 5,
    candidate_limit: int = 100,
    max_chunks_per_source: int = 1,
):
    candidates = vector_search(
        query,
        limit=candidate_limit,
    )

    results = []
    source_counts = {}

    for result in candidates:

        source = result.get("url", "")

        if (
            source_counts.get(source, 0)
            >= max_chunks_per_source
        ):
            continue

        results.append(result)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(results) >= limit:
            break

    return results

# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    result_lists,
    k: int = 60,
    limit: int = 5,
):
    """
    Merge multiple ranked result lists using
    Reciprocal Rank Fusion (RRF).

    RRF:

        score(d) = sum(1 / (k + rank))

    rank starts from 1.
    """

    scores = {}
    results_by_id = {}

    for results in result_lists:

        for rank, result in enumerate(
            results,
            start=1,
        ):

            result_id = result["id"]

            scores[result_id] = (
                scores.get(
                    result_id,
                    0.0,
                )
                + 1.0 / (
                    k + rank
                )
            )

            results_by_id[
                result_id
            ] = result

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    output = []

    for result_id in ranked_ids[:limit]:

        result = dict(
            results_by_id[result_id]
        )

        result["rrf_score"] = (
            scores[result_id]
        )

        output.append(
            result
        )

    return output


# ============================================================
# Hybrid Search
# ============================================================

def hybrid_search(
    query: str,
    limit: int = 10,
    candidate_limit: int = 30,
    rrf_k: int = 60,
):
    """
    Hybrid retrieval:

        Vector Search
              +
        Keyword Search
              ↓
             RRF
              ↓
           Top-K
    """

    vector_results = vector_search(
        query,
        limit=candidate_limit,
    )

    keyword_results = keyword_search(
        query,
        limit=candidate_limit,
    )

    return reciprocal_rank_fusion(
        [
            vector_results,
            keyword_results,
        ],
        k=rrf_k,
        limit=limit,
    )


# ============================================================
# Diverse Hybrid Search
# ============================================================

def hybrid_search_diverse(
    query: str,
    limit: int = 5,
    candidate_limit: int = 30,
    rrf_k: int = 60,
):
    """
    Hybrid retrieval with source-level deduplication.

    Pipeline:

        Vector Search
              +
        Keyword Search
              ↓
             RRF
              ↓
        Source Dedup
              ↓
           Top-K

    candidate_limit controls how many candidates are
    retrieved from each retrieval method before RRF.
    """

    if candidate_limit < limit:

        raise ValueError(
            "candidate_limit must be "
            "greater than or equal to limit."
        )

    vector_results = vector_search(
        query,
        limit=candidate_limit,
    )

    keyword_results = keyword_search(
        query,
        limit=candidate_limit,
    )

    # --------------------------------------------------------
    # RRF first.
    #
    # We intentionally keep this as chunk-level RRF.
    # Source-level dedup happens after ranking.
    # --------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        [
            vector_results,
            keyword_results,
        ],
        k=rrf_k,
        limit=candidate_limit,
    )

    # --------------------------------------------------------
    # Deduplicate sources.
    # --------------------------------------------------------

    return deduplicate_by_source(
        fused_results,
        limit=limit,
    )


# ============================================================
# Print Results
# ============================================================

def print_results(
    results,
    title="Search Results",
):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"Rank: {rank}"
        )

        if "rrf_score" in result:

            print(
                f"RRF Score: "
                f"{result['rrf_score']:.6f}"
            )

        else:

            print(
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )

        print(
            f"Title: "
            f"{result['title']}"
        )

        print(
            f"Section: "
            f"{result['section']}"
        )

        print(
            f"URL: "
            f"{result['url']}"
        )

        print()

        print(
            result["content"]
        )

        print()

        print("-" * 80)


# ============================================================
# CLI
# ============================================================

def main():

    query = input(
        "Query: "
    )

    print()
    print(
        "Select search mode:"
    )

    print(
        "1. Vector"
    )

    print(
        "2. Keyword"
    )

    print(
        "3. Hybrid"
    )

    print(
        "4. Vector + Source Dedup"
    )

    print(
        "5. Hybrid + Source Dedup"
    )

    mode = input(
        "\nMode [1/2/3/4/5]: "
    ).strip()

    # --------------------------------------------------------
    # Original Vector
    # --------------------------------------------------------

    if mode == "1":

        results = vector_search(
            query,
            limit=5,
        )

        print_results(
            results,
            title=(
                "Vector Search Results"
            ),
        )

    # --------------------------------------------------------
    # Original Keyword
    # --------------------------------------------------------

    elif mode == "2":

        results = keyword_search(
            query,
            limit=5,
        )

        print_results(
            results,
            title=(
                "Keyword Search Results"
            ),
        )

    # --------------------------------------------------------
    # Original Hybrid
    # --------------------------------------------------------

    elif mode == "3":

        results = hybrid_search(
            query,
            limit=5,
            candidate_limit=10,
        )

        print_results(
            results,
            title=(
                "Hybrid Search Results"
            ),
        )

    # --------------------------------------------------------
    # Vector + Source Dedup
    # --------------------------------------------------------

    elif mode == "4":

        results = vector_search_diverse(
            query,
            limit=5,
            candidate_limit=30,
        )

        print_results(
            results,
            title=(
                "Vector + Source Dedup Results"
            ),
        )

    # --------------------------------------------------------
    # Hybrid + Source Dedup
    # --------------------------------------------------------

    elif mode == "5":

        results = hybrid_search_diverse(
            query,
            limit=5,
            candidate_limit=30,
        )

        print_results(
            results,
            title=(
                "Hybrid + Source Dedup Results"
            ),
        )

    else:

        print(
            "Invalid mode."
        )


if __name__ == "__main__":
    main()