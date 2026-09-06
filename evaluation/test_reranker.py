
from retrieval.search import (
    vector_search,
    vector_search_diverse,
)
from retrieval.reranker import rerank


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "q001",
        "question": "How do I create an MCP tool in Python?",
    },
    {
        "id": "q014",
        "question": "Can an MCP tool return a Pydantic model?",
    },
    {
        "id": "q021",
        "question": "When should I use the low-level MCP Server instead of FastMCP?",
    },
]


# ============================================================
# CONFIG
# ============================================================

BASELINE_CANDIDATE_LIMIT = 20

DIVERSE_CANDIDATE_LIMIT = 100
DIVERSE_FINAL_CANDIDATES = 20
MAX_CHUNKS_PER_SOURCE = 1

FINAL_TOP_K = 10


# ============================================================
# HELPERS
# ============================================================

def source_of(result):
    return result.get("url", "")


def print_source_statistics(results, title):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)

    total = len(results)

    source_counts = {}

    for result in results:
        source = source_of(result)

        if not source:
            source = "<EMPTY URL>"

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

    print(f"Total candidates: {total}")
    print(f"Unique sources:   {len(source_counts)}")

    print("\nSource distribution:")

    for source, count in sorted(
        source_counts.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(f"  {count:2d}x  {source}")


def print_reranked_results(
    results,
    title,
    top_k=10,
):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for rank, result in enumerate(
        results[:top_k],
        start=1,
    ):

        print(f"\nRank {rank}")

        print(
            f"ID:         "
            f"{result.get('id')}"
        )

        print(
            f"Title:      "
            f"{result.get('title', '')}"
        )

        print(
            f"Section:    "
            f"{result.get('section', '')}"
        )

        print(
            f"URL:        "
            f"{result.get('url', '')}"
        )

        print(
            f"Vector:     "
            f"{result.get('similarity', 'N/A')}"
        )

        print(
            f"Reranker:   "
            f"{result.get('reranker_score', 'N/A')}"
        )


def dedup_by_source(
    results,
    limit,
    max_chunks_per_source=1,
):
    output = []
    source_counts = {}

    for result in results:

        source = source_of(result)

        if (
            source_counts.get(source, 0)
            >= max_chunks_per_source
        ):
            continue

        output.append(result)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(output) >= limit:
            break

    return output


def print_id_comparison(
    baseline,
    diverse,
    rerank_dedup,
):
    print("\n" + "=" * 100)
    print("D. RERANKED TOP-10 ID COMPARISON")
    print("=" * 100)

    print("\nBaseline:")

    for rank, result in enumerate(
        baseline[:FINAL_TOP_K],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"{result.get('id')}"
        )

    print("\nDiverse:")

    for rank, result in enumerate(
        diverse[:FINAL_TOP_K],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"{result.get('id')}"
        )

    print("\nRerank → Dedup:")

    for rank, result in enumerate(
        rerank_dedup[:FINAL_TOP_K],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"{result.get('id')}"
        )


def compare_candidates(
    baseline_candidates,
    diverse_candidates,
    rerank_candidates,
):
    baseline_ids = {
        result.get("id")
        for result in baseline_candidates
    }

    diverse_ids = {
        result.get("id")
        for result in diverse_candidates
    }

    rerank_ids = {
        result.get("id")
        for result in rerank_candidates
    }

    print("\n" + "=" * 100)
    print("C. CANDIDATE COMPARISON")
    print("=" * 100)

    print(
        f"\nBaseline candidates: "
        f"{len(baseline_ids)}"
    )

    print(
        f"Diverse candidates:  "
        f"{len(diverse_ids)}"
    )

    print(
        f"Rerank candidates:   "
        f"{len(rerank_ids)}"
    )

    print(
        f"\nBaseline ∩ Diverse: "
        f"{len(baseline_ids & diverse_ids)}"
    )

    print(
        f"Baseline ∩ Rerank:  "
        f"{len(baseline_ids & rerank_ids)}"
    )

    print(
        f"Diverse ∩ Rerank:   "
        f"{len(diverse_ids & rerank_ids)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    for case in TEST_CASES:

        question_id = case["id"]
        question = case["question"]

        print("\n\n" + "#" * 100)
        print(question_id)
        print(question)
        print("#" * 100)

        # ====================================================
        # A. BASELINE
        # Vector Search → Reranker
        # ====================================================

        print("\n")
        print("=" * 100)
        print("A. BASELINE: VECTOR SEARCH")
        print("=" * 100)

        baseline_candidates = vector_search(
            question,
            limit=BASELINE_CANDIDATE_LIMIT,
        )

        print(
            f"\nVector candidates: "
            f"{len(baseline_candidates)}"
        )

        print_source_statistics(
            baseline_candidates,
            "Baseline source statistics",
        )

        baseline_reranked = rerank(
            question,
            baseline_candidates,
        )

        print_reranked_results(
            baseline_reranked,
            "BASELINE — RERANKED TOP 10",
            FINAL_TOP_K,
        )

        # ====================================================
        # B. DIVERSE
        # Vector Search → Source Dedup → Reranker
        # ====================================================

        print("\n")
        print("=" * 100)
        print("B. DIVERSE: VECTOR SEARCH + SOURCE DEDUP")
        print("=" * 100)

        diverse_candidates = vector_search_diverse(
            question,
            limit=DIVERSE_FINAL_CANDIDATES,
            candidate_limit=DIVERSE_CANDIDATE_LIMIT,
            max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
        )

        print(
            f"\nDiverse candidates: "
            f"{len(diverse_candidates)}"
        )

        print_source_statistics(
            diverse_candidates,
            "Diverse source statistics",
        )

        diverse_reranked = rerank(
            question,
            diverse_candidates,
        )

        print_reranked_results(
            diverse_reranked,
            "DIVERSE — RERANKED TOP 10",
            FINAL_TOP_K,
        )

        # ====================================================
        # C. RERANK FIRST + SOURCE DEDUP
        # Vector Search → Reranker → Source Dedup
        # ====================================================

        print("\n")
        print("=" * 100)
        print("C. RERANK FIRST + SOURCE DEDUP")
        print("=" * 100)

        rerank_candidates = vector_search(
            question,
            limit=DIVERSE_CANDIDATE_LIMIT,
        )

        print(
            f"\nRerank candidates: "
            f"{len(rerank_candidates)}"
        )

        print_source_statistics(
            rerank_candidates,
            "Rerank candidate source statistics",
        )

        reranked_all = rerank(
            question,
            rerank_candidates,
        )

        reranked_dedup = dedup_by_source(
            reranked_all,
            limit=FINAL_TOP_K,
            max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
        )

        print(
            f"\nFinal candidates after dedup: "
            f"{len(reranked_dedup)}"
        )

        print_reranked_results(
            reranked_dedup,
            "C — RERANK THEN DEDUP TOP 10",
            FINAL_TOP_K,
        )

        # ====================================================
        # CANDIDATE COMPARISON
        # ====================================================

        compare_candidates(
            baseline_candidates,
            diverse_candidates,
            rerank_candidates,
        )

        # ====================================================
        # FINAL ID COMPARISON
        # ====================================================

        print_id_comparison(
            baseline_reranked,
            diverse_reranked,
            reranked_dedup,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

