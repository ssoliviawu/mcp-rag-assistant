
import json
from pathlib import Path

from retrieval.search import vector_search


# ============================================================
# Config
# ============================================================

DATASET_PATH = Path("mcp_rag_eval_dataset.jsonl")

CANDIDATE_LIMIT = 100
TOP_K = 10

CONFIGS = {
    "e2_1": 1,
    "e2_2": 2,
}


# ============================================================
# Helpers
# ============================================================

def normalize_url(url):
    """
    Normalize URL for comparison.

    Handles trailing slash differences.
    """
    if not url:
        return ""

    return url.rstrip("/")


def load_dataset():
    questions = []

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            questions.append(item)

    return questions


def get_relevant_sources(item):
    """
    Get relevant source URLs from dataset.

    Uses the same field and logic as the
    already-validated E2 diversity analysis.
    """

    sources = item.get(
        "relevant_sources",
        [],
    )

    if isinstance(sources, str):
        sources = [sources]

    return [
        normalize_url(source)
        for source in sources
        if source
    ]


def find_source_rank(
    results,
    relevant_sources,
):
    """
    Find the first rank where a relevant source appears.
    """

    relevant_sources = set(
        relevant_sources
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        source = normalize_url(
            result.get("url", "")
        )

        if source in relevant_sources:
            return rank, result

    return None, None


# ============================================================
# E2 Retrieval
# ============================================================

def run_e2(
    query,
    max_chunks_per_source,
):
    """
    Pure vector + source diversity.

    No reranker.
    No keyword search.
    No hybrid.
    """

    candidates = vector_search(
        query,
        limit=CANDIDATE_LIMIT,
    )

    results = []

    source_counts = {}

    for result in candidates:

        source = normalize_url(
            result.get("url", "")
        )

        if (
            source_counts.get(source, 0)
            >= max_chunks_per_source
        ):
            continue

        results.append(result)

        source_counts[source] = (
            source_counts.get(source, 0)
            + 1
        )

        if len(results) >= TOP_K:
            break

    return results


# ============================================================
# Analyze one configuration
# ============================================================

def analyze_config(
    dataset,
    config_name,
    max_chunks_per_source,
):

    print()
    print("=" * 100)
    print(
        f"{config_name.upper()} | "
        f"max_chunks_per_source="
        f"{max_chunks_per_source}"
    )
    print("=" * 100)

    results = []

    ranks = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        question_id = item.get(
            "id",
            "unknown",
        )

        question = item.get(
            "question",
            "",
        )

        relevant_sources = get_relevant_sources(
            item
        )

        if not relevant_sources:

            result = {
                "id": question_id,
                "question": question,
                "status": "NO_GOLD_SOURCE",
                "rank": None,
            }

            results.append(result)
            ranks.append(None)

            continue

        # ----------------------------------------------------
        # Vector Top 100
        # ----------------------------------------------------

        vector_results = vector_search(
            question,
            limit=CANDIDATE_LIMIT,
        )

        vector_rank, vector_result = (
            find_source_rank(
                vector_results,
                relevant_sources,
            )
        )

        # ----------------------------------------------------
        # E2
        # ----------------------------------------------------

        e2_results = run_e2(
            question,
            max_chunks_per_source,
        )

        e2_rank, e2_result = (
            find_source_rank(
                e2_results,
                relevant_sources,
            )
        )

        ranks.append(e2_rank)

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if vector_rank is None:

            status = "NOT_IN_VECTOR_100"

        elif e2_rank is None:

            status = "FILTERED_BY_DIVERSITY"

        else:

            status = "SURVIVED_E2"

        result = {
            "id": question_id,
            "question": question,
            "gold_sources": relevant_sources,

            "vector_rank": vector_rank,

            "e2_rank": e2_rank,

            "status": status,

            "vector_chunk": (
                {
                    "title": vector_result.get(
                        "title"
                    ),
                    "section": vector_result.get(
                        "section"
                    ),
                    "url": vector_result.get(
                        "url"
                    ),
                    "similarity": vector_result.get(
                        "similarity"
                    ),
                }
                if vector_result
                else None
            ),

            "e2_chunk": (
                {
                    "title": e2_result.get(
                        "title"
                    ),
                    "section": e2_result.get(
                        "section"
                    ),
                    "url": e2_result.get(
                        "url"
                    ),
                    "similarity": e2_result.get(
                        "similarity"
                    ),
                }
                if e2_result
                else None
            ),

            "e2_results": [
                {
                    "rank": rank,
                    "title": result_item.get(
                        "title"
                    ),
                    "section": result_item.get(
                        "section"
                    ),
                    "url": result_item.get(
                        "url"
                    ),
                    "similarity": result_item.get(
                        "similarity"
                    ),
                }
                for rank, result_item in enumerate(
                    e2_results,
                    start=1,
                )
            ],
        }

        results.append(result)

        print(
            f"\rAnalyzing "
            f"{index}/{len(dataset)}...",
            end="",
            flush=True,
        )

    print()

    # ========================================================
    # Metrics
    # ========================================================

    total = len(dataset)

    def hit_at_k(k):
        return sum(
            1
            for rank in ranks
            if rank is not None
            and rank <= k
        ) / total

    mrr = sum(
        (
            1.0 / rank
            if rank is not None
            else 0.0
        )
        for rank in ranks
    ) / total

    hits_at_10 = sum(
        1
        for rank in ranks
        if rank is not None
        and rank <= TOP_K
    )

    misses_at_10 = total - hits_at_10

    metrics = {
        "hit@1": round(
            hit_at_k(1),
            4,
        ),
        "hit@3": round(
            hit_at_k(3),
            4,
        ),
        "hit@5": round(
            hit_at_k(5),
            4,
        ),
        "hit@10": round(
            hit_at_k(10),
            4,
        ),
        "mrr": round(
            mrr,
            4,
        ),
        "questions": total,
        "hits@10": hits_at_10,
        "misses@10": misses_at_10,
    }

    # ========================================================
    # Status counts
    # ========================================================

    status_counts = {
        "SURVIVED_E2": 0,
        "FILTERED_BY_DIVERSITY": 0,
        "NOT_IN_VECTOR_100": 0,
        "NO_GOLD_SOURCE": 0,
    }

    for result in results:

        status = result.get(
            "status"
        )

        if status in status_counts:
            status_counts[status] += 1

    return {
        "config": {
            "max_chunks_per_source":
                max_chunks_per_source,
            "candidate_limit":
                CANDIDATE_LIMIT,
            "top_k":
                TOP_K,
        },

        "metrics": metrics,

        "status_counts":
            status_counts,

        "questions":
            results,
    }


# ============================================================
# Compare configurations
# ============================================================

def compare_configs(
    e2_1,
    e2_2,
):

    results_1 = {
        item["id"]: item
        for item in e2_1["questions"]
    }

    results_2 = {
        item["id"]: item
        for item in e2_2["questions"]
    }

    comparison = []

    improved = 0
    worsened = 0
    unchanged = 0

    for question_id in results_1:

        item_1 = results_1[
            question_id
        ]

        item_2 = results_2[
            question_id
        ]

        rank_1 = item_1.get(
            "e2_rank"
        )

        rank_2 = item_2.get(
            "e2_rank"
        )

        # ----------------------------------------------------
        # MISS = infinity
        # ----------------------------------------------------

        score_1 = (
            rank_1
            if rank_1 is not None
            else float("inf")
        )

        score_2 = (
            rank_2
            if rank_2 is not None
            else float("inf")
        )

        if score_2 < score_1:

            change = "IMPROVED"
            improved += 1

        elif score_2 > score_1:

            change = "WORSENED"
            worsened += 1

        else:

            change = "UNCHANGED"
            unchanged += 1

        # ----------------------------------------------------
        # Rank difference
        # ----------------------------------------------------

        if (
            rank_1 is not None
            and rank_2 is not None
        ):

            rank_delta = (
                rank_1 - rank_2
            )

        elif (
            rank_1 is None
            and rank_2 is not None
        ):

            rank_delta = "MISS_TO_HIT"

        elif (
            rank_1 is not None
            and rank_2 is None
        ):

            rank_delta = "HIT_TO_MISS"

        else:

            rank_delta = 0

        comparison.append(
            {
                "id": question_id,

                "question":
                    item_1["question"],

                "e2_1_rank":
                    rank_1,

                "e2_2_rank":
                    rank_2,

                "rank_delta":
                    rank_delta,

                "change":
                    change,

                "e2_1_chunk":
                    item_1.get(
                        "e2_chunk"
                    ),

                "e2_2_chunk":
                    item_2.get(
                        "e2_chunk"
                    ),
            }
        )

    return {
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "details": comparison,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print("E2 max_chunks_per_source A/B EXPERIMENT")
    print("=" * 100)

    print()
    print(
        "Pure Vector Retrieval"
    )

    print(
        f"Candidate limit: "
        f"{CANDIDATE_LIMIT}"
    )

    print(
        f"Top-K: "
        f"{TOP_K}"
    )

    print(
        "Reranker: False"
    )

    print(
        "Hybrid: False"
    )

    print()

    dataset = load_dataset()

    print(
        f"Loaded {len(dataset)} questions."
    )

    # ========================================================
    # E2-1
    # ========================================================

    e2_1 = analyze_config(
        dataset,
        "e2_1",
        CONFIGS["e2_1"],
    )

    # ========================================================
    # E2-2
    # ========================================================

    e2_2 = analyze_config(
        dataset,
        "e2_2",
        CONFIGS["e2_2"],
    )

    # ========================================================
    # Comparison
    # ========================================================

    comparison = compare_configs(
        e2_1,
        e2_2,
    )

    # ========================================================
    # Final JSON
    # ========================================================

    output = {

        "experiment":
            "E2 max_chunks_per_source",

        "dataset":
            str(DATASET_PATH),

        "question_count":
            len(dataset),

        "fixed_parameters": {
            "retrieval":
                "vector",

            "candidate_limit":
                CANDIDATE_LIMIT,

            "top_k":
                TOP_K,

            "reranker":
                False,

            "hybrid":
                False,
        },

        "configs": {

            "e2_1": {
                "max_chunks_per_source":
                    CONFIGS["e2_1"],

                "metrics":
                    e2_1["metrics"],

                "status_counts":
                    e2_1["status_counts"],
            },

            "e2_2": {
                "max_chunks_per_source":
                    CONFIGS["e2_2"],

                "metrics":
                    e2_2["metrics"],

                "status_counts":
                    e2_2["status_counts"],
            },
        },

        "rank_change": {
            "improved":
                comparison["improved"],

            "worsened":
                comparison["worsened"],

            "unchanged":
                comparison["unchanged"],
        },

        "details":
            comparison["details"],
    }

    # ========================================================
    # Save
    # ========================================================

    output_path = Path(
        "evaluation/results/"
        "e2_max_chunks_comparison.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    print()

    print(
        "E2-1 | max_chunks_per_source=1"
    )

    print(
        json.dumps(
            e2_1["metrics"],
            indent=2,
        )
    )

    print()

    print(
        "E2-2 | max_chunks_per_source=2"
    )

    print(
        json.dumps(
            e2_2["metrics"],
            indent=2,
        )
    )

    print()

    print(
        "Rank changes:"
    )

    print(
        f"  Improved : "
        f"{comparison['improved']}"
    )

    print(
        f"  Worsened : "
        f"{comparison['worsened']}"
    )

    print(
        f"  Unchanged: "
        f"{comparison['unchanged']}"
    )

    print()

    print(
        "Saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()

