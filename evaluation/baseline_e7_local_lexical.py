# evaluation/baseline_e7_local_lexical.py

import json
import re
import statistics
import time
from pathlib import Path

from retrieval.search import vector_search


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

SUMMARY_PATH = RESULTS_DIR / "e7_local_lexical_summary.json"
DETAILS_PATH = RESULTS_DIR / "e7_local_lexical_details.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fixed parameters
# ============================================================

TOP_K = 10
CANDIDATE_LIMIT = 100
MAX_CHUNKS_PER_SOURCE = 1

ALPHA = 0.05

SEMANTIC_GAPS = [
    0.02,
    0.03,
    0.05,
]


# ============================================================
# Dataset
# ============================================================

def load_dataset(path: Path):
    dataset = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # Same evaluation rule used previously.
            if item.get("domain") == "spec":
                continue

            dataset.append(item)

    return dataset


# ============================================================
# Lexical scoring
# ============================================================

TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
)


def tokenize(text: str):
    if not text:
        return set()

    tokens = TOKEN_PATTERN.findall(
        text.lower()
    )

    return {
        token
        for token in tokens
        if len(token) >= 2
    }


def lexical_score(
    query: str,
    content: str,
):
    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    content_tokens = tokenize(content)

    matched = sum(
        1
        for token in query_tokens
        if token in content_tokens
    )

    return matched / len(query_tokens)


# ============================================================
# Source
# ============================================================

def get_source(result):
    source = result.get("url")

    if source:
        return source

    source = result.get("id")

    if source:
        return source

    return ""


# ============================================================
# Local lexical correction
# ============================================================

def local_lexical_rerank(
    query: str,
    candidates,
    alpha: float,
    semantic_gap: float,
):
    """
    Vector-first local lexical correction.

    IMPORTANT:
    - Preserve original Vector order.
    - No repeated bubbling.
    - A candidate can move forward by AT MOST ONE position.
    - Lexical correction is allowed only when the ORIGINAL
      adjacent Vector similarity gap is <= semantic_gap.

    This prevents a candidate from repeatedly swapping through
    multiple neighbors and effectively escaping the semantic-gap
    constraint.
    """

    if not candidates:
        return []

    # --------------------------------------------------------
    # Copy candidates so the original Vector result is preserved.
    # --------------------------------------------------------

    ranked = list(candidates)

    # --------------------------------------------------------
    # Calculate lexical scores.
    # --------------------------------------------------------

    for result in ranked:
        result["_vector_score"] = float(
            result.get("similarity", 0.0)
        )

        result["_lexical_score"] = lexical_score(
            query,
            result.get("content", ""),
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only perform ONE top-to-bottom pass.
    #
    # No while-loop.
    # No repeated swaps.
    #
    # Therefore a candidate cannot bubble through the ranking.
    # --------------------------------------------------------

    for i in range(len(ranked) - 1):

        current = ranked[i]
        next_item = ranked[i + 1]

        current_sim = current["_vector_score"]
        next_sim = next_item["_vector_score"]

        gap = current_sim - next_sim

        # Vector similarity is already tied or reversed.
        if gap <= 0:
            continue

        # Semantic difference is too large.
        if gap > semantic_gap:
            continue

        current_lex = current["_lexical_score"]
        next_lex = next_item["_lexical_score"]

        # No lexical advantage.
        if next_lex <= current_lex:
            continue

        current_score = (
            current_sim
            + alpha * current_lex
        )

        next_score = (
            next_sim
            + alpha * next_lex
        )

        # ----------------------------------------------------
        # Only allow one local swap.
        # ----------------------------------------------------

        if next_score > current_score:
            ranked[i], ranked[i + 1] = (
                ranked[i + 1],
                ranked[i],
            )

    return ranked


# ============================================================
# Source diversity
# ============================================================

def apply_source_diversity(
    results,
    limit=TOP_K,
    max_chunks_per_source=1,
):
    output = []
    source_counts = {}

    for result in results:

        source = get_source(result)

        if source_counts.get(
            source,
            0,
        ) >= max_chunks_per_source:
            continue

        output.append(result)

        source_counts[source] = (
            source_counts.get(
                source,
                0,
            ) + 1
        )

        if len(output) >= limit:
            break

    return output


# ============================================================
# Gold sources
# ============================================================

def get_relevant_sources(item):
    return set(
        item.get(
            "relevant_sources",
            [],
        )
    )


def is_hit(
    result,
    relevant_sources,
):
    return (
        get_source(result)
        in relevant_sources
    )


# ============================================================
# Metrics
# ============================================================

def calculate_hit_at_k(
    results,
    relevant_sources,
    k,
):
    return int(
        any(
            is_hit(
                result,
                relevant_sources,
            )
            for result in results[:k]
        )
    )


def calculate_recall_at_k(
    results,
    relevant_sources,
    k,
):
    if not relevant_sources:
        return 0.0

    retrieved_sources = {
        get_source(result)
        for result in results[:k]
    }

    matched = (
        retrieved_sources
        & relevant_sources
    )

    return (
        len(matched)
        / len(relevant_sources)
    )


def calculate_mrr(
    results,
    relevant_sources,
):
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if is_hit(
            result,
            relevant_sources,
        ):
            return 1.0 / rank

    return 0.0


# ============================================================
# Experiment
# ============================================================

def run_experiment(
    dataset,
    experiment_name,
    alpha,
    semantic_gap,
):
    details = []

    latencies = []

    hit1_values = []
    hit3_values = []
    hit5_values = []
    hit10_values = []

    recall10_values = []
    mrr_values = []

    for item in dataset:

        question_id = item.get("id")
        question = item.get(
            "question",
            "",
        )

        relevant_sources = (
            get_relevant_sources(item)
        )

        start = time.perf_counter()

        # ----------------------------------------------------
        # Vector Top100
        # ----------------------------------------------------

        candidates = vector_search(
            question,
            limit=CANDIDATE_LIMIT,
        )

        # ----------------------------------------------------
        # Local lexical correction
        # ----------------------------------------------------

        ranked = local_lexical_rerank(
            query=question,
            candidates=candidates,
            alpha=alpha,
            semantic_gap=semantic_gap,
        )

        # ----------------------------------------------------
        # Source diversity
        # ----------------------------------------------------

        final_results = apply_source_diversity(
            ranked,
            limit=TOP_K,
            max_chunks_per_source=(
                MAX_CHUNKS_PER_SOURCE
            ),
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            latency_ms
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        hit1 = calculate_hit_at_k(
            final_results,
            relevant_sources,
            1,
        )

        hit3 = calculate_hit_at_k(
            final_results,
            relevant_sources,
            3,
        )

        hit5 = calculate_hit_at_k(
            final_results,
            relevant_sources,
            5,
        )

        hit10 = calculate_hit_at_k(
            final_results,
            relevant_sources,
            10,
        )

        recall10 = calculate_recall_at_k(
            final_results,
            relevant_sources,
            10,
        )

        mrr = calculate_mrr(
            final_results,
            relevant_sources,
        )

        hit1_values.append(hit1)
        hit3_values.append(hit3)
        hit5_values.append(hit5)
        hit10_values.append(hit10)

        recall10_values.append(
            recall10
        )

        mrr_values.append(mrr)

        # ----------------------------------------------------
        # Details
        # ----------------------------------------------------

        retrieved_details = []

        for rank, result in enumerate(
            final_results,
            start=1,
        ):
            retrieved_details.append(
                {
                    "rank": rank,
                    "id": result.get("id"),
                    "title": result.get(
                        "title"
                    ),
                    "section": result.get(
                        "section"
                    ),
                    "url": result.get(
                        "url"
                    ),
                    "similarity": result.get(
                        "similarity"
                    ),
                    "lexical_score": result.get(
                        "_lexical_score",
                        0.0,
                    ),
                    "final_score": (
                        result.get(
                            "_vector_score",
                            0.0,
                        )
                        + alpha
                        * result.get(
                            "_lexical_score",
                            0.0,
                        )
                    ),
                }
            )

        details.append(
            {
                "id": question_id,
                "question": question,
                "experiment": experiment_name,
                "mode": "vector_local_lexical",
                "alpha": alpha,
                "semantic_gap": semantic_gap,
                "relevant_sources": list(
                    relevant_sources
                ),
                "retrieved_sources": [
                    get_source(result)
                    for result in final_results
                ],
                "retrieved_details": retrieved_details,
                "latency_ms": latency_ms,
                "metrics": {
                    "hit@1": hit1,
                    "hit@3": hit3,
                    "hit@5": hit5,
                    "hit@10": hit10,
                    "recall@10": recall10,
                    "mrr": mrr,
                },
            }
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    def mean(values):
        if not values:
            return 0.0

        return sum(values) / len(values)

    sorted_latencies = sorted(
        latencies
    )

    if sorted_latencies:
        p95_index = min(
            len(sorted_latencies) - 1,
            int(
                len(sorted_latencies)
                * 0.95
            ),
        )

        p95 = sorted_latencies[
            p95_index
        ]

        median = statistics.median(
            sorted_latencies
        )
    else:
        p95 = 0.0
        median = 0.0

    metrics = {
        "hit@1": mean(
            hit1_values
        ),
        "hit@3": mean(
            hit3_values
        ),
        "hit@5": mean(
            hit5_values
        ),
        "hit@10": mean(
            hit10_values
        ),
        "recall@10": mean(
            recall10_values
        ),
        "mrr": mean(
            mrr_values
        ),
        "questions": len(dataset),
        "hits@10": sum(
            hit10_values
        ),
        "misses@10": (
            len(dataset)
            - sum(hit10_values)
        ),
        "avg_latency_ms": mean(
            latencies
        ),
        "median_latency_ms": median,
        "p95_latency_ms": p95,
    }

    return metrics, details


# ============================================================
# Main
# ============================================================

def main():

    dataset = load_dataset(
        DATASET_PATH
    )

    print("=" * 90)
    print("E7 LOCAL LEXICAL EXPERIMENT")
    print("=" * 90)

    print(
        f"Questions: {len(dataset)}"
    )

    print(
        f"Candidate limit: "
        f"{CANDIDATE_LIMIT}"
    )

    print(
        f"Top K: {TOP_K}"
    )

    print(
        f"Max chunks/source: "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )

    print(
        f"Alpha: {ALPHA}"
    )

    all_summary = []
    all_details = {}

    for semantic_gap in SEMANTIC_GAPS:

        experiment_name = (
            f"E7_gap_{semantic_gap:.2f}"
        )

        print()
        print("-" * 90)
        print(
            f"{experiment_name} "
            f"(alpha={ALPHA}, "
            f"gap={semantic_gap})"
        )
        print("-" * 90)

        metrics, details = run_experiment(
            dataset=dataset,
            experiment_name=experiment_name,
            alpha=ALPHA,
            semantic_gap=semantic_gap,
        )

        all_summary.append(
            {
                "experiment": experiment_name,
                "alpha": ALPHA,
                "semantic_gap": semantic_gap,
                "metrics": metrics,
            }
        )

        all_details[
            experiment_name
        ] = details

        print(
            f"Hit@1     : "
            f"{metrics['hit@1']:.4f}"
        )

        print(
            f"Hit@3     : "
            f"{metrics['hit@3']:.4f}"
        )

        print(
            f"Hit@5     : "
            f"{metrics['hit@5']:.4f}"
        )

        print(
            f"Hit@10    : "
            f"{metrics['hit@10']:.4f}"
        )

        print(
            f"Recall@10 : "
            f"{metrics['recall@10']:.4f}"
        )

        print(
            f"MRR       : "
            f"{metrics['mrr']:.4f}"
        )

        print(
            f"Avg ms    : "
            f"{metrics['avg_latency_ms']:.2f}"
        )

        print(
            f"Median ms : "
            f"{metrics['median_latency_ms']:.2f}"
        )

        print(
            f"P95 ms    : "
            f"{metrics['p95_latency_ms']:.2f}"
        )

    # ========================================================
    # Save summary
    # ========================================================

    summary_output = {
        "dataset": str(
            DATASET_PATH
        ),
        "question_count": len(dataset),
        "pipeline": {
            "baseline": "vector_search",
            "local_lexical": True,
            "candidate_limit": (
                CANDIDATE_LIMIT
            ),
            "top_k": TOP_K,
            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),
            "alpha": ALPHA,
            "semantic_gaps": (
                SEMANTIC_GAPS
            ),
            "local_swap": {
                "max_move": 1,
                "repeated_bubbling": False,
            },
        },
        "configs": all_summary,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Save details
    # ========================================================

    details_output = {
        "dataset": str(
            DATASET_PATH
        ),
        "question_count": len(dataset),
        "experiments": all_details,
    }

    with open(
        DETAILS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            details_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Final table
    # ========================================================

    print()
    print("=" * 110)
    print("E7 SUMMARY")
    print("=" * 110)

    print(
        f"{'Exp':<18}"
        f"{'Gap':>8}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MRR':>10}"
        f"{'Avg ms':>12}"
    )

    print("-" * 110)

    for item in all_summary:

        metrics = item[
            "metrics"
        ]

        print(
            f"{item['experiment']:<18}"
            f"{item['semantic_gap']:>8.2f}"
            f"{metrics['hit@1']:>10.4f}"
            f"{metrics['hit@3']:>10.4f}"
            f"{metrics['hit@5']:>10.4f}"
            f"{metrics['hit@10']:>10.4f}"
            f"{metrics['recall@10']:>12.4f}"
            f"{metrics['mrr']:>10.4f}"
            f"{metrics['avg_latency_ms']:>12.2f}"
        )

    print()
    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        f"Details: {DETAILS_PATH}"
    )


if __name__ == "__main__":
    main()