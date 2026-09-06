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

SUMMARY_PATH = (
    RESULTS_DIR
    / "e7b_local_lexical_top10_summary.json"
)

DETAILS_PATH = (
    RESULTS_DIR
    / "e7b_local_lexical_top10_details.json"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Parameters
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

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # Same evaluation rule as previous experiments.
            if item.get("domain") == "spec":
                continue

            dataset.append(item)

    return dataset


# ============================================================
# Tokenization
# ============================================================

TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
)


def tokenize(text):

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
    query,
    content,
):

    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    content_tokens = tokenize(content)

    matched = sum(
        token in content_tokens
        for token in query_tokens
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
# Source diversity
# ============================================================

def source_diversity(
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
            )
            + 1
        )

        if len(output) >= limit:
            break

    return output


# ============================================================
# Candidate identity
# ============================================================

def result_identity(result):

    """
    Use chunk id when available.

    Fall back to:
        url + title + section
    """

    result_id = result.get("id")

    if result_id:
        return str(result_id)

    return (
        str(result.get("url", ""))
        + "||"
        + str(result.get("title", ""))
        + "||"
        + str(result.get("section", ""))
    )


def candidate_set(results):

    return {
        result_identity(result)
        for result in results
    }


# ============================================================
# Local lexical correction
# ============================================================

def local_lexical_correction(
    query,
    results,
    alpha,
    semantic_gap,
):

    """
    Input:
        E2 Vector-Diverse Top10

    Output:
        Same 10 candidates, reordered only.

    A candidate can move up by at most one position.

    Swap decisions are calculated from the ORIGINAL
    ordering to avoid chaining.
    """

    if len(results) <= 1:
        return list(results)

    original = list(results)

    # --------------------------------------------------------
    # Calculate lexical information.
    # --------------------------------------------------------

    for result in original:

        result["_vector_score"] = float(
            result.get(
                "similarity",
                0.0,
            )
        )

        result["_lexical_score"] = (
            lexical_score(
                query,
                result.get(
                    "content",
                    "",
                ),
            )
        )

    # --------------------------------------------------------
    # Decide swaps from ORIGINAL ranking.
    # --------------------------------------------------------

    swap_positions = set()

    for i in range(
        len(original) - 1
    ):

        current = original[i]

        next_item = original[i + 1]

        current_sim = (
            current["_vector_score"]
        )

        next_sim = (
            next_item["_vector_score"]
        )

        gap = (
            current_sim
            - next_sim
        )

        # Only consider semantic neighbors.
        if gap < 0:
            continue

        if gap > semantic_gap:
            continue

        current_lex = (
            current["_lexical_score"]
        )

        next_lex = (
            next_item["_lexical_score"]
        )

        # Next candidate does not have
        # a lexical advantage.
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

        if next_score > current_score:

            swap_positions.add(i)

    # --------------------------------------------------------
    # Apply non-chaining swaps.
    # --------------------------------------------------------

    output = []

    i = 0

    while i < len(original):

        if (
            i in swap_positions
            and i + 1 < len(original)
            and (i + 1)
            not in swap_positions
        ):

            output.append(
                original[i + 1]
            )

            output.append(
                original[i]
            )

            i += 2

        else:

            output.append(
                original[i]
            )

            i += 1

    return output


# ============================================================
# Metrics
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


def hit_at_k(
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


def recall_at_k(
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


def mrr(
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
# Build one E2 baseline
# ============================================================

def build_e2_baseline(dataset):

    baseline = {}

    print()
    print("=" * 90)
    print("BUILDING E2 BASELINE")
    print("=" * 90)

    for item in dataset:

        question_id = item.get("id")

        question = item.get(
            "question",
            "",
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
        # E2 source diversity
        # ----------------------------------------------------

        e2_results = source_diversity(
            candidates,
            limit=TOP_K,
            max_chunks_per_source=(
                MAX_CHUNKS_PER_SOURCE
            ),
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        baseline[question_id] = {
            "question": question,
            "e2_results": e2_results,
            "vector_latency_ms": latency_ms,
        }

    print(
        f"E2 baseline built for "
        f"{len(baseline)} questions."
    )

    return baseline


# ============================================================
# Run E7B from SAME E2 candidate sets
# ============================================================

def run_experiment(
    dataset,
    e2_baseline,
    experiment_name,
    alpha,
    semantic_gap,
):

    details = []

    latencies = []

    hit1 = []
    hit3 = []
    hit5 = []
    hit10 = []

    recall10 = []
    mrr_values = []

    same_candidate_set_count = 0
    changed_candidate_set_count = 0

    for item in dataset:

        question_id = item.get("id")

        question = item.get(
            "question",
            "",
        )

        relevant_sources = (
            get_relevant_sources(item)
        )

        baseline_item = e2_baseline[
            question_id
        ]

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Reuse the EXACT E2 Top10.
        #
        # No vector search here.
        # ----------------------------------------------------

        e2_results = list(
            baseline_item[
                "e2_results"
            ]
        )

        start = time.perf_counter()

        final_results = (
            local_lexical_correction(
                query=question,
                results=e2_results,
                alpha=alpha,
                semantic_gap=semantic_gap,
            )
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            latency_ms
        )

        # ----------------------------------------------------
        # Candidate-set verification
        # ----------------------------------------------------

        e2_set = candidate_set(
            e2_results
        )

        e7b_set = candidate_set(
            final_results
        )

        candidate_set_preserved = (
            e2_set == e7b_set
        )

        if candidate_set_preserved:

            same_candidate_set_count += 1

        else:

            changed_candidate_set_count += 1

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        h1 = hit_at_k(
            final_results,
            relevant_sources,
            1,
        )

        h3 = hit_at_k(
            final_results,
            relevant_sources,
            3,
        )

        h5 = hit_at_k(
            final_results,
            relevant_sources,
            5,
        )

        h10 = hit_at_k(
            final_results,
            relevant_sources,
            10,
        )

        r10 = recall_at_k(
            final_results,
            relevant_sources,
            10,
        )

        mrr_value = mrr(
            final_results,
            relevant_sources,
        )

        hit1.append(h1)
        hit3.append(h3)
        hit5.append(h5)
        hit10.append(h10)

        recall10.append(r10)
        mrr_values.append(
            mrr_value
        )

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
                "alpha": alpha,
                "semantic_gap": semantic_gap,
                "relevant_sources": list(
                    relevant_sources
                ),
                "candidate_set_preserved": (
                    candidate_set_preserved
                ),
                "e2_candidate_ids": [
                    result_identity(result)
                    for result in e2_results
                ],
                "e7b_candidate_ids": [
                    result_identity(result)
                    for result in final_results
                ],
                "retrieved_sources": [
                    get_source(result)
                    for result in final_results
                ],
                "retrieved_details": (
                    retrieved_details
                ),
                "lexical_latency_ms": (
                    latency_ms
                ),
                "metrics": {
                    "hit@1": h1,
                    "hit@3": h3,
                    "hit@5": h5,
                    "hit@10": h10,
                    "recall@10": r10,
                    "mrr": mrr_value,
                },
            }
        )

    # ========================================================
    # Aggregate
    # ========================================================

    def mean(values):

        if not values:
            return 0.0

        return sum(values) / len(values)

    sorted_latencies = sorted(
        latencies
    )

    if sorted_latencies:

        median_latency = (
            statistics.median(
                sorted_latencies
            )
        )

        p95_index = min(
            len(sorted_latencies) - 1,
            int(
                len(sorted_latencies)
                * 0.95
            ),
        )

        p95_latency = (
            sorted_latencies[
                p95_index
            ]
        )

    else:

        median_latency = 0.0
        p95_latency = 0.0

    metrics = {
        "hit@1": mean(hit1),
        "hit@3": mean(hit3),
        "hit@5": mean(hit5),
        "hit@10": mean(hit10),
        "recall@10": mean(
            recall10
        ),
        "mrr": mean(
            mrr_values
        ),
        "questions": len(dataset),
        "hits@10": sum(hit10),
        "misses@10": (
            len(dataset)
            - sum(hit10)
        ),
        "avg_lexical_latency_ms": (
            mean(latencies)
        ),
        "median_lexical_latency_ms": (
            median_latency
        ),
        "p95_lexical_latency_ms": (
            p95_latency
        ),
        "same_candidate_set": (
            same_candidate_set_count
        ),
        "changed_candidate_set": (
            changed_candidate_set_count
        ),
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
    print(
        "E7B LOCAL LEXICAL ON EXACT E2 TOP10"
    )
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
        f"Alpha: {ALPHA}"
    )

    print(
        f"Semantic gaps: "
        f"{SEMANTIC_GAPS}"
    )

    # ========================================================
    # Build E2 ONCE
    # ========================================================

    e2_baseline = build_e2_baseline(
        dataset
    )

    all_summary = []

    all_details = {}

    # ========================================================
    # Run E7B variants
    # ========================================================

    for semantic_gap in SEMANTIC_GAPS:

        experiment_name = (
            f"E7B_gap_{semantic_gap:.2f}"
        )

        print()
        print("-" * 90)
        print(
            experiment_name
        )
        print("-" * 90)

        metrics, details = (
            run_experiment(
                dataset=dataset,
                e2_baseline=e2_baseline,
                experiment_name=(
                    experiment_name
                ),
                alpha=ALPHA,
                semantic_gap=(
                    semantic_gap
                ),
            )
        )

        all_summary.append(
            {
                "experiment": (
                    experiment_name
                ),
                "alpha": ALPHA,
                "semantic_gap": (
                    semantic_gap
                ),
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
            f"Same set  : "
            f"{metrics['same_candidate_set']}/"
            f"{len(dataset)}"
        )

        print(
            f"Changed   : "
            f"{metrics['changed_candidate_set']}/"
            f"{len(dataset)}"
        )

        print(
            f"Avg ms    : "
            f"{metrics['avg_lexical_latency_ms']:.2f}"
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
            "baseline": (
                "vector_search_diverse"
            ),
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
            "lexical_stage": (
                "after_source_diversity"
            ),
            "candidate_set_preserved": (
                "verified"
            ),
            "vector_search_per_question": 1,
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
    print("=" * 120)
    print("E7B SUMMARY")
    print("=" * 120)

    print(
        f"{'Exp':<18}"
        f"{'Gap':>8}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MRR':>10}"
        f"{'Same':>10}"
        f"{'Avg ms':>12}"
    )

    print("-" * 120)

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
            f"{metrics['same_candidate_set']:>10}"
            f"{metrics['avg_lexical_latency_ms']:>12.2f}"
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