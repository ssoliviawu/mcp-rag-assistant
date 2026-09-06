import json
from pathlib import Path

from retrieval.search import (
    vector_search_diverse,
    vector_rerank_dedup,
)


# ============================================================
# CONFIG
# ============================================================

DATASET = Path("mcp_rag_eval_dataset.jsonl")

TOP_K = 10

CANDIDATE_K = 100

MAX_CHUNKS_PER_SOURCE = 1

OUTPUT_FILE = Path(
    "evaluation/results/"
    "reranker_overall_analysis.json"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    records = []

    with open(
        DATASET,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):

    if not url:
        return ""

    return url.rstrip("/")


# ============================================================
# RELEVANCE
# ============================================================

def is_relevant(
    result,
    relevant_sources,
):

    result_url = normalize_url(
        result.get("url", "")
    )

    for source in relevant_sources:

        if (
            result_url
            == normalize_url(source)
        ):
            return True

    return False


# ============================================================
# FIND TARGET RANK
# ============================================================

def find_target_rank(
    results,
    relevant_sources,
):

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if is_relevant(
            result,
            relevant_sources,
        ):
            return rank

    return None


# ============================================================
# METRICS
# ============================================================

def reciprocal_rank(rank):

    if rank is None:
        return 0.0

    return 1.0 / rank


def calculate_metrics(
    results,
    rank_field,
):

    total = len(results)

    metrics = {}

    for k in [1, 3, 5, 10]:

        hits = sum(
            1
            for result in results
            if (
                result[rank_field] is not None
                and result[rank_field] <= k
            )
        )

        metrics[f"hit@{k}"] = (
            hits / total
        )

    metrics["mrr"] = (
        sum(
            reciprocal_rank(
                result[rank_field]
            )
            for result in results
        )
        / total
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "VECTOR DIVERSE vs "
        "RERANKER + DEDUP ANALYSIS"
    )
    print("=" * 80)

    dataset = load_dataset()

    print(
        f"Questions: {len(dataset)}"
    )

    print()

    print(
        "Pipeline A:"
    )
    print(
        "  Vector Search "
        "-> Source Dedup"
    )

    print()

    print(
        "Pipeline B:"
    )
    print(
        "  Vector Search "
        "-> ONNX Reranker "
        "-> Source Dedup"
    )

    print()

    print(
        f"Candidate K       : {CANDIDATE_K}"
    )

    print(
        f"Final Top K       : {TOP_K}"
    )

    print(
        f"Max chunks/source : "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )

    print()

    # --------------------------------------------------------
    # PROCESS QUESTIONS
    # --------------------------------------------------------

    analysis_results = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        qid = item["id"]

        question = item["question"]

        relevant_sources = (
            item["relevant_sources"]
        )

        print(
            f"[{index}/{len(dataset)}] "
            f"{qid}"
        )

        print(
            f"  {question}"
        )

        # ====================================================
        # PIPELINE A
        #
        # Vector → Diverse/Dedup
        # ====================================================

        vector_results = (
            vector_search_diverse(
                question,
                limit=TOP_K,
                candidate_limit=CANDIDATE_K,
                max_chunks_per_source=(
                    MAX_CHUNKS_PER_SOURCE
                ),
            )
        )

        vector_rank = find_target_rank(
            vector_results,
            relevant_sources,
        )

        # ====================================================
        # PIPELINE B
        #
        # Vector → Reranker → Dedup
        # ====================================================

        reranker_results = (
            vector_rerank_dedup(
                question,
                limit=TOP_K,
                candidate_limit=CANDIDATE_K,
            )
        )

        reranker_rank = find_target_rank(
            reranker_results,
            relevant_sources,
        )

        # ====================================================
        # RANK CHANGE
        # ====================================================

        if vector_rank is None:

            direction = "VECTOR_DIVERSE_MISS"

            rank_change = None

        elif reranker_rank is None:

            direction = "RERANKER_MISS"

            rank_change = None

        else:

            # Positive = moved UP
            # Negative = moved DOWN

            rank_change = (
                vector_rank
                - reranker_rank
            )

            if rank_change > 0:

                direction = "IMPROVED"

            elif rank_change < 0:

                direction = "WORSENED"

            else:

                direction = "UNCHANGED"

        analysis_results.append(
            {
                "id": qid,
                "question": question,
                "vector_diverse_rank": (
                    vector_rank
                ),
                "reranker_rank": (
                    reranker_rank
                ),
                "rank_change": rank_change,
                "direction": direction,
            }
        )

        print(
            f"  vector diverse rank : "
            f"{vector_rank}"
        )

        print(
            f"  reranker rank       : "
            f"{reranker_rank}"
        )

        print(
            f"  result              : "
            f"{direction}"
        )

        print()

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(
        analysis_results
    )

    improved = sum(
        1
        for r in analysis_results
        if r["direction"]
        == "IMPROVED"
    )

    unchanged = sum(
        1
        for r in analysis_results
        if r["direction"]
        == "UNCHANGED"
    )

    worsened = sum(
        1
        for r in analysis_results
        if r["direction"]
        == "WORSENED"
    )

    vector_miss = sum(
        1
        for r in analysis_results
        if r["direction"]
        == "VECTOR_DIVERSE_MISS"
    )

    reranker_miss = sum(
        1
        for r in analysis_results
        if r["direction"]
        == "RERANKER_MISS"
    )

    # ========================================================
    # METRICS
    # ========================================================

    vector_metrics = (
        calculate_metrics(
            analysis_results,
            "vector_diverse_rank",
        )
    )

    reranker_metrics = (
        calculate_metrics(
            analysis_results,
            "reranker_rank",
        )
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "RANK CHANGE SUMMARY"
    )
    print("=" * 80)

    print(
        f"Total questions : {total}"
    )

    print(
        f"Improved        : {improved}"
    )

    print(
        f"Unchanged       : {unchanged}"
    )

    print(
        f"Worsened        : {worsened}"
    )

    print(
        f"Vector miss     : {vector_miss}"
    )

    print(
        f"Reranker miss   : {reranker_miss}"
    )

    # ========================================================
    # METRIC COMPARISON
    # ========================================================

    print()
    print("=" * 80)
    print(
        "METRICS COMPARISON"
    )
    print("=" * 80)

    print(
        f"{'Metric':<12}"
        f"{'Diverse':>12}"
        f"{'Reranker':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 48)

    for metric in [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "mrr",
    ]:

        vector_value = (
            vector_metrics[metric]
        )

        reranker_value = (
            reranker_metrics[metric]
        )

        delta = (
            reranker_value
            - vector_value
        )

        print(
            f"{metric:<12}"
            f"{vector_value:>12.4f}"
            f"{reranker_value:>12.4f}"
            f"{delta:>12.4f}"
        )

    # ========================================================
    # WORST CASES
    # ========================================================

    worsened_cases = [
        r
        for r in analysis_results
        if r["direction"]
        == "WORSENED"
    ]

    worsened_cases.sort(
        key=lambda x: x["rank_change"]
    )

    print()
    print("=" * 80)
    print(
        "WORST RERANKER CASES"
    )
    print("=" * 80)

    if not worsened_cases:

        print("None.")

    else:

        for r in worsened_cases:

            print(
                f"{r['id']} | "
                f"diverse="
                f"{r['vector_diverse_rank']} "
                f"-> "
                f"reranker="
                f"{r['reranker_rank']} "
                f"| change="
                f"{r['rank_change']}"
            )

            print(
                f"  {r['question']}"
            )

    # ========================================================
    # BEST CASES
    # ========================================================

    improved_cases = [
        r
        for r in analysis_results
        if r["direction"]
        == "IMPROVED"
    ]

    improved_cases.sort(
        key=lambda x: x["rank_change"],
        reverse=True,
    )

    print()
    print("=" * 80)
    print(
        "BEST RERANKER CASES"
    )
    print("=" * 80)

    if not improved_cases:

        print("None.")

    else:

        for r in improved_cases[:10]:

            print(
                f"{r['id']} | "
                f"diverse="
                f"{r['vector_diverse_rank']} "
                f"-> "
                f"reranker="
                f"{r['reranker_rank']} "
                f"| change="
                f"{r['rank_change']}"
            )

            print(
                f"  {r['question']}"
            )

    # ========================================================
    # SAVE JSON
    # ========================================================

    output = {

        "total_questions": total,

        "pipeline": {
            "baseline": (
                "vector_search_diverse"
            ),
            "reranker": (
                "vector_rerank_dedup"
            ),
            "candidate_limit": (
                CANDIDATE_K
            ),
            "top_k": TOP_K,
            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),
        },

        "rank_change": {

            "improved": improved,

            "unchanged": unchanged,

            "worsened": worsened,

            "vector_diverse_miss": (
                vector_miss
            ),

            "reranker_miss": (
                reranker_miss
            ),
        },

        "metrics": {

            "vector_diverse": (
                vector_metrics
            ),

            "reranker": (
                reranker_metrics
            ),
        },

        "questions": analysis_results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        OUTPUT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()