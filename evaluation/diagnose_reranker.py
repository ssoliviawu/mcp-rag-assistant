import json
from pathlib import Path

from retrieval.search import vector_search
from retrieval.reranker import rerank


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "reranker_diagnostic.json"
)


# ============================================================
# Configuration
# ============================================================

CANDIDATE_LIMIT = 100
TOP_K = 10


# ============================================================
# Dataset
# ============================================================

def load_dataset(path: Path):

    dataset = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # Skip specification questions
            if item.get("domain") == "spec":
                continue

            dataset.append(item)

    return dataset


# ============================================================
# Normalization
# ============================================================

def normalize_source(source: str) -> str:

    return (
        source
        .strip()
        .rstrip("/")
        .lower()
    )


# ============================================================
# Build rank map
# ============================================================

def build_rank_map(results):

    rank_map = {}

    for rank, result in enumerate(
        results,
        start=1,
    ):

        source = normalize_source(
            result.get("url", "")
        )

        if source and source not in rank_map:

            rank_map[source] = {
                "rank": rank,
                "title": result.get(
                    "title",
                    "",
                ),
                "section": result.get(
                    "section",
                    "",
                ),
                "similarity": result.get(
                    "similarity"
                ),
                "reranker_score": result.get(
                    "reranker_score"
                ),
                "content": result.get(
                    "content",
                    "",
                ),
            }

    return rank_map


# ============================================================
# Analyze one question
# ============================================================

def analyze_question(item):

    query = item["question"]

    relevant_sources = {
        normalize_source(source)
        for source in item[
            "relevant_sources"
        ]
    }

    # --------------------------------------------------------
    # Vector
    # --------------------------------------------------------

    vector_results = vector_search(
        query=query,
        limit=CANDIDATE_LIMIT,
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    reranked_results = rerank(
        query,
        vector_results,
    )

    vector_top10 = vector_results[:TOP_K]

    reranked_top10 = reranked_results[:TOP_K]

    vector_rank_map = build_rank_map(
        vector_results
    )

    reranked_rank_map = build_rank_map(
        reranked_results
    )

    # --------------------------------------------------------
    # Find relevant sources
    # --------------------------------------------------------

    relevant_analysis = []

    for source in relevant_sources:

        vector_info = vector_rank_map.get(
            source
        )

        reranked_info = reranked_rank_map.get(
            source
        )

        relevant_analysis.append(
            {
                "source": source,

                "vector_rank": (
                    vector_info["rank"]
                    if vector_info
                    else None
                ),

                "reranker_rank": (
                    reranked_info["rank"]
                    if reranked_info
                    else None
                ),

                "rank_change": (
                    (
                        reranked_info["rank"]
                        - vector_info["rank"]
                    )
                    if vector_info
                    and reranked_info
                    else None
                ),

                "vector_similarity": (
                    vector_info.get(
                        "similarity"
                    )
                    if vector_info
                    else None
                ),

                "reranker_score": (
                    reranked_info.get(
                        "reranker_score"
                    )
                    if reranked_info
                    else None
                ),
            }
        )

    # --------------------------------------------------------
    # Determine top1
    # --------------------------------------------------------

    vector_top1_source = (
        normalize_source(
            vector_results[0].get(
                "url",
                "",
            )
        )
        if vector_results
        else None
    )

    reranker_top1_source = (
        normalize_source(
            reranked_results[0].get(
                "url",
                "",
            )
        )
        if reranked_results
        else None
    )

    vector_top1_hit = (
        vector_top1_source
        in relevant_sources
    )

    reranker_top1_hit = (
        reranker_top1_source
        in relevant_sources
    )

    # --------------------------------------------------------
    # Relevant in Top-K
    # --------------------------------------------------------

    vector_top10_sources = {
        normalize_source(
            r.get("url", "")
        )
        for r in vector_top10
    }

    reranker_top10_sources = {
        normalize_source(
            r.get("url", "")
        )
        for r in reranked_top10
    }

    vector_hit = bool(
        vector_top10_sources
        & relevant_sources
    )

    reranker_hit = bool(
        reranker_top10_sources
        & relevant_sources
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if vector_top1_hit and not reranker_top1_hit:

        category = "A_TOP1_BROKEN"

    elif vector_hit and not reranker_hit:

        category = "C_TOP10_BROKEN"

    elif not vector_hit and reranker_hit:

        category = "D_FIXED"

    elif vector_hit and reranker_hit:

        category = "E_BOTH_HIT"

    else:

        category = "F_BOTH_MISS"

    return {
        "id": item["id"],
        "question": query,

        "relevant_sources": list(
            relevant_sources
        ),

        "category": category,

        "vector_top1_hit": vector_top1_hit,
        "reranker_top1_hit": reranker_top1_hit,

        "vector_top10_hit": vector_hit,
        "reranker_top10_hit": reranker_hit,

        "relevant_analysis": (
            relevant_analysis
        ),

        "vector_top10": [
            {
                "rank": rank,
                "title": result.get(
                    "title",
                    "",
                ),
                "section": result.get(
                    "section",
                    "",
                ),
                "url": result.get(
                    "url",
                    "",
                ),
                "similarity": result.get(
                    "similarity"
                ),
            }
            for rank, result in enumerate(
                vector_top10,
                start=1,
            )
        ],

        "reranker_top10": [
            {
                "rank": rank,
                "title": result.get(
                    "title",
                    "",
                ),
                "section": result.get(
                    "section",
                    "",
                ),
                "url": result.get(
                    "url",
                    "",
                ),
                "similarity": result.get(
                    "similarity"
                ),
                "reranker_score": result.get(
                    "reranker_score"
                ),
            }
            for rank, result in enumerate(
                reranked_top10,
                start=1,
            )
        ],
    }


# ============================================================
# Summary
# ============================================================

def calculate_summary(results):

    summary = {}

    for category in [
        "A_TOP1_BROKEN",
        "C_TOP10_BROKEN",
        "D_FIXED",
        "E_BOTH_HIT",
        "F_BOTH_MISS",
    ]:

        summary[category] = sum(
            1
            for result in results
            if result["category"] == category
        )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    dataset = load_dataset(
        DATASET_PATH
    )

    print()
    print("=" * 80)
    print("RERANKER DIAGNOSTIC")
    print("=" * 80)

    print(
        f"Questions: {len(dataset)}"
    )

    results = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        print(
            f"[{index:03d}/{len(dataset):03d}] "
            f"{item['id']}"
        )

        result = analyze_question(
            item
        )

        results.append(result)

    summary = calculate_summary(
        results
    )

    output = {
        "config": {
            "candidate_limit": CANDIDATE_LIMIT,
            "top_k": TOP_K,
        },

        "summary": summary,

        "results": results,
    }

    with OUTPUT_PATH.open(
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
    print("SUMMARY")
    print("=" * 80)

    for key, value in summary.items():

        print(
            f"{key:<20} {value}"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()