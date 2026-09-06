import json
import statistics
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

E2_DETAILS_PATH = RESULTS_DIR / "retrieval_matrix_details.json"
E6_DETAILS_PATH = RESULTS_DIR / "e6_lexical_alpha_details.json"

OUTPUT_PATH = RESULTS_DIR / "e6_rank_change_analysis.json"


def normalize_source(url: str) -> str:
    if not url:
        return ""

    return url.strip().rstrip("/")


def get_gold_sources(item: dict) -> set[str]:
    sources = item.get("relevant_sources", [])

    if isinstance(sources, str):
        sources = [sources]

    return {
        normalize_source(source)
        for source in sources
        if source
    }


def get_retrieved_sources(item: dict) -> list[str]:
    sources = item.get("retrieved_sources", [])

    if isinstance(sources, list):
        return [
            normalize_source(source)
            for source in sources
            if source
        ]

    return []


def get_first_relevant_rank(
    retrieved_sources: list[str],
    gold_sources: set[str],
) -> int | None:

    if not gold_sources:
        return None

    for rank, source in enumerate(retrieved_sources, start=1):
        if source in gold_sources:
            return rank

    return None


def load_experiment_results(
    path: Path,
    experiment: str,
) -> dict[str, dict]:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected JSON root type: {type(data)}"
        )

    results = data.get("results")

    if not isinstance(results, dict):
        raise ValueError(
            f"'results' is not a dict in {path}"
        )

    experiment_results = results.get(experiment)

    if experiment_results is None:
        raise ValueError(
            f"Experiment '{experiment}' not found in {path}. "
            f"Available experiments: {list(results.keys())}"
        )

    if not isinstance(experiment_results, list):
        raise ValueError(
            f"results['{experiment}'] is not a list"
        )

    output = {}

    for item in experiment_results:
        if not isinstance(item, dict):
            continue

        qid = item.get("id")

        if qid:
            output[qid] = item

    return output


def analyze():
    print("=" * 90)
    print("E2 → E6(alpha=0.05) RANK CHANGE ANALYSIS")
    print("=" * 90)

    print(f"E2 details: {E2_DETAILS_PATH}")
    print(f"E6 details: {E6_DETAILS_PATH}")
    print()

    # ------------------------------------------------------------------
    # Load exact experiment blocks
    # ------------------------------------------------------------------

    e2 = load_experiment_results(
        E2_DETAILS_PATH,
        "E2",
    )

    e6 = load_experiment_results(
        E6_DETAILS_PATH,
        "E6_05",
    )

    common_ids = sorted(
        set(e2.keys()) & set(e6.keys())
    )

    print(f"E2 questions : {len(e2)}")
    print(f"E6_05 questions: {len(e6)}")
    print(f"Common questions: {len(common_ids)}")
    print()

    # ------------------------------------------------------------------
    # Analyze rank changes
    # ------------------------------------------------------------------

    improved = []
    worsened = []
    unchanged = []

    e6_top1_hits = []
    e6_top1_misses = []

    rank_changes = []

    for qid in common_ids:

        e2_item = e2[qid]
        e6_item = e6[qid]

        gold_sources = get_gold_sources(e6_item)

        if not gold_sources:
            gold_sources = get_gold_sources(e2_item)

        e2_sources = get_retrieved_sources(e2_item)
        e6_sources = get_retrieved_sources(e6_item)

        e2_rank = get_first_relevant_rank(
            e2_sources,
            gold_sources,
        )

        e6_rank = get_first_relevant_rank(
            e6_sources,
            gold_sources,
        )

        # Positive = improved
        # Negative = worsened
        if e2_rank is not None and e6_rank is not None:
            rank_change = e2_rank - e6_rank
        elif e2_rank is None and e6_rank is not None:
            rank_change = 999
        elif e2_rank is not None and e6_rank is None:
            rank_change = -999
        else:
            rank_change = 0

        record = {
            "id": qid,
            "question": e6_item.get(
                "question",
                e2_item.get("question", ""),
            ),
            "gold_sources": sorted(gold_sources),
            "e2_rank": e2_rank,
            "e6_rank": e6_rank,
            "rank_change": rank_change,
            "e2_top1_source": (
                e2_sources[0]
                if e2_sources
                else None
            ),
            "e6_top1_source": (
                e6_sources[0]
                if e6_sources
                else None
            ),
        }

        if e2_rank is not None and e6_rank is not None:
            rank_changes.append(rank_change)

        if rank_change > 0:
            improved.append(record)

        elif rank_change < 0:
            worsened.append(record)

        else:
            unchanged.append(record)

        # E6 Top1
        if e6_rank == 1:
            e6_top1_hits.append(record)
        else:
            e6_top1_misses.append(record)

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------

    improved.sort(
        key=lambda x: (
            -x["rank_change"],
            x["id"],
        )
    )

    worsened.sort(
        key=lambda x: (
            x["rank_change"],
            x["id"],
        )
    )

    unchanged.sort(
        key=lambda x: x["id"]
    )

    e6_top1_misses.sort(
        key=lambda x: (
            x["e6_rank"] is None,
            x["e6_rank"]
            if x["e6_rank"] is not None
            else 999,
            x["id"],
        )
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    total = len(common_ids)

    summary = {
        "total_questions": total,

        "rank_change_counts": {
            "improved": len(improved),
            "worsened": len(worsened),
            "unchanged": len(unchanged),
        },

        "e6_top1": {
            "hits": len(e6_top1_hits),
            "misses": len(e6_top1_misses),
            "hit_rate": (
                len(e6_top1_hits) / total
                if total
                else 0
            ),
        },

        "rank_change_statistics": {
            "average_rank_change": (
                statistics.mean(rank_changes)
                if rank_changes
                else 0
            ),
            "median_rank_change": (
                statistics.median(rank_changes)
                if rank_changes
                else 0
            ),
        },
    }

    output = {
        "experiment": "E6_05",

        "description": (
            "Compare E2 vector + source diversity "
            "against E6 lexical boost alpha=0.05"
        ),

        "baseline": {
            "experiment": "E2",
            "candidate_limit": 100,
            "top_k": 10,
            "max_chunks_per_source": 1,
        },

        "e6": {
            "experiment": "E6_05",
            "candidate_limit": 100,
            "top_k": 10,
            "alpha": 0.05,
            "max_chunks_per_source": 1,
        },

        "summary": summary,

        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,

        "e6_top1_hits": e6_top1_hits,
        "e6_top1_misses": e6_top1_misses,
    }

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"Improved : {len(improved)}"
    )
    print(
        f"Worsened : {len(worsened)}"
    )
    print(
        f"Unchanged: {len(unchanged)}"
    )

    print()

    print(
        f"E6 Top1 hits : "
        f"{len(e6_top1_hits)}/{total}"
    )

    print(
        f"E6 Top1 misses: "
        f"{len(e6_top1_misses)}/{total}"
    )

    print()

    print("=" * 90)
    print("IMPROVED")
    print("=" * 90)

    if improved:
        for item in improved:
            print(
                f"{item['id']} "
                f"{item['e2_rank']} → {item['e6_rank']} "
                f"(Δ={item['rank_change']:+d}) "
                f"| {item['question']}"
            )
    else:
        print("None")

    print()

    print("=" * 90)
    print("WORSENED")
    print("=" * 90)

    if worsened:
        for item in worsened:
            print(
                f"{item['id']} "
                f"{item['e2_rank']} → {item['e6_rank']} "
                f"(Δ={item['rank_change']:+d}) "
                f"| {item['question']}"
            )
    else:
        print("None")

    print()

    print("=" * 90)
    print("UNCHANGED")
    print("=" * 90)

    print(
        f"{len(unchanged)} questions"
    )

    print()

    print("=" * 90)
    print("E6 TOP1 MISS")
    print("=" * 90)

    if e6_top1_misses:
        for item in e6_top1_misses:
            print(
                f"{item['id']} "
                f"E2={item['e2_rank']} "
                f"E6={item['e6_rank']} "
                f"| {item['question']}"
            )
    else:
        print("None")

    print()

    print("=" * 90)
    print("FILE SAVED")
    print("=" * 90)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    analyze()