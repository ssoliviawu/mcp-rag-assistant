import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e21_full_embedding_details.json"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e22_regression_analysis.json"
)


REP_A = "title_content"
REP_B = "title_section_content"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_first_relevant_rank(item):
    rank = item.get("metrics", {}).get("first_relevant_rank")
    if rank is not None:
        return rank

    ranks = item.get("gold_ranks", [])
    if ranks:
        return min(ranks)

    return None


def get_hit_at_1(item):
    return item.get("metrics", {}).get("hit@1", 0)


def get_top1(item):
    top10 = item.get("top10", [])
    if not top10:
        return None
    return top10[0]


def get_gold_results(item):
    return {
        x.get("url", "").rstrip("/") + "/"
        for x in item.get("top10", [])
        if x.get("is_gold") is True
    }


def normalize_url(url):
    if not url:
        return ""
    return url.rstrip("/") + "/"


def get_relevant_sources(item):
    return {
        normalize_url(x)
        for x in item.get("relevant_sources", [])
        if x
    }


def build_top10_source_set(item):
    sources = set()

    for result in item.get("top10", []):
        url = normalize_url(result.get("url", ""))
        if url:
            sources.add(url)

    return sources


def classify_change(rank_a, rank_b):
    if rank_a is None and rank_b is None:
        return "BOTH_MISS"

    if rank_a is None and rank_b is not None:
        return "NEW_HIT"

    if rank_a is not None and rank_b is None:
        return "NEW_MISS"

    if rank_b < rank_a:
        return "IMPROVED"

    if rank_b > rank_a:
        return "WORSENED"

    return "UNCHANGED"


def main():
    data = load_json(INPUT_PATH)

    questions = data["questions"]

    a_items = {
        item["id"]: item
        for item in questions[REP_A]
    }

    b_items = {
        item["id"]: item
        for item in questions[REP_B]
    }

    common_ids = sorted(
        set(a_items.keys()) & set(b_items.keys())
    )

    print("=" * 80)
    print("E22 - E21 REPRESENTATION REGRESSION ANALYSIS")
    print("=" * 80)

    print(f"\nRepresentation A : {REP_A}")
    print(f"Representation B : {REP_B}")
    print(f"Questions         : {len(common_ids)}")

    improved = []
    worsened = []
    unchanged = []
    new_hit = []
    new_miss = []

    details = []

    for qid in common_ids:
        a = a_items[qid]
        b = b_items[qid]

        rank_a = get_first_relevant_rank(a)
        rank_b = get_first_relevant_rank(b)

        change = classify_change(rank_a, rank_b)

        top1_a = get_top1(a)
        top1_b = get_top1(b)

        record = {
            "id": qid,
            "question": a.get("question"),
            "representation_a": REP_A,
            "representation_b": REP_B,
            "rank_a": rank_a,
            "rank_b": rank_b,
            "rank_delta": (
                None
                if rank_a is None or rank_b is None
                else rank_a - rank_b
            ),
            "change": change,
            "hit_at_1_a": get_hit_at_1(a),
            "hit_at_1_b": get_hit_at_1(b),
            "top1_a": top1_a,
            "top1_b": top1_b,
        }

        details.append(record)

        if change == "IMPROVED":
            improved.append(record)
        elif change == "WORSENED":
            worsened.append(record)
        elif change == "UNCHANGED":
            unchanged.append(record)
        elif change == "NEW_HIT":
            new_hit.append(record)
        elif change == "NEW_MISS":
            new_miss.append(record)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nImproved  : {len(improved)}")
    print(f"Worsened  : {len(worsened)}")
    print(f"Unchanged : {len(unchanged)}")
    print(f"New hit   : {len(new_hit)}")
    print(f"New miss  : {len(new_miss)}")

    # ------------------------------------------------------------------
    # Metric comparison
    # ------------------------------------------------------------------

    def metric_average(rep, metric):
        values = []

        for qid in common_ids:
            value = questions[rep][
                next(
                    i
                    for i, x in enumerate(questions[rep])
                    if x["id"] == qid
                )
            ].get("metrics", {}).get(metric)

            if value is not None:
                values.append(value)

        if not values:
            return None

        return sum(values) / len(values)

    metrics = {}

    for metric in [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
    ]:
        a_value = metric_average(REP_A, metric)
        b_value = metric_average(REP_B, metric)

        metrics[metric] = {
            REP_A: a_value,
            REP_B: b_value,
            "delta": (
                None
                if a_value is None or b_value is None
                else b_value - a_value
            ),
        }

    print("\n" + "=" * 80)
    print("METRIC COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<15}"
        f"{REP_A:>18}"
        f"{REP_B:>25}"
        f"{'Delta':>12}"
    )

    print("-" * 75)

    for metric, values in metrics.items():
        a_value = values[REP_A]
        b_value = values[REP_B]
        delta = values["delta"]

        print(
            f"{metric:<15}"
            f"{a_value:>18.4f}"
            f"{b_value:>25.4f}"
            f"{delta:>12.4f}"
        )

    # ------------------------------------------------------------------
    # Print improved cases
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("IMPROVED CASES")
    print("=" * 80)

    for item in sorted(
        improved,
        key=lambda x: (
            x["rank_a"] if x["rank_a"] is not None else 999,
            x["rank_b"] if x["rank_b"] is not None else 999,
        ),
    ):
        top_a = item["top1_a"] or {}
        top_b = item["top1_b"] or {}

        print(
            f"\n{item['id']} | "
            f"{item['rank_a']} -> {item['rank_b']} | "
            f"{item['question']}"
        )

        print(
            f"  A Top1 : "
            f"{top_a.get('title', '')} | "
            f"{normalize_url(top_a.get('url', ''))}"
        )

        print(
            f"  B Top1 : "
            f"{top_b.get('title', '')} | "
            f"{normalize_url(top_b.get('url', ''))}"
        )

    # ------------------------------------------------------------------
    # Print worsened cases
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("WORSENED CASES")
    print("=" * 80)

    for item in sorted(
        worsened,
        key=lambda x: (
            x["rank_a"] if x["rank_a"] is not None else 999,
            x["rank_b"] if x["rank_b"] is not None else 999,
        ),
    ):
        top_a = item["top1_a"] or {}
        top_b = item["top1_b"] or {}

        print(
            f"\n{item['id']} | "
            f"{item['rank_a']} -> {item['rank_b']} | "
            f"{item['question']}"
        )

        print(
            f"  A Top1 : "
            f"{top_a.get('title', '')} | "
            f"{normalize_url(top_a.get('url', ''))}"
        )

        print(
            f"  B Top1 : "
            f"{top_b.get('title', '')} | "
            f"{normalize_url(top_b.get('url', ''))}"
        )

    # ------------------------------------------------------------------
    # Top1 transition
    # ------------------------------------------------------------------

    transition = {
        "non_gold_to_gold": 0,
        "gold_to_non_gold": 0,
        "gold_to_gold": 0,
        "non_gold_to_non_gold": 0,
    }

    for qid in common_ids:
        a = a_items[qid]
        b = b_items[qid]

        top_a = get_top1(a)
        top_b = get_top1(b)

        if not top_a or not top_b:
            continue

        gold_sources_a = get_relevant_sources(a)
        gold_sources_b = get_relevant_sources(b)

        top_a_gold = (
            normalize_url(top_a.get("url", ""))
            in gold_sources_a
        )

        top_b_gold = (
            normalize_url(top_b.get("url", ""))
            in gold_sources_b
        )

        if not top_a_gold and top_b_gold:
            transition["non_gold_to_gold"] += 1
        elif top_a_gold and not top_b_gold:
            transition["gold_to_non_gold"] += 1
        elif top_a_gold and top_b_gold:
            transition["gold_to_gold"] += 1
        else:
            transition["non_gold_to_non_gold"] += 1

    print("\n" + "=" * 80)
    print("TOP1 TRANSITION")
    print("=" * 80)

    print(
        f"Non-gold -> Gold     : "
        f"{transition['non_gold_to_gold']}"
    )

    print(
        f"Gold -> Non-gold     : "
        f"{transition['gold_to_non_gold']}"
    )

    print(
        f"Gold -> Gold         : "
        f"{transition['gold_to_gold']}"
    )

    print(
        f"Non-gold -> Non-gold : "
        f"{transition['non_gold_to_non_gold']}"
    )

    # ------------------------------------------------------------------
    # Large movements
    # ------------------------------------------------------------------

    large_improvements = [
        x for x in improved
        if x["rank_delta"] is not None
        and x["rank_delta"] >= 3
    ]

    large_regressions = [
        x for x in worsened
        if x["rank_delta"] is not None
        and x["rank_delta"] <= -3
    ]

    print("\n" + "=" * 80)
    print("LARGE IMPROVEMENTS (>= 3 ranks)")
    print("=" * 80)

    for item in sorted(
        large_improvements,
        key=lambda x: x["rank_delta"],
        reverse=True,
    ):
        print(
            f"{item['id']} | "
            f"{item['rank_a']} -> {item['rank_b']} | "
            f"{item['question']}"
        )

    print("\n" + "=" * 80)
    print("LARGE REGRESSIONS (>= 3 ranks)")
    print("=" * 80)

    for item in sorted(
        large_regressions,
        key=lambda x: x["rank_delta"],
    ):
        print(
            f"{item['id']} | "
            f"{item['rank_a']} -> {item['rank_b']} | "
            f"{item['question']}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    output = {
        "experiment": "E22_representation_regression_analysis",
        "source": str(INPUT_PATH),
        "representations": {
            "A": REP_A,
            "B": REP_B,
        },
        "question_count": len(common_ids),
        "summary": {
            "improved": len(improved),
            "worsened": len(worsened),
            "unchanged": len(unchanged),
            "new_hit": len(new_hit),
            "new_miss": len(new_miss),
        },
        "metrics": metrics,
        "top1_transition": transition,
        "large_improvements_count": len(large_improvements),
        "large_regressions_count": len(large_regressions),
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "new_hit": new_hit,
        "new_miss": new_miss,
        "all_questions": details,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("E22 COMPLETE")
    print("=" * 80)

    print(f"\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()