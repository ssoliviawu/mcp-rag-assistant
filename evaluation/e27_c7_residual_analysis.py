import json
from pathlib import Path
from urllib.parse import urlparse


INPUT = Path("evaluation/results/e27_input.json")
OUTPUT_JSON = Path("evaluation/results/e27_c7_residual_analysis.json")


def normalize_source(url):
    if not url:
        return ""

    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def get_source(item):
    return (
        item.get("source")
        or item.get("url")
        or item.get("source_url")
        or item.get("metadata", {}).get("source")
        or item.get("metadata", {}).get("url")
        or ""
    )


def get_title(item):
    return (
        item.get("title")
        or item.get("metadata", {}).get("title")
        or ""
    )


def get_similarity(item):
    for key in (
        "similarity",
        "score",
        "vector_score",
        "distance",
    ):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    return None


def classify_case(gold_rank, margin, same_source):
    """
    First-pass diagnostic classification.

    IMPORTANT:
    This is deliberately conservative.
    We do not claim semantic failure merely because the gold
    chunk is below rank 1.
    """

    if margin is not None and margin <= 0.01:
        return "NEAR_TIE"

    if same_source:
        return "SAME_SOURCE_COMPETITION"

    if gold_rank >= 4:
        return "LOWER_RANK_RETRIEVAL"

    return "WRONG_DOCUMENT_COMPETITION"


def analyze_question(q):
    top10 = q.get("top10", [])

    if not top10:
        return {
            "id": q.get("id"),
            "question": q.get("question"),
            "gold_rank": q.get("gold_rank"),
            "diagnosis": "NO_TOP10_DATA",
        }

    top1 = top10[0]

    gold_rank = q.get("gold_rank")

    gold_item = None
    if isinstance(gold_rank, int) and 1 <= gold_rank <= len(top10):
        gold_item = top10[gold_rank - 1]

    top1_score = get_similarity(top1)
    gold_score = get_similarity(gold_item) if gold_item else None

    margin = None
    if top1_score is not None and gold_score is not None:
        margin = top1_score - gold_score

    top1_source = normalize_source(get_source(top1))
    gold_source = normalize_source(get_source(gold_item)) if gold_item else ""

    same_source = (
        bool(top1_source)
        and bool(gold_source)
        and top1_source == gold_source
    )

    diagnosis = classify_case(
        gold_rank=gold_rank,
        margin=margin,
        same_source=same_source,
    )

    return {
        "id": q.get("id"),
        "question": q.get("question"),
        "gold_rank": gold_rank,

        "diagnosis": diagnosis,

        "top1": {
            "title": get_title(top1),
            "source": get_source(top1),
            "similarity": top1_score,
        },

        "gold": {
            "title": get_title(gold_item) if gold_item else "",
            "source": get_source(gold_item) if gold_item else "",
            "similarity": gold_score,
        },

        "margin_top1_minus_gold": margin,
        "same_source": same_source,

        "top10": top10,
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    analyses = [
        analyze_question(q)
        for q in questions
    ]

    counts = {}

    for item in analyses:
        diagnosis = item["diagnosis"]
        counts[diagnosis] = counts.get(diagnosis, 0) + 1

    margins = [
        item["margin_top1_minus_gold"]
        for item in analyses
        if item.get("margin_top1_minus_gold") is not None
    ]

    summary = {
        "configuration": data.get("configuration"),
        "total_questions": data.get("total_questions"),
        "non_top1_questions": len(questions),

        "diagnosis_counts": counts,

        "margin": {
            "count": len(margins),
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
            "avg": (
                sum(margins) / len(margins)
                if margins
                else None
            ),
        },

        "questions": analyses,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = []

    lines.append("=" * 80)
    lines.append("E27 - C7 RESIDUAL ERROR ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Configuration : {data.get('configuration')}")
    lines.append(f"Total         : {data.get('total_questions')}")
    lines.append(f"Non-Top1      : {len(questions)}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("DIAGNOSIS COUNTS")
    lines.append("-" * 80)

    for diagnosis, count in sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        lines.append(f"{diagnosis:<30} {count}")

    lines.append("")

    lines.append("-" * 80)
    lines.append("QUESTION-LEVEL ANALYSIS")
    lines.append("-" * 80)

    for item in analyses:
        lines.append("")
        lines.append(
            f"{item['id']} | "
            f"gold_rank={item['gold_rank']} | "
            f"{item['diagnosis']}"
        )
        lines.append(
            f"Q: {item['question']}"
        )

        top1 = item.get("top1", {})
        gold = item.get("gold", {})

        lines.append(
            f"TOP1: {top1.get('title', '')}"
        )
        lines.append(
            f"TOP1 source: {top1.get('source', '')}"
        )
        lines.append(
            f"TOP1 similarity: {top1.get('similarity')}"
        )

        lines.append(
            f"GOLD: {gold.get('title', '')}"
        )
        lines.append(
            f"GOLD source: {gold.get('source', '')}"
        )
        lines.append(
            f"GOLD similarity: {gold.get('similarity')}"
        )

        lines.append(
            f"MARGIN: {item.get('margin_top1_minus_gold')}"
        )
        lines.append(
            f"SAME SOURCE: {item.get('same_source')}"
        )

    with OUTPUT_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 80)
    print("E27 COMPLETE")
    print("=" * 80)

    print(f"Questions analyzed : {len(analyses)}")
    print("")

    for diagnosis, count in sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(f"{diagnosis:<30} {count}")

    print("")
    print(f"JSON : {OUTPUT_JSON}")
    print(f"TXT  : {OUTPUT_TXT}")


if __name__ == "__main__":
    main()