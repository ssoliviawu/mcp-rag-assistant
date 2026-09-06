import json
from pathlib import Path


INPUT = Path("evaluation/results/e27_input.json")
OUTPUT_JSON = Path(
    "evaluation/results/e28_migration_analysis.json"
)


MIGRATION_SOURCE = (
    "https://py.sdk.modelcontextprotocol.io/migration/"
)


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


def get_section(item):
    return (
        item.get("section")
        or item.get("metadata", {}).get("section")
        or ""
    )


def get_content(item):
    return (
        item.get("content")
        or item.get("text")
        or ""
    )


def get_score(item):
    for key in (
        "similarity",
        "score",
        "vector_score",
    ):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    return None


def normalize_source(source):
    return source.rstrip("/")


def is_migration(item):
    return (
        normalize_source(get_source(item))
        == MIGRATION_SOURCE.rstrip("/")
    )


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    migration_cases = []
    all_migration_top10 = []

    for q in questions:
        top10 = q.get("top10", [])

        if not top10:
            continue

        top1 = top10[0]

        if not is_migration(top1):
            continue

        gold_rank = q.get("gold_rank")

        gold = None
        if isinstance(gold_rank, int):
            index = gold_rank - 1
            if 0 <= index < len(top10):
                gold = top10[index]

        top1_score = get_score(top1)
        gold_score = get_score(gold) if gold else None

        margin = None
        if (
            top1_score is not None
            and gold_score is not None
        ):
            margin = top1_score - gold_score

        case = {
            "id": q.get("id"),
            "question": q.get("question"),
            "gold_rank": gold_rank,
            "margin": margin,

            "migration_top1": {
                "title": get_title(top1),
                "section": get_section(top1),
                "source": get_source(top1),
                "similarity": top1_score,
                "content": get_content(top1),
            },

            "gold": {
                "title": get_title(gold) if gold else "",
                "section": get_section(gold) if gold else "",
                "source": get_source(gold) if gold else "",
                "similarity": gold_score,
                "content": get_content(gold) if gold else "",
            },

            "migration_chunks_in_top10": [
                {
                    "rank": rank,
                    "title": get_title(item),
                    "section": get_section(item),
                    "similarity": get_score(item),
                }
                for rank, item in enumerate(top10, start=1)
                if is_migration(item)
            ],
        }

        migration_cases.append(case)

        for rank, item in enumerate(top10, start=1):
            if is_migration(item):
                all_migration_top10.append({
                    "id": q.get("id"),
                    "rank": rank,
                    "title": get_title(item),
                    "section": get_section(item),
                    "similarity": get_score(item),
                })

    output = {
        "configuration": data.get("configuration"),
        "migration_top1_count": len(migration_cases),
        "migration_top1_cases": migration_cases,
        "all_migration_top10": all_migration_top10,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = []

    lines.append("=" * 80)
    lines.append("E28-A - MIGRATION DOMINANCE ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"Migration Top1 cases: {len(migration_cases)}"
    )
    lines.append("")

    for case in migration_cases:
        lines.append("-" * 80)
        lines.append(
            f"{case['id']} | "
            f"gold_rank={case['gold_rank']} | "
            f"margin={case['margin']}"
        )
        lines.append(
            f"Q: {case['question']}"
        )

        migration = case["migration_top1"]

        lines.append("")
        lines.append("MIGRATION TOP1")
        lines.append(
            f"Title      : {migration['title']}"
        )
        lines.append(
            f"Section    : {migration['section']}"
        )
        lines.append(
            f"Similarity : {migration['similarity']}"
        )
        lines.append(
            f"Source     : {migration['source']}"
        )

        lines.append("")
        lines.append("MIGRATION CONTENT")
        lines.append(
            migration["content"]
        )

        gold = case["gold"]

        lines.append("")
        lines.append("GOLD")
        lines.append(
            f"Title      : {gold['title']}"
        )
        lines.append(
            f"Section    : {gold['section']}"
        )
        lines.append(
            f"Similarity : {gold['similarity']}"
        )
        lines.append(
            f"Source     : {gold['source']}"
        )

        lines.append("")
        lines.append("GOLD CONTENT")
        lines.append(
            gold["content"]
        )

        lines.append("")
        lines.append("MIGRATION CHUNKS IN TOP10")

        for item in case["migration_chunks_in_top10"]:
            lines.append(
                f"rank={item['rank']} | "
                f"score={item['similarity']} | "
                f"title={item['title']} | "
                f"section={item['section']}"
            )

        lines.append("")

    with OUTPUT_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 80)
    print("E28-A COMPLETE")
    print("=" * 80)
    print(
        f"Migration Top1 cases: {len(migration_cases)}"
    )
    print("")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"TXT : {OUTPUT_TXT}")

    print("")
    print("Migration Top1:")
    for case in migration_cases:
        migration = case["migration_top1"]
        print(
            f"{case['id']} | "
            f"gold={case['gold_rank']} | "
            f"margin={case['margin']:.6f} | "
            f"{migration['title']}"
        )


if __name__ == "__main__":
    main()