import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e18_representation_details.json"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e20_regression_extract.json"
)

TARGET_IDS = [
    "q002",
    "q006",
    "q034",
    "q043",
]


MODES = [
    "content",
    "title_content",
    "section_content",
    "title_section_content",
]


def load_data():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_question(data, mode, question_id):
    questions = data["questions"][mode]

    for item in questions:
        if item["id"] == question_id:
            return item

    return None


def is_gold(result, relevant_sources):
    return result["url"].rstrip("/") in {
        url.rstrip("/")
        for url in relevant_sources
    }


def simplify_result(result, relevant_sources):
    return {
        "rank": result["rank"],
        "id": result["id"],
        "title": result.get("title"),
        "section": result.get("section"),
        "url": result.get("url"),
        "similarity": result.get("similarity"),
        "original_vector_similarity": result.get(
            "original_vector_similarity"
        ),
        "is_gold": is_gold(result, relevant_sources),
    }


def get_gold_results(item):
    relevant_sources = item["relevant_sources"]

    return [
        simplify_result(r, relevant_sources)
        for r in item["top10"]
        if is_gold(r, relevant_sources)
    ]


def get_top1(item):
    relevant_sources = item["relevant_sources"]

    if not item["top10"]:
        return None

    return simplify_result(
        item["top10"][0],
        relevant_sources,
    )


def get_gold_best(item):
    gold_results = get_gold_results(item)

    if not gold_results:
        return None

    return min(
        gold_results,
        key=lambda x: x["rank"],
    )


def compare_modes(items):
    result = {
        "id": items["content"]["id"],
        "question": items["content"]["question"],
        "relevant_sources": items["content"]["relevant_sources"],
        "modes": {},
        "comparison": {},
    }

    for mode in MODES:
        item = items[mode]

        result["modes"][mode] = {
            "metrics": item["metrics"],
            "top1": get_top1(item),
            "best_gold": get_gold_best(item),
            "gold_results": get_gold_results(item),
            "top10": [
                simplify_result(
                    r,
                    item["relevant_sources"],
                )
                for r in item["top10"]
            ],
        }

    content_gold = get_gold_best(items["content"])
    best_gold = get_gold_best(items["title_section_content"])

    content_top1 = get_top1(items["content"])
    best_top1 = get_top1(items["title_section_content"])

    result["comparison"] = {
        "content_gold_rank": (
            content_gold["rank"]
            if content_gold
            else None
        ),
        "best_representation_gold_rank": (
            best_gold["rank"]
            if best_gold
            else None
        ),
        "content_top1": content_top1,
        "best_representation_top1": best_top1,
    }

    return result


def print_result(item):
    question = item["question"]

    print()
    print("=" * 100)
    print(f"{item['id']} | {question}")
    print("=" * 100)

    for mode in MODES:
        data = item["modes"][mode]

        metrics = data["metrics"]
        top1 = data["top1"]
        gold = data["best_gold"]

        print()
        print("-" * 100)
        print(f"[{mode}]")

        print(
            f"Gold rank : "
            f"{gold['rank'] if gold else None}"
        )

        print(
            f"Hit@1    : {metrics.get('hit@1')}"
        )

        if top1:
            print()
            print("TOP1")
            print(f"  id         : {top1['id']}")
            print(f"  title      : {top1['title']}")
            print(f"  section    : {top1['section']}")
            print(f"  url        : {top1['url']}")
            print(f"  similarity : {top1['similarity']}")
            print(f"  gold       : {top1['is_gold']}")

        if gold:
            print()
            print("BEST GOLD")
            print(f"  rank       : {gold['rank']}")
            print(f"  id         : {gold['id']}")
            print(f"  title      : {gold['title']}")
            print(f"  section    : {gold['section']}")
            print(f"  similarity : {gold['similarity']}")

    print()
    print("-" * 100)
    print("RANK SUMMARY")
    print("-" * 100)

    for mode in MODES:
        gold = item["modes"][mode]["best_gold"]

        print(
            f"{mode:25s} "
            f"rank={gold['rank'] if gold else None}"
        )

    print()
    print("TOP1 SIMILARITY SUMMARY")
    print("-" * 100)

    for mode in MODES:
        top1 = item["modes"][mode]["top1"]

        if top1:
            print(
                f"{mode:25s} "
                f"{top1['similarity']:.6f} | "
                f"{'GOLD' if top1['is_gold'] else 'NON-GOLD'} | "
                f"{top1['title']}"
            )

    print()
    print("GOLD SIMILARITY SUMMARY")
    print("-" * 100)

    for mode in MODES:
        gold = item["modes"][mode]["best_gold"]

        if gold:
            print(
                f"{mode:25s} "
                f"{gold['similarity']:.6f} | "
                f"{gold['title']}"
            )

    print()
    print("TOP1 vs GOLD GAP")
    print("-" * 100)

    for mode in MODES:
        top1 = item["modes"][mode]["top1"]
        gold = item["modes"][mode]["best_gold"]

        if not top1 or not gold:
            continue

        gap = (
            top1["similarity"]
            - gold["similarity"]
        )

        print(
            f"{mode:25s} "
            f"gap={gap:+.6f}"
        )

    print()
    print("TOP10")
    print("-" * 100)

    for mode in MODES:
        print()
        print(f"[{mode}]")

        for r in item["modes"][mode]["top10"]:
            marker = "GOLD" if r["is_gold"] else "    "

            print(
                f"{r['rank']:2d}. "
                f"{marker} "
                f"{r['similarity']:.6f} | "
                f"{r['title']} | "
                f"{r['url']}"
            )


def main():
    print("Loading:")
    print(INPUT_PATH)

    data = load_data()

    output = {
        "experiment": "E20_regression_extract",
        "target_ids": TARGET_IDS,
        "modes": MODES,
        "questions": [],
    }

    for question_id in TARGET_IDS:
        items = {}

        for mode in MODES:
            item = find_question(
                data,
                mode,
                question_id,
            )

            if item is None:
                raise RuntimeError(
                    f"Cannot find {question_id} "
                    f"in mode={mode}"
                )

            items[mode] = item

        result = compare_modes(items)

        output["questions"].append(result)

        print_result(result)

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
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()