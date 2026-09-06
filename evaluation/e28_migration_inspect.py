import json
from pathlib import Path


INPUT = Path("evaluation/results/e28_migration_analysis.json")


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data["migration_top1_cases"]

    print("=" * 100)
    print("E28-A2 - MIGRATION CONTENT INSPECTION")
    print("=" * 100)

    for case in cases:
        migration = case["migration_top1"]
        gold = case["gold"]

        print("\n" + "=" * 100)
        print(
            f"{case['id']} | "
            f"gold_rank={case['gold_rank']} | "
            f"margin={case['margin']:.6f}"
        )
        print(f"QUESTION: {case['question']}")

        print("\n" + "-" * 100)
        print("MIGRATION COMPETITOR")
        print("-" * 100)
        print(f"Title      : {migration['title']}")
        print(f"Section    : {migration['section']}")
        print(f"Similarity : {migration['similarity']}")
        print(f"Source     : {migration['source']}")

        print("\nCONTENT:")
        print(migration["content"])

        print("\n" + "-" * 100)
        print("GOLD")
        print("-" * 100)
        print(f"Title      : {gold['title']}")
        print(f"Section    : {gold['section']}")
        print(f"Similarity : {gold['similarity']}")
        print(f"Source     : {gold['source']}")

        print("\nCONTENT:")
        print(gold["content"])


if __name__ == "__main__":
    main()