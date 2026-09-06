import json
from pathlib import Path


INPUT = Path("evaluation/error_analysis.json")
OUTPUT = Path("evaluation/miss_review_compact.json")

EXCLUDE_IDS = {"q025", "q041"}


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    misses = []

    for item in results:
        if item["id"] in EXCLUDE_IDS:
            continue

        relevant = item.get("relevant_sources", [])

        # 找真正的 COMPLETE_MISS：
        # 前50名里一个 relevant source 都没有
        retrieved = item.get("retrieved_sources", [])

        if not set(relevant).intersection(retrieved):
            misses.append({
                "id": item["id"],
                "question": item["question"],
                "relevant_sources": relevant,
                "top10_retrieved": [
                    {
                        "rank": x["rank"],
                        "title": x["title"],
                        "url": x["url"],
                        "similarity": x["similarity"],
                    }
                    for x in item.get("retrieved_details", [])[:10]
                ],
            })

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            misses,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Found {len(misses)} complete misses")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()