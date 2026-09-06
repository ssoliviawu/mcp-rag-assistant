import json
from pathlib import Path


INPUT = Path("evaluation/error_analysis.json")
OUTPUT = Path("evaluation/low_rank_review.json")

# Ground Truth 本身没有 relevant_sources，直接忽略
EXCLUDE_IDS = {"q025", "q041"}


def main():

    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    review = []

    for item in data["results"]:

        qid = item["id"]

        if qid in EXCLUDE_IDS:
            continue

        relevant_sources = set(
            item.get("relevant_sources", [])
        )

        if not relevant_sources:
            continue

        retrieved_details = item.get(
            "retrieved_details",
            []
        )

        # ------------------------------------------------
        # 根据 URL 实际计算 first relevant rank
        # ------------------------------------------------

        first_relevant_rank = None

        for result in retrieved_details:

            rank = result.get("rank")
            url = result.get("url")

            if url in relevant_sources:
                first_relevant_rank = rank
                break

        # Top1 不需要人工审
        if first_relevant_rank == 1:
            continue

        # ------------------------------------------------
        # 保存前 10 个结果
        # ------------------------------------------------

        top10 = []

        for result in retrieved_details[:10]:

            top10.append({
                "rank": result.get("rank"),
                "title": result.get("title"),
                "url": result.get("url"),
                "similarity": result.get("similarity"),
                "is_relevant": (
                    result.get("url")
                    in relevant_sources
                ),
            })

        review.append({
            "id": qid,
            "question": item.get("question"),
            "first_relevant_rank": first_relevant_rank,
            "relevant_sources": list(
                relevant_sources
            ),
            "top10": top10,
        })

    # ----------------------------------------------------
    # 排序
    # ----------------------------------------------------

    review.sort(
        key=lambda x: (
            x["first_relevant_rank"]
            if x["first_relevant_rank"] is not None
            else 999
        )
    )

    output = {
        "total": len(review),
        "results": review,
    }

    with OUTPUT.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ----------------------------------------------------
    # Console summary
    # ----------------------------------------------------

    print("=" * 80)
    print("LOW-RANK RETRIEVAL REVIEW")
    print("=" * 80)

    print(
        f"Total cases: {len(review)}"
    )

    print()

    for item in review:

        rank = item["first_relevant_rank"]

        if rank is None:
            rank_text = "MISS"
        else:
            rank_text = str(rank)

        print(
            f'{item["id"]} | '
            f'rank={rank_text} | '
            f'{item["question"]}'
        )

    print()

    print(
        f"Saved to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()