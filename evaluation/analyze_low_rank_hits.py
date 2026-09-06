import json
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("evaluation/error_analysis.json")
OUTPUT_FILE = Path("evaluation/results/low_rank_analysis.json")


def normalize_url(url):
    if not url:
        return ""
    return url.rstrip("/")


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    # 只分析 LOW_RANK_HIT
    low_rank_hits = [
        item
        for item in results
        if item.get("error_type") == "LOW_RANK_HIT"
    ]

    print("=" * 80)
    print("LOW RANK HIT ANALYSIS")
    print("=" * 80)

    print(f"Total LOW_RANK_HIT: {len(low_rank_hits)}")

    analysis = []
    top1_sources = Counter()

    for item in low_rank_hits:
        question = item["question"]

        relevant_sources = {
            normalize_url(x)
            for x in item.get("relevant_sources", [])
        }

        details = item.get("retrieved_details", [])

        if not details:
            continue

        top1 = details[0]

        top1_source = normalize_url(top1.get("url"))
        top1_similarity = top1.get("similarity")

        # 找到所有 gold source 对应的 chunks
        gold_details = [
            d
            for d in details
            if normalize_url(d.get("url")) in relevant_sources
        ]

        if gold_details:
            best_gold = min(
                gold_details,
                key=lambda x: x["rank"]
            )

            gold_rank = best_gold["rank"]
            gold_similarity = best_gold.get("similarity")
            gold_source = normalize_url(best_gold.get("url"))
        else:
            gold_rank = None
            gold_similarity = None
            gold_source = None

        if (
            top1_similarity is not None
            and gold_similarity is not None
        ):
            similarity_gap = (
                top1_similarity - gold_similarity
            )
        else:
            similarity_gap = None

        result = {
            "id": item["id"],
            "question": question,

            "top1": {
                "rank": 1,
                "source": top1_source,
                "title": top1.get("title"),
                "similarity": top1_similarity,
            },

            "gold": {
                "source": gold_source,
                "rank": gold_rank,
                "similarity": gold_similarity,
            },

            "similarity_gap": similarity_gap,

            "gold_in_top3": (
                gold_rank is not None
                and gold_rank <= 3
            ),

            "gold_in_top5": (
                gold_rank is not None
                and gold_rank <= 5
            ),

            "top1_is_gold": (
                top1_source in relevant_sources
            ),
        }

        analysis.append(result)
        top1_sources[top1_source] += 1

    # --------------------------------------------------
    # 打印核心表格
    # --------------------------------------------------

    print()
    print(
        f"{'ID':<6}"
        f"{'GoldRank':<10}"
        f"{'Top1Sim':<10}"
        f"{'GoldSim':<10}"
        f"{'Gap':<10}"
        f"Top1 Source"
    )

    print("-" * 100)

    for item in analysis:
        top1 = item["top1"]
        gold = item["gold"]

        print(
            f"{item['id']:<6}"
            f"{str(gold['rank']):<10}"
            f"{top1['similarity']:<10.4f}"
            f"{gold['similarity']:<10.4f}"
            f"{item['similarity_gap']:<10.4f}"
            f"{top1['source']}"
        )

    # --------------------------------------------------
    # 统计
    # --------------------------------------------------

    total = len(analysis)

    gold_rank_2_3 = sum(
        1
        for x in analysis
        if x["gold"]["rank"] is not None
        and x["gold"]["rank"] <= 3
    )

    gold_rank_4_5 = sum(
        1
        for x in analysis
        if x["gold"]["rank"] is not None
        and 4 <= x["gold"]["rank"] <= 5
    )

    small_gap_003 = sum(
        1
        for x in analysis
        if x["similarity_gap"] is not None
        and x["similarity_gap"] < 0.03
    )

    small_gap_005 = sum(
        1
        for x in analysis
        if x["similarity_gap"] is not None
        and x["similarity_gap"] < 0.05
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"LOW_RANK_HIT                  : {total}")
    print(f"Gold rank <= 3                : {gold_rank_2_3}")
    print(f"Gold rank 4-5                : {gold_rank_4_5}")
    print(f"Similarity gap < 0.03        : {small_gap_003}")
    print(f"Similarity gap < 0.05        : {small_gap_005}")

    print()
    print("Top1 sources:")
    for source, count in top1_sources.most_common():
        print(f"  {count:>2}  {source}")

    # --------------------------------------------------
    # 保存 JSON
    # --------------------------------------------------

    output = {
        "total_low_rank_hits": total,
        "summary": {
            "gold_rank_le_3": gold_rank_2_3,
            "gold_rank_4_5": gold_rank_4_5,
            "similarity_gap_lt_003": small_gap_003,
            "similarity_gap_lt_005": small_gap_005,
        },
        "top1_sources": dict(top1_sources),
        "questions": analysis,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()