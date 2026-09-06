import json
from pathlib import Path


PATH = (
    Path(__file__).resolve().parent
    / "reranker_diagnostic.json"
)


def main():

    with PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    results = data["results"]

    broken = [
        r
        for r in results
        if r["category"]
        in [
            "A_TOP1_BROKEN",
            "C_TOP10_BROKEN",
        ]
    ]

    print()
    print("=" * 100)
    print("RERANKER BROKEN CASES")
    print("=" * 100)

    print(
        f"Total broken cases: {len(broken)}"
    )

    for r in broken:

        print()
        print("=" * 100)

        print(
            f"ID: {r['id']}"
        )

        print(
            f"Category: {r['category']}"
        )

        print(
            f"Question: {r['question']}"
        )

        print()
        print("Relevant source analysis:")
        print("-" * 100)

        for item in r[
            "relevant_analysis"
        ]:

            print(
                f"Source: {item['source']}"
            )

            print(
                f"  Vector rank:     "
                f"{item['vector_rank']}"
            )

            print(
                f"  Reranker rank:   "
                f"{item['reranker_rank']}"
            )

            print(
                f"  Rank change:     "
                f"{item['rank_change']}"
            )

            print(
                f"  Vector similarity:"
                f" {item['vector_similarity']}"
            )

            print(
                f"  Reranker score:  "
                f"{item['reranker_score']}"
            )

            print()

        print("Vector Top 10:")
        print("-" * 100)

        for item in r[
            "vector_top10"
        ]:

            print(
                f"{item['rank']:>2}. "
                f"sim={item['similarity']} | "
                f"{item['title']} | "
                f"{item['url']}"
            )

        print()

        print("Reranker Top 10:")
        print("-" * 100)

        for item in r[
            "reranker_top10"
        ]:

            print(
                f"{item['rank']:>2}. "
                f"rr={item['reranker_score']} | "
                f"sim={item['similarity']} | "
                f"{item['title']} | "
                f"{item['url']}"
            )


if __name__ == "__main__":
    main()