import json

from retrieval.search import vector_search
from retrieval.reranker import rerank


DATASET = "mcp_rag_eval_dataset.jsonl"


def normalize_source(url: str) -> str:
    return url.rstrip("/")


def load_dataset():
    items = []

    with open(
        DATASET,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            items.append(
                json.loads(line)
            )

    return items


def main():

    dataset = load_dataset()

    candidate_limits = [
        10,
        20,
        30,
        50,
        100,
        200,
    ]

    all_results = {}

    for candidate_limit in candidate_limits:

        print()
        print("=" * 100)
        print(
            f"Candidate Top {candidate_limit}"
        )
        print("=" * 100)

        ranks = []

        for i, item in enumerate(
            dataset,
            start=1,
        ):

            question = item["question"]

            relevant_sources = {
                normalize_source(x)
                for x in item["relevant_sources"]
            }

            vector_results = vector_search(
                question,
                limit=candidate_limit,
            )

            reranked = rerank(
                question,
                vector_results,
            )

            best_rank = None

            for rank, result in enumerate(
                reranked,
                start=1,
            ):

                url = normalize_source(
                    result["url"]
                )

                if url in relevant_sources:

                    best_rank = rank
                    break

            if best_rank is None:

                print(
                    f"{i:03d} | NO MATCH | "
                    f"{question}"
                )

            else:

                ranks.append(best_rank)

                top = reranked[
                    best_rank - 1
                ]

                print(
                    f"{i:03d} | "
                    f"rank={best_rank:4d} | "
                    f"score={top['reranker_score']:.6f} | "
                    f"{top['id']} | "
                    f"{question}"
                )

        total = len(dataset)

        metrics = {}

        for k in [
            1,
            3,
            5,
            10,
            20,
        ]:

            hit = sum(
                rank <= k
                for rank in ranks
            )

            metrics[k] = (
                hit / total
                if total
                else 0.0
            )

        all_results[
            candidate_limit
        ] = metrics

    print()
    print("=" * 100)
    print(
        "ONNX RERANKER CANDIDATE SIZE COMPARISON"
    )
    print("=" * 100)

    print(
        f"{'Candidate':>10} "
        f"{'R@1':>10} "
        f"{'R@3':>10} "
        f"{'R@5':>10} "
        f"{'R@10':>10} "
        f"{'R@20':>10}"
    )

    print("-" * 100)

    for candidate_limit in candidate_limits:

        m = all_results[
            candidate_limit
        ]

        print(
            f"{candidate_limit:>10} "
            f"{m[1]:>10.4f} "
            f"{m[3]:>10.4f} "
            f"{m[5]:>10.4f} "
            f"{m[10]:>10.4f} "
            f"{m[20]:>10.4f}"
        )


if __name__ == "__main__":
    main()