
from pathlib import Path
import time

from retrieval.search import vector_search
from retrieval.reranker import rerank


DATASET = Path(
    "mcp_rag_eval_dataset.jsonl"
)

CANDIDATE_LIMIT = 30


def load_dataset(path):
    dataset = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            import json

            dataset.append(
                json.loads(line)
            )

    return dataset


def main():

    dataset = load_dataset(
        DATASET
    )

    vector_times = []
    rerank_times = []
    total_times = []

    print(
        f"Questions: {len(dataset)}"
    )

    print()

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        query = item["question"]

        # ====================================================
        # Vector search
        # ====================================================

        start = time.perf_counter()

        candidates = vector_search(
            query,
            limit=CANDIDATE_LIMIT,
        )

        vector_time = (
            time.perf_counter()
            - start
        ) * 1000

        # ====================================================
        # Reranker
        # ====================================================

        start = time.perf_counter()

        reranked = rerank(
            query,
            candidates,
        )

        rerank_time = (
            time.perf_counter()
            - start
        ) * 1000

        # ====================================================
        # Total
        # ====================================================

        total_time = (
            vector_time
            + rerank_time
        )

        vector_times.append(
            vector_time
        )

        rerank_times.append(
            rerank_time
        )

        total_times.append(
            total_time
        )

        print(
            f"\r"
            f"{index:2d}/{len(dataset)} "
            f"vector="
            f"{vector_time:8.2f} ms "
            f"rerank="
            f"{rerank_time:8.2f} ms "
            f"total="
            f"{total_time:8.2f} ms",
            end="",
            flush=True,
        )

    print()
    print()

    def avg(values):
        return sum(values) / len(values)

    print("=" * 70)
    print("RERANKER BENCHMARK")
    print("=" * 70)

    print(
        f"Vector search avg : "
        f"{avg(vector_times):.2f} ms"
    )

    print(
        f"Reranker avg      : "
        f"{avg(rerank_times):.2f} ms"
    )

    print(
        f"Total avg         : "
        f"{avg(total_times):.2f} ms"
    )

    print()

    print(
        f"Vector median     : "
        f"{sorted(vector_times)[len(vector_times)//2]:.2f} ms"
    )

    print(
        f"Reranker median   : "
        f"{sorted(rerank_times)[len(rerank_times)//2]:.2f} ms"
    )

    print(
        f"Total median      : "
        f"{sorted(total_times)[len(total_times)//2]:.2f} ms"
    )


if __name__ == "__main__":
    main()

