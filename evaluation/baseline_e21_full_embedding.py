
import json
import time
from pathlib import Path

import numpy as np

from embedding.model import BGEEmbeddingModel


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"
CHUNKS_PATH = ROOT_DIR / "data" / "chunks.jsonl"

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

SUMMARY_PATH = RESULTS_DIR / "e21_full_embedding_summary.json"
DETAILS_PATH = RESULTS_DIR / "e21_full_embedding_details.json"


# ============================================================
# Parameters
# ============================================================

CANDIDATE_LIMIT = 100
TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 1

BATCH_SIZE = 32


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def normalize_url(url: str) -> str:
    if not url:
        return ""

    return url.rstrip("/") + "/"


def is_valid_question(item: dict) -> bool:
    # q025 / q041 are spec questions and are excluded
    # from the RAG retrieval evaluation.
    return item.get("domain") != "spec"


def build_representation(
    chunk: dict,
    representation: str,
) -> str:

    content = (chunk.get("content") or "").strip()
    title = (chunk.get("title") or "").strip()
    section = (chunk.get("section") or "").strip()

    if representation == "content":
        return content

    if representation == "title_content":
        return f"{title}\n\n{content}"

    if representation == "title_section_content":
        return f"{title}\n\n{section}\n\n{content}"

    raise ValueError(
        f"Unknown representation: {representation}"
    )


def cosine_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    document_embeddings = np.asarray(
        document_embeddings,
        dtype=np.float32,
    )

    # All embeddings from BGEEmbeddingModel are normalized.
    # Normalize again defensively.
    query_norm = np.linalg.norm(query_embedding)

    if query_norm > 0:
        query_embedding = (
            query_embedding / query_norm
        )

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1,
        keepdims=True,
    )

    document_embeddings = (
        document_embeddings
        / np.clip(
            document_norms,
            1e-12,
            None,
        )
    )

    return document_embeddings @ query_embedding


def source_diversity(
    ranked_results: list[dict],
    top_k: int,
    max_chunks_per_source: int,
) -> list[dict]:

    selected = []
    source_counts = {}

    for result in ranked_results:

        source = normalize_url(
            result.get("url", "")
        )

        # Fallback for safety.
        if not source:
            source = result.get("id", "")

        count = source_counts.get(source, 0)

        if count >= max_chunks_per_source:
            continue

        selected.append(result)
        source_counts[source] = count + 1

        if len(selected) >= top_k:
            break

    return selected


def source_is_gold(
    url: str,
    relevant_sources: list[str],
) -> bool:

    normalized_url = normalize_url(url)

    normalized_gold = {
        normalize_url(source)
        for source in relevant_sources
    }

    return normalized_url in normalized_gold


def find_first_gold_rank(
    results: list[dict],
    relevant_sources: list[str],
) -> int | None:

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if source_is_gold(
            result.get("url", ""),
            relevant_sources,
        ):
            return rank

    return None


def calculate_metrics(
    results: list[dict],
    relevant_sources: list[str],
) -> dict:

    gold_ranks = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if source_is_gold(
            result.get("url", ""),
            relevant_sources,
        ):
            gold_ranks.append(rank)

    first_rank = (
        min(gold_ranks)
        if gold_ranks
        else None
    )

    hit_at_1 = (
        1.0
        if any(rank <= 1 for rank in gold_ranks)
        else 0.0
    )

    hit_at_3 = (
        1.0
        if any(rank <= 3 for rank in gold_ranks)
        else 0.0
    )

    hit_at_5 = (
        1.0
        if any(rank <= 5 for rank in gold_ranks)
        else 0.0
    )

    hit_at_10 = (
        1.0
        if any(rank <= 10 for rank in gold_ranks)
        else 0.0
    )

    # Recall@K here is source-level recall:
    # number of relevant sources retrieved / number of
    # relevant sources in the dataset.
    #
    # Because source diversity can keep only one chunk
    # per source, this is intentionally source-level.
    def recall_at_k(k: int) -> float:
        retrieved_sources = {
            normalize_url(result.get("url", ""))
            for result in results[:k]
        }

        gold_sources = {
            normalize_url(source)
            for source in relevant_sources
        }

        if not gold_sources:
            return 0.0

        return (
            len(
                retrieved_sources
                & gold_sources
            )
            / len(gold_sources)
        )

    mrr = (
        1.0 / first_rank
        if first_rank is not None
        else 0.0
    )

    return {
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "hit@5": hit_at_5,
        "hit@10": hit_at_10,
        "recall@10": recall_at_k(10),
        "mrr": mrr,
        "first_relevant_rank": first_rank,
        "gold_ranks": gold_ranks,
    }


def aggregate_metrics(
    question_results: list[dict],
) -> dict:

    if not question_results:
        return {}

    metric_names = [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
    ]

    metrics = {}

    for name in metric_names:
        values = [
            item["metrics"][name]
            for item in question_results
        ]

        metrics[name] = float(
            np.mean(values)
        )

    metrics["questions"] = len(
        question_results
    )

    metrics["hits@1"] = int(
        sum(
            item["metrics"]["hit@1"]
            for item in question_results
        )
    )

    metrics["hits@3"] = int(
        sum(
            item["metrics"]["hit@3"]
            for item in question_results
        )
    )

    metrics["hits@5"] = int(
        sum(
            item["metrics"]["hit@5"]
            for item in question_results
        )
    )

    metrics["hits@10"] = int(
        sum(
            item["metrics"]["hit@10"]
            for item in question_results
        )
    )

    return metrics


# ============================================================
# Full embedding
# ============================================================

def embed_chunks(
    chunks: list[dict],
    model: BGEEmbeddingModel,
    representation: str,
) -> np.ndarray:

    embeddings = []

    total = len(chunks)

    print()
    print(
        f"Embedding representation: "
        f"{representation}"
    )
    print(
        f"Total chunks: {total}"
    )

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = chunks[start:end]

        texts = [
            build_representation(
                chunk,
                representation,
            )
            for chunk in batch
        ]

        batch_embeddings = (
            model.embed_documents(texts)
        )

        embeddings.append(
            batch_embeddings
        )

        print(
            f"  embedded "
            f"{end}/{total}"
        )

    return np.vstack(embeddings)


# ============================================================
# Retrieval
# ============================================================

def retrieve_for_question(
    question: dict,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    model: BGEEmbeddingModel,
) -> dict:

    qid = question["id"]
    query = question["question"]

    relevant_sources = (
        question["relevant_sources"]
    )

    start_time = time.perf_counter()

    query_embedding = model.embed_query(
        query
    )

    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )

    # Full ranking.
    ranked_indices = np.argsort(
        -similarities
    )

    # Keep Top100 before source diversity.
    top_indices = ranked_indices[
        :CANDIDATE_LIMIT
    ]

    candidate_results = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        chunk = chunks[int(index)]

        candidate_results.append(
            {
                "rank": rank,
                "id": chunk.get("id"),
                "title": chunk.get("title"),
                "section": chunk.get("section"),
                "content": chunk.get("content", ""),
                "url": chunk.get("url"),
                "similarity": float(
                    similarities[int(index)]
                ),
                "is_gold": source_is_gold(
                    chunk.get("url", ""),
                    relevant_sources,
                ),
            }
        )

    # Apply the exact same source diversity
    # policy used by the current vector pipeline.
    final_results = source_diversity(
        candidate_results,
        top_k=TOP_K,
        max_chunks_per_source=(
            MAX_CHUNKS_PER_SOURCE
        ),
    )

    # Re-number final ranks after diversity.
    for rank, result in enumerate(
        final_results,
        start=1,
    ):
        result["rank"] = rank

    metrics = calculate_metrics(
        final_results,
        relevant_sources,
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    return {
        "id": qid,
        "question": query,
        "relevant_sources": relevant_sources,
        "metrics": metrics,
        "latency_ms": latency_ms,
        "candidate_top100": candidate_results,
        "top10": final_results,
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("E21 - FULL DATABASE REPRESENTATION EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_jsonl(
        DATASET_PATH
    )

    dataset = [
        item
        for item in dataset
        if is_valid_question(item)
    ]

    print()
    print(
        f"Questions : {len(dataset)}"
    )

    # --------------------------------------------------------
    # Load all chunks
    # --------------------------------------------------------

    chunks = load_jsonl(
        CHUNKS_PATH
    )

    print(
        f"Chunks    : {len(chunks)}"
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    model = BGEEmbeddingModel()

    representations = [
        "title_content",
        "title_section_content",
    ]

    all_results = {}
    summary = {}

    # --------------------------------------------------------
    # Run both representations
    # --------------------------------------------------------

    for representation in representations:

        print()
        print("=" * 80)
        print(
            f"REPRESENTATION: {representation}"
        )
        print("=" * 80)

        # --------------------------------------------
        # Full database embedding
        # --------------------------------------------

        embedding_start = time.perf_counter()

        chunk_embeddings = embed_chunks(
            chunks,
            model,
            representation,
        )

        embedding_latency = (
            time.perf_counter()
            - embedding_start
        )

        print()
        print(
            f"Embedding finished in "
            f"{embedding_latency:.2f}s"
        )

        print(
            f"Embedding shape: "
            f"{chunk_embeddings.shape}"
        )

        # --------------------------------------------
        # Retrieval evaluation
        # --------------------------------------------

        question_results = []

        retrieval_start = (
            time.perf_counter()
        )

        for index, question in enumerate(
            dataset,
            start=1,
        ):

            print(
                f"[{index}/{len(dataset)}] "
                f"{question['id']}"
            )

            result = retrieve_for_question(
                question,
                chunks,
                chunk_embeddings,
                model,
            )

            question_results.append(
                result
            )

            print(
                f"  first relevant rank: "
                f"{result['metrics']['first_relevant_rank']}"
            )

            print(
                f"  hit@1: "
                f"{result['metrics']['hit@1']:.0f}"
            )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        aggregate = aggregate_metrics(
            question_results
        )

        aggregate[
            "embedding_seconds"
        ] = embedding_latency

        aggregate[
            "retrieval_seconds"
        ] = retrieval_latency

        aggregate[
            "avg_retrieval_latency_ms"
        ] = float(
            np.mean(
                [
                    item["latency_ms"]
                    for item in question_results
                ]
            )
        )

        all_results[
            representation
        ] = question_results

        summary[
            representation
        ] = aggregate

        print()
        print(
            f"{representation}"
        )
        print(
            f"  Hit@1      : "
            f"{aggregate['hit@1']:.4f}"
        )
        print(
            f"  Hit@3      : "
            f"{aggregate['hit@3']:.4f}"
        )
        print(
            f"  Hit@5      : "
            f"{aggregate['hit@5']:.4f}"
        )
        print(
            f"  Hit@10     : "
            f"{aggregate['hit@10']:.4f}"
        )
        print(
            f"  Recall@10  : "
            f"{aggregate['recall@10']:.4f}"
        )
        print(
            f"  MRR        : "
            f"{aggregate['mrr']:.4f}"
        )
        print(
            f"  Avg latency: "
            f"{aggregate['avg_retrieval_latency_ms']:.2f} ms"
        )

    # ========================================================
    # Compare
    # ========================================================

    a = summary["title_content"]
    b = summary["title_section_content"]

    comparison = {}

    for metric in [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
    ]:
        comparison[metric] = {
            "title_content": a[metric],
            "title_section_content": b[metric],
            "delta": (
                b[metric]
                - a[metric]
            ),
        }

    # ========================================================
    # Save summary
    # ========================================================

    summary_output = {
        "experiment": (
            "E21_full_database_representation"
        ),
        "dataset": str(
            DATASET_PATH
        ),
        "chunks": len(chunks),
        "question_count": len(dataset),
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "max_chunks_per_source": (
            MAX_CHUNKS_PER_SOURCE
        ),
        "representations": [
            "title_content",
            "title_section_content",
        ],
        "summary": summary,
        "comparison": comparison,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Save details
    # ========================================================

    details_output = {
        "experiment": (
            "E21_full_database_representation"
        ),
        "question_count": len(dataset),
        "chunk_count": len(chunks),
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "max_chunks_per_source": (
            MAX_CHUNKS_PER_SOURCE
        ),
        "questions": all_results,
    }

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            details_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 80)
    print("E21 FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<15}"
        f"{'title+content':>18}"
        f"{'title+section+content':>25}"
        f"{'Delta':>12}"
    )

    print("-" * 70)

    for metric in [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
    ]:

        item = comparison[metric]

        print(
            f"{metric:<15}"
            f"{item['title_content']:>18.4f}"
            f"{item['title_section_content']:>25.4f}"
            f"{item['delta']:>12.4f}"
        )

    print()
    print(
        "Summary saved to:"
    )
    print(
        SUMMARY_PATH
    )

    print()
    print(
        "Details saved to:"
    )
    print(
        DETAILS_PATH
    )


if __name__ == "__main__":
    main()

