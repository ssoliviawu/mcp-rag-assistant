import json
import re
import time
from pathlib import Path

import numpy as np

from embedding.model import BGEEmbeddingModel
from db.repository import search_chunks


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

SUMMARY_PATH = RESULTS_DIR / "e18_representation_summary.json"
DETAILS_PATH = RESULTS_DIR / "e18_representation_details.json"


# ============================================================
# Parameters
# ============================================================

CANDIDATE_LIMIT = 100
TOP_K = 10

# 48 valid RAG questions:
# exclude domain == "spec"
EXCLUDE_DOMAINS = {"spec"}


# ============================================================
# Dataset
# ============================================================

def load_dataset(path: Path):
    questions = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if item.get("domain") in EXCLUDE_DOMAINS:
                continue

            questions.append(item)

    return questions


# ============================================================
# Helpers
# ============================================================

def normalize_url(url):
    if not url:
        return ""

    return url.rstrip("/")


def get_relevant_sources(item):
    return {
        normalize_url(url)
        for url in item.get("relevant_sources", [])
        if url
    }


def is_relevant(result, relevant_sources):
    return (
        normalize_url(result.get("url"))
        in relevant_sources
    )


def calculate_metrics(results, relevant_sources):
    ranks = []

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, relevant_sources):
            ranks.append(rank)

    hit_at = {}

    for k in [1, 3, 5, 10]:
        hit_at[k] = (
            1.0
            if any(rank <= k for rank in ranks)
            else 0.0
        )

    if ranks:
        mrr = 1.0 / ranks[0]
    else:
        mrr = 0.0

    # Recall against number of relevant sources.
    retrieved_sources = {
        normalize_url(r.get("url"))
        for r in results[:TOP_K]
        if r.get("url")
    }

    matched_sources = (
        retrieved_sources & relevant_sources
    )

    recall_at_10 = (
        len(matched_sources) / len(relevant_sources)
        if relevant_sources
        else 0.0
    )

    return {
        "hit@1": hit_at[1],
        "hit@3": hit_at[3],
        "hit@5": hit_at[5],
        "hit@10": hit_at[10],
        "recall@10": recall_at_10,
        "mrr": mrr,
        "first_relevant_rank": (
            ranks[0] if ranks else None
        ),
    }


# ============================================================
# Representation builders
# ============================================================

def build_representation(chunk, mode):
    content = (chunk.get("content") or "").strip()
    title = (chunk.get("title") or "").strip()
    section = (chunk.get("section") or "").strip()

    if mode == "content":
        return content

    if mode == "title_content":
        parts = []

        if title:
            parts.append(title)

        if content:
            parts.append(content)

        return "\n\n".join(parts)

    if mode == "section_content":
        parts = []

        if section:
            parts.append(section)

        if content:
            parts.append(content)

        return "\n\n".join(parts)

    if mode == "title_section_content":
        parts = []

        if title:
            parts.append(title)

        if section:
            parts.append(section)

        if content:
            parts.append(content)

        return "\n\n".join(parts)

    raise ValueError(
        f"Unknown representation mode: {mode}"
    )


# ============================================================
# Source diversity
# ============================================================

def diversify(
    results,
    limit=TOP_K,
    max_chunks_per_source=1,
):
    output = []
    source_counts = {}

    for result in results:

        source = normalize_url(
            result.get("url")
        )

        if source_counts.get(source, 0) >= max_chunks_per_source:
            continue

        output.append(result)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(output) >= limit:
            break

    return output


# ============================================================
# Metrics aggregation
# ============================================================

def aggregate_metrics(question_results):

    n = len(question_results)

    if n == 0:
        return {}

    metrics = [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
    ]

    output = {
        "questions": n,
    }

    for metric in metrics:
        output[metric] = (
            sum(
                item["metrics"][metric]
                for item in question_results
            )
            / n
        )

    return output


# ============================================================
# Main experiment
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("E18 - EMBEDDING REPRESENTATION A/B TEST")
    print("=" * 80)

    print()

    print(
        f"Dataset : {DATASET_PATH}"
    )

    dataset = load_dataset(
        DATASET_PATH
    )

    print(
        f"Questions : {len(dataset)}"
    )

    print(
        f"Candidate : {CANDIDATE_LIMIT}"
    )

    print(
        f"Top K     : {TOP_K}"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading embedding model...")

    model = BGEEmbeddingModel()

    print("Embedding model loaded.")

    print()

    # --------------------------------------------------------
    # Representation modes
    # --------------------------------------------------------

    modes = [
        "content",
        "title_content",
        "section_content",
        "title_section_content",
    ]

    all_details = {}

    all_summaries = {}

    # --------------------------------------------------------
    # Run each representation
    # --------------------------------------------------------

    for mode in modes:

        print("=" * 80)
        print(f"REPRESENTATION: {mode}")
        print("=" * 80)

        question_results = []

        total_latency_ms = 0.0

        for index, item in enumerate(
            dataset,
            start=1,
        ):

            question_id = item["id"]
            question = item["question"]

            relevant_sources = (
                get_relevant_sources(item)
            )

            start_time = time.perf_counter()

            # ------------------------------------------------
            # IMPORTANT:
            #
            # We intentionally use the SAME query embedding
            # for all representation modes.
            #
            # Only document representation changes.
            # ------------------------------------------------

            query_embedding = model.embed_query(
                question
            )

            candidates = search_chunks(
                query_embedding,
                limit=CANDIDATE_LIMIT,
            )

            # ------------------------------------------------
            # Re-embed the retrieved chunks using the
            # current representation.
            #
            # This is NOT a production retrieval pipeline.
            # It is an offline representation experiment.
            # ------------------------------------------------

            texts = [
                build_representation(
                    chunk,
                    mode,
                )
                for chunk in candidates
            ]

            document_embeddings = (
                model.embed_documents(
                    texts
                )
            )

            query_vector = np.asarray(
                query_embedding,
                dtype=np.float32,
            )

            # ------------------------------------------------
            # Cosine similarity
            #
            # Embeddings are already normalized by BGE model.
            # Therefore dot product == cosine similarity.
            # ------------------------------------------------

            scores = (
                document_embeddings
                @ query_vector
            )

            ranked_indices = np.argsort(
                -scores
            )

            reranked_candidates = []

            for new_rank, idx in enumerate(
                ranked_indices,
                start=1,
            ):

                original = candidates[
                    int(idx)
                ]

                result = dict(original)

                result["representation"] = mode

                result["representation_similarity"] = (
                    float(scores[int(idx)])
                )

                result["representation_rank"] = (
                    new_rank
                )

                reranked_candidates.append(
                    result
                )

            # ------------------------------------------------
            # Source diversity
            # ------------------------------------------------

            final_results = diversify(
                reranked_candidates,
                limit=TOP_K,
                max_chunks_per_source=1,
            )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            total_latency_ms += elapsed_ms

            metrics = calculate_metrics(
                final_results,
                relevant_sources,
            )

            gold_ranks = []

            for rank, result in enumerate(
                final_results,
                start=1,
            ):
                if is_relevant(
                    result,
                    relevant_sources,
                ):
                    gold_ranks.append(rank)

            question_results.append(
                {
                    "id": question_id,
                    "question": question,
                    "representation": mode,
                    "relevant_sources": sorted(
                        relevant_sources
                    ),
                    "metrics": metrics,
                    "gold_ranks": gold_ranks,
                    "top10": [
                        {
                            "rank": rank,
                            "id": result.get("id"),
                            "title": result.get("title"),
                            "section": result.get("section"),
                            "url": result.get("url"),
                            "similarity": result.get(
                                "representation_similarity"
                            ),
                            "original_vector_similarity": result.get(
                                "similarity"
                            ),
                        }
                        for rank, result in enumerate(
                            final_results,
                            start=1,
                        )
                    ],
                    "latency_ms": elapsed_ms,
                }
            )

            if index % 10 == 0 or index == len(dataset):
                print(
                    f"Processed {index}/{len(dataset)}"
                )

        summary = aggregate_metrics(
            question_results
        )

        summary["avg_latency_ms"] = (
            total_latency_ms / len(dataset)
        )

        all_details[mode] = question_results
        all_summaries[mode] = summary

        print()
        print(
            f"{mode}:"
        )
        print(
            f"  Hit@1     : {summary['hit@1']:.4f}"
        )
        print(
            f"  Hit@3     : {summary['hit@3']:.4f}"
        )
        print(
            f"  Hit@5     : {summary['hit@5']:.4f}"
        )
        print(
            f"  Hit@10    : {summary['hit@10']:.4f}"
        )
        print(
            f"  Recall@10 : {summary['recall@10']:.4f}"
        )
        print(
            f"  MRR       : {summary['mrr']:.4f}"
        )
        print(
            f"  Avg ms    : {summary['avg_latency_ms']:.2f}"
        )

        print()

    # ========================================================
    # Compare against CONTENT baseline
    # ========================================================

    baseline = all_details["content"]

    comparisons = {}

    for mode in modes:

        if mode == "content":
            continue

        current = all_details[mode]

        improved = 0
        worsened = 0
        unchanged = 0

        rank_changes = []

        for base_item, current_item in zip(
            baseline,
            current,
        ):

            base_rank = (
                base_item["metrics"][
                    "first_relevant_rank"
                ]
            )

            current_rank = (
                current_item["metrics"][
                    "first_relevant_rank"
                ]
            )

            if (
                base_rank is None
                and current_rank is None
            ):
                unchanged += 1

            elif base_rank is None:
                improved += 1

            elif current_rank is None:
                worsened += 1

            elif current_rank < base_rank:
                improved += 1

            elif current_rank > base_rank:
                worsened += 1

            else:
                unchanged += 1

            rank_changes.append(
                {
                    "id": base_item["id"],
                    "question": base_item["question"],
                    "baseline_rank": base_rank,
                    "new_rank": current_rank,
                    "delta": (
                        None
                        if (
                            base_rank is None
                            or current_rank is None
                        )
                        else base_rank - current_rank
                    ),
                }
            )

        comparisons[mode] = {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
            "rank_changes": rank_changes,
        }

    # ========================================================
    # Print comparison table
    # ========================================================

    print("=" * 80)
    print("E18 SUMMARY")
    print("=" * 80)

    print()

    print(
        f"{'Representation':<25}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MRR':>10}"
    )

    print("-" * 87)

    for mode in modes:

        s = all_summaries[mode]

        print(
            f"{mode:<25}"
            f"{s['hit@1']:>10.4f}"
            f"{s['hit@3']:>10.4f}"
            f"{s['hit@5']:>10.4f}"
            f"{s['hit@10']:>10.4f}"
            f"{s['recall@10']:>12.4f}"
            f"{s['mrr']:>10.4f}"
        )

    print()

    print("=" * 80)
    print("RANK CHANGE VS CONTENT")
    print("=" * 80)

    print()

    for mode in modes:

        if mode == "content":
            continue

        c = comparisons[mode]

        print(
            f"{mode:<25}"
            f"Improve={c['improved']:>3}  "
            f"Worsen={c['worsened']:>3}  "
            f"Same={c['unchanged']:>3}"
        )

    # ========================================================
    # Save summary
    # ========================================================

    summary_data = {
        "experiment": "E18_embedding_representation",
        "dataset": str(DATASET_PATH),
        "question_count": len(dataset),
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "source_diversity": {
            "enabled": True,
            "max_chunks_per_source": 1,
        },
        "representations": {
            "content": "content",
            "title_content": "title + content",
            "section_content": "section + content",
            "title_section_content": (
                "title + section + content"
            ),
        },
        "summaries": all_summaries,
        "comparisons_vs_content": comparisons,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    details_data = {
        "experiment": "E18_embedding_representation",
        "questions": all_details,
    }

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            details_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Summary saved to:"
    )
    print(
        SUMMARY_PATH
    )

    print()
    print(
        f"Details saved to:"
    )
    print(
        DETAILS_PATH
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()