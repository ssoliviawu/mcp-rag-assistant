import json
import re
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = (
    RESULTS_DIR / "e30_final_retrieval_summary.json"
)

DETAILS_PATH = (
    RESULTS_DIR / "e30_final_retrieval_details.json"
)


# ============================================================
# FINAL PIPELINE CONFIGURATION
# ============================================================

TOP_K = 10
CANDIDATE_LIMIT = 100
MAX_CHUNKS_PER_SOURCE = 1

BATCH_SIZE = 32

# ============================================================
# IMPORTANT:
#
# This is EXACTLY E25 C7.
#
# No reranker
# No hybrid retrieval
# No lexical boost
# No migration-specific rule
# No new heuristic
# ============================================================

FINAL_COMBINATION = "C7_all_three"

FINAL_GENERIC_RULES = {
    "recap",
    "installation",
    "requirements",
}


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(path: Path):

    records = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


# ============================================================
# URL normalization
# ============================================================

def normalize_url(url):

    if not url:
        return ""

    return url.rstrip("/") + "/"


def get_relevant_sources(item):

    sources = item.get(
        "relevant_sources",
        [],
    )

    return {
        normalize_url(source)
        for source in sources
        if source
    }


# ============================================================
# EXACT E25 C7 section quality
# ============================================================

def clean_section(section):

    if not section:
        return ""

    section = section.strip()

    section = re.sub(
        r"\s+",
        " ",
        section,
    )

    return section


def section_quality_with_rules(
    section,
    generic_rules,
):
    """
    EXACT E25 section-quality logic.

    C7 generic rules:

        recap
        installation
        requirements

    Other E25 rules remain unchanged.
    """

    section = clean_section(section)

    if not section:
        return False

    lower = section.lower().strip()

    # --------------------------------------------------------
    # 1. Exact generic sections
    # --------------------------------------------------------

    if lower in generic_rules:
        return False

    # --------------------------------------------------------
    # 2. Very short sections
    # --------------------------------------------------------

    words = re.findall(
        r"[A-Za-z0-9_]+",
        lower,
    )

    if len(words) <= 1:
        return False

    # --------------------------------------------------------
    # 3. Generic final breadcrumb component
    # --------------------------------------------------------

    parts = [
        p.strip()
        for p in re.split(
            r"\s*>\s*",
            section,
        )
        if p.strip()
    ]

    if parts:

        last = parts[-1].lower()

        if last in generic_rules:
            return False

        for generic in generic_rules:

            if last.startswith(
                generic + " "
            ):
                return False

    # --------------------------------------------------------
    # 4. Extremely long navigation paths
    # --------------------------------------------------------

    if len(section) > 300:
        return False

    return True


# ============================================================
# EXACT E25 C7 representation
# ============================================================

def build_representation(
    chunk,
    generic_rules,
):

    content = (
        chunk.get("content") or ""
    ).strip()

    title = (
        chunk.get("title") or ""
    ).strip()

    section = (
        chunk.get("section") or ""
    ).strip()

    if section_quality_with_rules(
        section,
        generic_rules,
    ):

        return (
            f"{title}\n\n"
            f"{section}\n\n"
            f"{content}"
        )

    return (
        f"{title}\n\n"
        f"{content}"
    )


# ============================================================
# Cosine similarity
#
# EXACTLY same as E25
# ============================================================

def cosine_similarity(
    query_embedding,
    document_embeddings,
):

    return np.dot(
        document_embeddings,
        query_embedding,
    )


# ============================================================
# Source diversity
#
# EXACTLY same as E25
# ============================================================

def diversify(
    results,
    limit=TOP_K,
):

    output = []
    source_counts = {}

    for result in results:

        source = normalize_url(
            result.get("url", "")
        )

        if (
            source_counts.get(
                source,
                0,
            )
            >= MAX_CHUNKS_PER_SOURCE
        ):
            continue

        output.append(result)

        source_counts[source] = (
            source_counts.get(
                source,
                0,
            )
            + 1
        )

        if len(output) >= limit:
            break

    return output


# ============================================================
# Metrics
#
# EXACTLY same metric definitions as E25
# ============================================================

def calculate_metrics(
    results,
    relevant_sources,
):

    retrieved_sources = [
        normalize_url(
            r.get("url", "")
        )
        for r in results
    ]

    hits = [
        source in relevant_sources
        for source in retrieved_sources
    ]

    metrics = {}

    for k in [
        1,
        3,
        5,
        10,
    ]:

        top_k = hits[:k]

        metrics[
            f"hit@{k}"
        ] = (
            1.0
            if any(top_k)
            else 0.0
        )

    gold_count = len(
        relevant_sources
    )

    retrieved_gold = len(
        set(
            retrieved_sources[:10]
        )
        & relevant_sources
    )

    if gold_count:

        metrics["recall@10"] = (
            retrieved_gold
            / gold_count
        )

    else:

        metrics["recall@10"] = 0.0

    mrr = 0.0

    for rank, hit in enumerate(
        hits,
        start=1,
    ):

        if hit:

            mrr = 1.0 / rank
            break

    metrics["mrr"] = mrr

    return metrics


# ============================================================
# Aggregate metrics
# ============================================================

def aggregate_metrics(
    question_results,
):

    if not question_results:
        return {}

    keys = [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
    ]

    output = {}

    for key in keys:

        values = [
            item["metrics"][key]
            for item in question_results
        ]

        output[key] = float(
            np.mean(values)
        )

    return output


# ============================================================
# Gold rank
# ============================================================

def get_gold_rank(
    results,
    relevant_sources,
):

    for rank, result in enumerate(
        results,
        start=1,
    ):

        source = normalize_url(
            result.get("url", "")
        )

        if source in relevant_sources:
            return rank

    return None


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print("E30-A - FINAL RETRIEVAL BENCHMARK")
    print("=" * 100)

    print()
    print("FINAL PIPELINE")
    print("-" * 100)

    print(f"Combination          : {FINAL_COMBINATION}")
    print(
        f"Generic rules        : "
        f"{sorted(FINAL_GENERIC_RULES)}"
    )
    print(f"Candidate limit      : {CANDIDATE_LIMIT}")
    print(f"Top K                : {TOP_K}")
    print(
        f"Max chunks/source   : "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )

    print()
    print("Components")
    print("  Embedding          : BGEEmbeddingModel")
    print("  Retrieval          : Vector similarity")
    print("  Section rules      : E25 C7")
    print("  Source diversity   : 1 chunk/source")
    print("  Reranker           : NONE")
    print("  Hybrid             : NONE")
    print("  Lexical boost      : NONE")
    print("  Query expansion    : NONE")

    # ========================================================
    # Load dataset
    # ========================================================

    dataset = load_jsonl(
        DATASET_PATH
    )

    chunks = load_jsonl(
        CHUNKS_PATH
    )

    # --------------------------------------------------------
    # Same dataset scope as E25 / E29
    # --------------------------------------------------------

    dataset = [
        item
        for item in dataset
        if item.get("domain") != "spec"
    ]

    print()
    print("-" * 100)

    print(
        f"Questions            : {len(dataset)}"
    )

    print(
        f"Chunks               : {len(chunks)}"
    )

    print()

    # ========================================================
    # Build EXACT C7 representation
    # ========================================================

    print("-" * 100)
    print("BUILDING FINAL C7 REPRESENTATION")
    print("-" * 100)

    texts = [
        build_representation(
            chunk,
            FINAL_GENERIC_RULES,
        )
        for chunk in chunks
    ]

    print(
        f"Representation count : {len(texts)}"
    )

    # ========================================================
    # Embed all chunks
    # ========================================================

    model = BGEEmbeddingModel()

    print()
    print("-" * 100)
    print("EMBEDDING FINAL C7 REPRESENTATION")
    print("-" * 100)

    embedding_start = time.perf_counter()

    embeddings = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        batch_embeddings = (
            model.embed_documents(
                batch
            )
        )

        embeddings.append(
            batch_embeddings
        )

        end = min(
            start + BATCH_SIZE,
            len(texts),
        )

        print(
            f"  Embedded "
            f"{end:>5}/{len(texts)}"
        )

    doc_embeddings = np.vstack(
        embeddings
    )

    embedding_elapsed = (
        time.perf_counter()
        - embedding_start
    )

    print()
    print(
        f"Embedding shape      : "
        f"{doc_embeddings.shape}"
    )

    print(
        f"Embedding time       : "
        f"{embedding_elapsed:.2f}s"
    )

    # ========================================================
    # Retrieval benchmark
    # ========================================================

    print()
    print("=" * 100)
    print("RUNNING FINAL RETRIEVAL BENCHMARK")
    print("=" * 100)

    question_results = []

    retrieval_times = []

    benchmark_start = time.perf_counter()

    for question_index, item in enumerate(
        dataset,
        start=1,
    ):

        question_id = item["id"]
        question = item["question"]

        relevant_sources = (
            get_relevant_sources(
                item
            )
        )

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_start = time.perf_counter()

        query_embedding = (
            model.embed_query(
                question
            )
        )

        # ----------------------------------------------------
        # Vector similarity
        # ----------------------------------------------------

        similarities = (
            cosine_similarity(
                query_embedding,
                doc_embeddings,
            )
        )

        # ----------------------------------------------------
        # Top 100 candidates
        # ----------------------------------------------------

        top_indices = np.argsort(
            -similarities
        )[:CANDIDATE_LIMIT]

        candidates = []

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            chunk = chunks[index]

            candidates.append(
                {
                    "rank": rank,
                    "id": chunk.get("id"),
                    "title": chunk.get("title"),
                    "section": chunk.get("section"),
                    "url": normalize_url(
                        chunk.get(
                            "url",
                            "",
                        )
                    ),
                    "similarity": float(
                        similarities[index]
                    ),
                }
            )

        # ----------------------------------------------------
        # Source diversity
        # ----------------------------------------------------

        final_results = diversify(
            candidates,
            limit=TOP_K,
        )

        query_elapsed = (
            time.perf_counter()
            - query_start
        )

        retrieval_times.append(
            query_elapsed
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            final_results,
            relevant_sources,
        )

        gold_rank = get_gold_rank(
            final_results,
            relevant_sources,
        )

        question_result = {
            "id": question_id,
            "question": question,
            "gold_rank": gold_rank,
            "metrics": metrics,
            "retrieval_latency_ms": (
                query_elapsed * 1000
            ),
            "top10": final_results,
        }

        question_results.append(
            question_result
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        rank_text = (
            str(gold_rank)
            if gold_rank is not None
            else "MISS"
        )

        print(
            f"[{question_index:>2}/"
            f"{len(dataset)}] "
            f"{question_id:<6} "
            f"gold_rank={rank_text:<5} "
            f"latency="
            f"{query_elapsed * 1000:>8.2f} ms"
        )

    benchmark_elapsed = (
        time.perf_counter()
        - benchmark_start
    )

    # ========================================================
    # Aggregate
    # ========================================================

    aggregate = aggregate_metrics(
        question_results
    )

    avg_latency_ms = float(
        np.mean(retrieval_times) * 1000
    )

    p50_latency_ms = float(
        np.percentile(
            retrieval_times,
            50,
        )
        * 1000
    )

    p95_latency_ms = float(
        np.percentile(
            retrieval_times,
            95,
        )
        * 1000
    )

    # ========================================================
    # Rank distribution
    # ========================================================

    rank_distribution = {
        "rank_1": 0,
        "rank_2_3": 0,
        "rank_4_5": 0,
        "rank_6_10": 0,
        "miss": 0,
    }

    for item in question_results:

        rank = item["gold_rank"]

        if rank is None:

            rank_distribution["miss"] += 1

        elif rank == 1:

            rank_distribution["rank_1"] += 1

        elif rank <= 3:

            rank_distribution["rank_2_3"] += 1

        elif rank <= 5:

            rank_distribution["rank_4_5"] += 1

        else:

            rank_distribution["rank_6_10"] += 1

    # ========================================================
    # Residuals
    # ========================================================

    residuals = []

    for item in question_results:

        if item["gold_rank"] is None:

            residuals.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "gold_rank": None,
                }
            )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "experiment": "E30_final_retrieval",

        "status": "FINAL",

        "pipeline": {
            "name": FINAL_COMBINATION,
            "candidate_limit": CANDIDATE_LIMIT,
            "top_k": TOP_K,
            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),
            "generic_rules": sorted(
                FINAL_GENERIC_RULES
            ),
            "embedding_model": (
                "BGEEmbeddingModel"
            ),
            "retrieval": "vector_similarity",
            "reranker": False,
            "hybrid": False,
            "lexical_boost": False,
            "query_expansion": False,
        },

        "dataset": {
            "file": DATASET_PATH.name,
            "total_original_questions": len(
                load_jsonl(DATASET_PATH)
            ),
            "evaluated_questions": len(
                dataset
            ),
            "excluded_spec_questions": (
                len(
                    load_jsonl(
                        DATASET_PATH
                    )
                )
                - len(dataset)
            ),
            "chunks": len(chunks),
        },

        "metrics": aggregate,

        "latency": {
            "embedding_all_chunks_seconds": (
                embedding_elapsed
            ),
            "retrieval_total_seconds": (
                benchmark_elapsed
            ),
            "retrieval_avg_ms": (
                avg_latency_ms
            ),
            "retrieval_p50_ms": (
                p50_latency_ms
            ),
            "retrieval_p95_ms": (
                p95_latency_ms
            ),
        },

        "rank_distribution": rank_distribution,

        "residuals": {
            "count": len(residuals),
            "questions": residuals,
        },

        "interpretation": {
            "retrieval_goal": (
                "Final benchmark of frozen E25 C7 "
                "retrieval pipeline."
            ),
            "optimization_status": (
                "Retrieval pipeline is frozen; "
                "no new retrieval component is "
                "introduced in E30."
            ),
        },
    }

    # ========================================================
    # Details
    # ========================================================

    details = {
        "summary": summary,
        "questions": question_results,
    }

    # ========================================================
    # Save
    # ========================================================

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            details,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 100)
    print("E30 FINAL RETRIEVAL RESULT")
    print("=" * 100)

    print()
    print(f"Pipeline       : {FINAL_COMBINATION}")
    print(f"Questions      : {len(dataset)}")
    print()

    print(
        f"Hit@1          : "
        f"{aggregate['hit@1']:.4f}"
    )

    print(
        f"Hit@3          : "
        f"{aggregate['hit@3']:.4f}"
    )

    print(
        f"Hit@5          : "
        f"{aggregate['hit@5']:.4f}"
    )

    print(
        f"Hit@10         : "
        f"{aggregate['hit@10']:.4f}"
    )

    print(
        f"Recall@10      : "
        f"{aggregate['recall@10']:.4f}"
    )

    print(
        f"MRR            : "
        f"{aggregate['mrr']:.4f}"
    )

    print()
    print("LATENCY")
    print("-" * 100)

    print(
        f"Average        : "
        f"{avg_latency_ms:.2f} ms"
    )

    print(
        f"P50            : "
        f"{p50_latency_ms:.2f} ms"
    )

    print(
        f"P95            : "
        f"{p95_latency_ms:.2f} ms"
    )

    print()
    print("RANK DISTRIBUTION")
    print("-" * 100)

    for key, value in rank_distribution.items():

        print(
            f"{key:<12}: {value}"
        )

    print()
    print(
        f"Residual misses: "
        f"{len(residuals)}"
    )

    print()
    print("=" * 100)
    print("FILES")
    print("=" * 100)

    print(
        f"Summary:\n{SUMMARY_PATH}"
    )

    print(
        f"Details:\n{DETAILS_PATH}"
    )

    print()
    print("=" * 100)
    print("E30-A COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()