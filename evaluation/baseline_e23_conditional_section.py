import json
import re
import time
from pathlib import Path

import numpy as np

from embedding.model import BGEEmbeddingModel


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"
CHUNKS_PATH = ROOT_DIR / "data" / "chunks.jsonl"

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = RESULTS_DIR / "e23_conditional_section_summary.json"
DETAILS_PATH = RESULTS_DIR / "e23_conditional_section_details.json"

TOP_K = 10
CANDIDATE_LIMIT = 100
MAX_CHUNKS_PER_SOURCE = 1
BATCH_SIZE = 32


# ============================================================
# Load
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


def normalize_url(url):
    if not url:
        return ""

    return url.rstrip("/") + "/"


def get_relevant_sources(item):
    sources = item.get("relevant_sources", [])

    return {
        normalize_url(source)
        for source in sources
        if source
    }


# ============================================================
# Section quality
# ============================================================

GENERIC_SECTIONS = {
    "recap",
    "requirements",
    "create it",
    "try it",
    "example",
    "installation",
    "getting started",
    "introduction",
    "overview",
    "summary",
    "conclusion",
    "references",
    "contents",
    "table of contents",
}


def clean_section(section):
    if not section:
        return ""

    section = section.strip()

    # Collapse whitespace
    section = re.sub(r"\s+", " ", section)

    return section


def section_quality(section):
    """
    Return:

        True  -> informative section
        False -> generic / low-information section
    """

    section = clean_section(section)

    if not section:
        return False

    lower = section.lower().strip()

    # --------------------------------------------------------
    # 1. Exact generic sections
    # --------------------------------------------------------

    if lower in GENERIC_SECTIONS:
        return False

    # --------------------------------------------------------
    # 2. Very short sections
    # --------------------------------------------------------

    words = re.findall(r"[A-Za-z0-9_]+", lower)

    if len(words) <= 1:
        return False

    # --------------------------------------------------------
    # 3. Generic final path component
    #
    # Example:
    # MCP Python SDK > Example > Create it
    #
    # We don't want the entire path to become useful merely
    # because the prefix contains meaningful words.
    # --------------------------------------------------------

    parts = [
        p.strip()
        for p in re.split(r"\s*>\s*", section)
        if p.strip()
    ]

    if parts:

        last = parts[-1].lower()

        if last in GENERIC_SECTIONS:
            return False

        # Also handle things like:
        # "Recap — ..."
        # "Example — ..."
        # "Requirements — ..."
        for generic in GENERIC_SECTIONS:
            if last.startswith(generic + " "):
                return False

    # --------------------------------------------------------
    # 4. Extremely long navigation paths
    #
    # Very long breadcrumb-only sections are more likely to
    # introduce noise than useful semantic information.
    # --------------------------------------------------------

    if len(section) > 300:
        return False

    return True


# ============================================================
# Representation
# ============================================================

def build_representation(chunk, representation):
    content = (chunk.get("content") or "").strip()
    title = (chunk.get("title") or "").strip()
    section = (chunk.get("section") or "").strip()

    if representation == "content":
        return content

    if representation == "title_content":
        return f"{title}\n\n{content}"

    if representation == "title_section_content":
        return f"{title}\n\n{section}\n\n{content}"

    if representation == "conditional":
        if section_quality(section):
            return f"{title}\n\n{section}\n\n{content}"

        return f"{title}\n\n{content}"

    raise ValueError(
        f"Unknown representation: {representation}"
    )


# ============================================================
# Similarity
# ============================================================

def cosine_similarity(query_embedding, document_embeddings):
    """
    Both query and documents are normalized.

    Returns:
        np.ndarray shape = (N,)
    """

    return np.dot(
        document_embeddings,
        query_embedding,
    )


# ============================================================
# Source diversity
# ============================================================

def diversify(results, limit=TOP_K):
    output = []
    source_counts = {}

    for result in results:

        source = normalize_url(
            result.get("url", "")
        )

        if source_counts.get(source, 0) >= MAX_CHUNKS_PER_SOURCE:
            continue

        output.append(result)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(output) >= limit:
            break

    return output


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(results, relevant_sources):
    retrieved_sources = [
        normalize_url(r.get("url", ""))
        for r in results
    ]

    hits = [
        source in relevant_sources
        for source in retrieved_sources
    ]

    metrics = {}

    for k in [1, 3, 5, 10]:

        top_k = hits[:k]

        metrics[f"hit@{k}"] = (
            1.0 if any(top_k) else 0.0
        )

    gold_count = len(relevant_sources)

    retrieved_gold = len(
        set(retrieved_sources[:10])
        & relevant_sources
    )

    if gold_count:
        metrics["recall@10"] = (
            retrieved_gold / gold_count
        )
    else:
        metrics["recall@10"] = 0.0

    mrr = 0.0

    for rank, hit in enumerate(hits, start=1):

        if hit:
            mrr = 1.0 / rank
            break

    metrics["mrr"] = mrr

    return metrics


def aggregate_metrics(question_results):

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
# Rank / transition analysis
# ============================================================

def get_gold_rank(results, relevant_sources):

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


def compare_rank(rank_a, rank_b):

    if rank_a is None and rank_b is None:
        return "BOTH_MISSING"

    if rank_a is None:
        return "NEW_HIT"

    if rank_b is None:
        return "NEW_MISS"

    if rank_b < rank_a:
        return "IMPROVED"

    if rank_b > rank_a:
        return "WORSENED"

    return "UNCHANGED"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E23 - CONDITIONAL SECTION REPRESENTATION")
    print("=" * 80)

    dataset = load_jsonl(DATASET_PATH)
    chunks = load_jsonl(CHUNKS_PATH)

    # --------------------------------------------------------
    # Remove spec questions
    # --------------------------------------------------------

    dataset = [
        item
        for item in dataset
        if item.get("domain") != "spec"
    ]

    print()
    print(f"Questions : {len(dataset)}")
    print(f"Chunks    : {len(chunks)}")
    print()

    model = BGEEmbeddingModel()

    representations = [
        "title_content",
        "title_section_content",
        "conditional",
    ]

    all_details = {}

    # ========================================================
    # Build representations
    # ========================================================

    embeddings_by_representation = {}

    for representation in representations:

        print("-" * 80)
        print(
            f"Embedding representation: {representation}"
        )

        texts = [
            build_representation(
                chunk,
                representation,
            )
            for chunk in chunks
        ]

        start_time = time.perf_counter()

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
                model.embed_documents(batch)
            )

            embeddings.append(
                batch_embeddings
            )

        embeddings = np.vstack(embeddings)

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Embedding finished in "
            f"{elapsed:.2f}s"
        )

        print(
            f"Embedding shape: "
            f"{embeddings.shape}"
        )

        embeddings_by_representation[
            representation
        ] = embeddings

    # ========================================================
    # Query evaluation
    # ========================================================

    for representation in representations:

        print()
        print("=" * 80)
        print(
            f"EVALUATING: {representation}"
        )
        print("=" * 80)

        doc_embeddings = (
            embeddings_by_representation[
                representation
            ]
        )

        question_results = []

        retrieval_start = time.perf_counter()

        for item in dataset:

            question_id = item["id"]
            question = item["question"]

            relevant_sources = (
                get_relevant_sources(item)
            )

            query_embedding = (
                model.embed_query(question)
            )

            similarities = cosine_similarity(
                query_embedding,
                doc_embeddings,
            )

            # ------------------------------------------------
            # Top100
            # ------------------------------------------------

            top_indices = np.argsort(
                -similarities
            )[:CANDIDATE_LIMIT]

            candidates = []

            for rank, index in enumerate(
                top_indices,
                start=1,
            ):

                chunk = chunks[index]

                candidates.append({
                    "rank": rank,
                    "id": chunk.get("id"),
                    "title": chunk.get("title"),
                    "section": chunk.get("section"),
                    "url": normalize_url(
                        chunk.get("url", "")
                    ),
                    "similarity": float(
                        similarities[index]
                    ),
                })

            # ------------------------------------------------
            # Source diversity
            # ------------------------------------------------

            final_results = diversify(
                candidates,
                limit=TOP_K,
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                final_results,
                relevant_sources,
            )

            gold_rank = get_gold_rank(
                final_results,
                relevant_sources,
            )

            question_results.append({
                "id": question_id,
                "question": question,
                "representation": representation,
                "gold_rank": gold_rank,
                "metrics": metrics,
                "top10": final_results,
            })

        retrieval_elapsed = (
            time.perf_counter()
            - retrieval_start
        )

        aggregate = aggregate_metrics(
            question_results
        )

        print()
        print(
            f"Hit@1      : "
            f"{aggregate['hit@1']:.4f}"
        )

        print(
            f"Hit@3      : "
            f"{aggregate['hit@3']:.4f}"
        )

        print(
            f"Hit@5      : "
            f"{aggregate['hit@5']:.4f}"
        )

        print(
            f"Hit@10     : "
            f"{aggregate['hit@10']:.4f}"
        )

        print(
            f"Recall@10  : "
            f"{aggregate['recall@10']:.4f}"
        )

        print(
            f"MRR        : "
            f"{aggregate['mrr']:.4f}"
        )

        print(
            f"Retrieval time: "
            f"{retrieval_elapsed:.2f}s"
        )

        all_details[representation] = {
            "metrics": aggregate,
            "questions": question_results,
        }

    # ========================================================
    # Compare conditional against title+content
    # ========================================================

    baseline = all_details[
        "title_content"
    ]["questions"]

    conditional = all_details[
        "conditional"
    ]["questions"]

    full_compare = []

    improved = 0
    worsened = 0
    unchanged = 0

    new_hit = 0
    new_miss = 0

    section_used = 0
    section_skipped = 0

    for a, b, item in zip(
        baseline,
        conditional,
        dataset,
    ):

        # ----------------------------------------------------
        # Section quality
        # ----------------------------------------------------

        chunk_sections = [
            chunk.get("section", "")
            for chunk in chunks
        ]

        # Determine whether the conditional top1
        # candidate used section.
        #
        # For detailed analysis, classify all chunks
        # appearing in conditional Top10.
        # ----------------------------------------------------

        conditional_top10 = b["top10"]

        used = 0
        skipped = 0

        for result in conditional_top10:

            chunk_id = result.get("id")

            matching = next(
                (
                    chunk
                    for chunk in chunks
                    if chunk.get("id") == chunk_id
                ),
                None,
            )

            if matching is None:
                continue

            if section_quality(
                matching.get("section", "")
            ):
                used += 1
            else:
                skipped += 1

        section_used += used
        section_skipped += skipped

        rank_a = a["gold_rank"]
        rank_b = b["gold_rank"]

        transition = compare_rank(
            rank_a,
            rank_b,
        )

        if transition == "IMPROVED":
            improved += 1

        elif transition == "WORSENED":
            worsened += 1

        elif transition == "UNCHANGED":
            unchanged += 1

        elif transition == "NEW_HIT":
            new_hit += 1

        elif transition == "NEW_MISS":
            new_miss += 1

        full_compare.append({
            "id": item["id"],
            "question": item["question"],
            "title_content_rank": rank_a,
            "conditional_rank": rank_b,
            "transition": transition,
            "title_content_metrics": a["metrics"],
            "conditional_metrics": b["metrics"],
        })

    # ========================================================
    # Section quality statistics
    # ========================================================

    quality_counts = {}

    for chunk in chunks:

        section = clean_section(
            chunk.get("section", "")
        )

        key = (
            "INFORMATIVE"
            if section_quality(section)
            else "LOW_INFORMATION"
        )

        quality_counts[key] = (
            quality_counts.get(key, 0) + 1
        )

    # ========================================================
    # Final output
    # ========================================================

    summary = {
        "experiment": "E23_conditional_section",
        "dataset": DATASET_PATH.name,
        "question_count": len(dataset),
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "max_chunks_per_source": (
            MAX_CHUNKS_PER_SOURCE
        ),

        "representations": {
            representation: all_details[
                representation
            ]["metrics"]
            for representation in representations
        },

        "conditional_vs_title_content": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
            "new_hit": new_hit,
            "new_miss": new_miss,
        },

        "section_quality_rule": {
            "generic_sections": sorted(
                GENERIC_SECTIONS
            ),
            "max_section_length_chars": 300,
            "minimum_word_count": 2,
        },

        "chunk_section_quality_counts":
            quality_counts,

        "conditional_top10_section_usage": {
            "informative_section_occurrences":
                section_used,
            "low_information_section_occurrences":
                section_skipped,
        },
    }

    details = {
        "summary": summary,
        "representations": all_details,
        "comparison": full_compare,
    }

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
    # Print final comparison
    # ========================================================

    print()
    print("=" * 80)
    print("E23 FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<15}"
        f"{'Title+Content':>18}"
        f"{'Title+Section+Content':>25}"
        f"{'Conditional':>18}"
    )

    print("-" * 80)

    for metric in [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
    ]:

        a = all_details[
            "title_content"
        ]["metrics"][metric]

        b = all_details[
            "title_section_content"
        ]["metrics"][metric]

        c = all_details[
            "conditional"
        ]["metrics"][metric]

        print(
            f"{metric:<15}"
            f"{a:>18.4f}"
            f"{b:>25.4f}"
            f"{c:>18.4f}"
        )

    print()
    print("CONDITIONAL VS TITLE+CONTENT")
    print("-" * 80)

    print(
        f"Improved : {improved}"
    )

    print(
        f"Worsened  : {worsened}"
    )

    print(
        f"Unchanged : {unchanged}"
    )

    print(
        f"New hit   : {new_hit}"
    )

    print(
        f"New miss  : {new_miss}"
    )

    print()
    print(
        f"Summary saved to:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"Details saved to:\n"
        f"{DETAILS_PATH}"
    )


if __name__ == "__main__":
    main()