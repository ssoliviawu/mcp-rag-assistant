from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from embedding.model import BGEEmbeddingModel
from retrieval.search import search_chunks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = PROJECT_ROOT / "mcp_rag_eval_dataset.jsonl"
E27_INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "e27_input.json"
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CANDIDATE_LIMIT = 100
TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 1

MIGRATION_SOURCE = "https://py.sdk.modelcontextprotocol.io/migration/"

GENERIC_SECTIONS = {
    "recap",
    "installation",
    "requirements",
}

# ---------------------------------------------------------------------------
# Section representation
# ---------------------------------------------------------------------------


def clean_section(section: str | None) -> str:
    if not section:
        return ""

    return " ".join(str(section).split())


def section_quality(section: str | None) -> bool:
    """
    Exact C7 section-quality logic.

    The only generic rules used by C7 are:
        - recap
        - installation
        - requirements

    All other non-generic heuristics remain unchanged.
    """

    section = clean_section(section)

    # 1. Empty section
    if not section:
        return False

    # 2. Exact generic section
    if section.lower() in GENERIC_SECTIONS:
        return False

    # 3. One-word section
    if len(section.split()) <= 1:
        return False

    # 4. Breadcrumb final component is generic
    parts = [part.strip() for part in section.split(">")]
    final_component = parts[-1].strip().lower()

    if final_component in GENERIC_SECTIONS:
        return False

    # 5. Generic prefix
    for generic in GENERIC_SECTIONS:
        if final_component.startswith(generic + " "):
            return False

    # 6. Extremely long section
    if len(section) > 300:
        return False

    return True


def build_representation(
    title: str,
    section: str | None,
    content: str,
    source: str,
) -> str:
    """
    E28-B representation.

    Baseline:
        C7 conditional section representation.

    Ablation:
        Only migration source has section disabled.

    Everything else remains identical to C7.
    """

    title = title or ""
    content = content or ""

    # ---------------------------------------------------------------
    # E28-B ablation:
    # migration source -> title + content
    # ---------------------------------------------------------------

    if source == MIGRATION_SOURCE:
        return f"{title}\n\n{content}"

    # ---------------------------------------------------------------
    # All other sources -> exact C7 conditional representation
    # ---------------------------------------------------------------

    if section_quality(section):
        return f"{title}\n\n{clean_section(section)}\n\n{content}"

    return f"{title}\n\n{content}"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # Match previous experiments:
            # q025 / q041 are spec questions and excluded.
            if item.get("domain") == "spec":
                continue

            dataset.append(item)

    return dataset


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


def normalize_source(source: str | None) -> str:
    if not source:
        return ""

    return source.rstrip("/")


def is_same_source(
    result_source: str | None,
    relevant_source: str | None,
) -> bool:
    return (
        normalize_source(result_source)
        == normalize_source(relevant_source)
    )


def get_relevant_sources(item: dict) -> set[str]:
    sources = item.get("relevant_sources", [])

    if isinstance(sources, str):
        return {normalize_source(sources)}

    return {
        normalize_source(source)
        for source in sources
        if source
    }


def get_result_source(result: dict) -> str:
    return normalize_source(
        result.get("source")
        or result.get("url")
        or result.get("metadata", {}).get("source")
        or ""
    )


def get_result_title(result: dict) -> str:
    return (
        result.get("title")
        or result.get("metadata", {}).get("title")
        or ""
    )


def get_result_section(result: dict) -> str:
    return (
        result.get("section")
        or result.get("metadata", {}).get("section")
        or ""
    )


def get_result_content(result: dict) -> str:
    return result.get("content") or ""


def get_similarity(result: dict) -> float:
    """
    Support the similarity field names used by the retrieval code.
    """

    value = (
        result.get("similarity")
        if result.get("similarity") is not None
        else result.get("score")
    )

    if value is None:
        return 0.0

    return float(value)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def apply_source_diversity(
    results: list[dict],
    max_chunks_per_source: int,
) -> list[dict]:
    """
    Keep at most N chunks from each source.

    This reproduces the C7 / E25 diversity behavior.
    """

    counts: dict[str, int] = {}
    selected: list[dict] = []

    for result in results:
        source = get_result_source(result)

        count = counts.get(source, 0)

        if count >= max_chunks_per_source:
            continue

        counts[source] = count + 1
        selected.append(result)

    return selected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def first_relevant_rank(
    results: list[dict],
    relevant_sources: set[str],
) -> int | None:
    for rank, result in enumerate(results, start=1):
        source = get_result_source(result)

        if source in relevant_sources:
            return rank

    return None


def hit_at_k(
    ranks: list[int | None],
    k: int,
) -> float:
    hits = sum(
        1
        for rank in ranks
        if rank is not None and rank <= k
    )

    return hits / len(ranks) if ranks else 0.0


def recall_at_k(
    ranks: list[int | None],
    k: int,
) -> float:
    """
    Same definition used by the previous evaluation:
    first relevant source within top-k.
    """

    return hit_at_k(ranks, k)


def mean_reciprocal_rank(
    ranks: list[int | None],
) -> float:
    values = [
        1.0 / rank
        for rank in ranks
        if rank is not None
    ]

    return mean(values) if values else 0.0


def calculate_metrics(
    ranks: list[int | None],
) -> dict:
    return {
        "hit@1": hit_at_k(ranks, 1),
        "hit@3": hit_at_k(ranks, 3),
        "hit@5": hit_at_k(ranks, 5),
        "hit@10": hit_at_k(ranks, 10),
        "recall@10": recall_at_k(ranks, 10),
        "mrr": mean_reciprocal_rank(ranks),
    }


# ---------------------------------------------------------------------------
# Load E27 migration residual IDs
# ---------------------------------------------------------------------------


def load_migration_residual_ids() -> list[str]:
    """
    Read the 10 migration Top1 residual cases from E27 input.

    We use the already-generated E27 input instead of re-running E27.
    """

    if not E27_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"E27 input not found: {E27_INPUT_PATH}"
        )

    with E27_INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []

    questions = data.get("questions", [])

    for item in questions:
        top10 = item.get("top10", [])

        if not top10:
            continue

        top1 = top10[0]

        source = normalize_source(
            top1.get("source")
            or top1.get("url")
            or ""
        )

        if source == normalize_source(MIGRATION_SOURCE):
            ids.append(item["id"])

    return ids


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_question(
    model: BGEEmbeddingModel,
    item: dict,
) -> dict:
    question_id = item["id"]
    question = item["question"]

    relevant_sources = get_relevant_sources(item)

    query_embedding = model.embed_query(question)

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    candidates = search_chunks(
        query_embedding,
        limit=CANDIDATE_LIMIT,
    )

    # ------------------------------------------------------------------
    # Build E28-B representation
    #
    # IMPORTANT:
    # We do NOT re-embed the candidates here.
    #
    # search_chunks() already returns the vector similarity produced by
    # the database embedding.
    #
    # Therefore this script assumes the representation change is applied
    # before ingestion / embedding in the same way as the previous E25
    # experiments.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # IMPORTANT SAFETY CHECK
    #
    # This experiment must actually compare embeddings generated from
    # the E28-B representation. Therefore, candidates are re-embedded
    # below using the same query/candidate cosine similarity.
    # ------------------------------------------------------------------

    candidate_texts = []

    for candidate in candidates:
        title = get_result_title(candidate)
        section = get_result_section(candidate)
        content = get_result_content(candidate)
        source = get_result_source(candidate)

        representation = build_representation(
            title=title,
            section=section,
            content=content,
            source=source,
        )

        candidate_texts.append(representation)

    candidate_embeddings = model.embed_documents(
        candidate_texts,
        batch_size=16,
    )

    # query_embedding is already normalized.
    # candidate_embeddings are normalized by BGEEmbeddingModel.
    similarities = candidate_embeddings @ query_embedding

    reranked = []

    for candidate, similarity in zip(
        candidates,
        similarities,
    ):
        result = dict(candidate)

        result["e28_similarity"] = float(similarity)

        reranked.append(result)

    reranked.sort(
        key=lambda x: x["e28_similarity"],
        reverse=True,
    )

    # ------------------------------------------------------------------
    # Source diversity
    # ------------------------------------------------------------------

    final_results = apply_source_diversity(
        reranked,
        MAX_CHUNKS_PER_SOURCE,
    )

    final_results = final_results[:TOP_K]

    gold_rank = first_relevant_rank(
        final_results,
        relevant_sources,
    )

    return {
        "id": question_id,
        "question": question,
        "gold_rank": gold_rank,
        "relevant_sources": sorted(relevant_sources),
        "top10": [
            {
                "rank": rank,
                "title": get_result_title(result),
                "section": get_result_section(result),
                "source": get_result_source(result),
                "similarity": get_similarity_e28(result),
            }
            for rank, result in enumerate(
                final_results,
                start=1,
            )
        ],
    }


def get_similarity_e28(result: dict) -> float:
    return float(result.get("e28_similarity", 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 100)
    print("E28-B - MIGRATION SECTION ABLATION")
    print("=" * 100)

    print()
    print("Representation:")
    print("  Baseline: E25 C7")
    print("  Ablation: migration -> title + content")
    print("  Other sources -> C7 conditional representation")
    print()
    print(f"Candidate limit       : {CANDIDATE_LIMIT}")
    print(f"Top K                 : {TOP_K}")
    print(
        f"Max chunks per source: "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )
    print(
        "Generic sections      : "
        f"{sorted(GENERIC_SECTIONS)}"
    )
    print()

    dataset = load_dataset()

    print(f"Questions              : {len(dataset)}")

    migration_ids = load_migration_residual_ids()

    print(
        "E27 migration residuals: "
        f"{len(migration_ids)}"
    )

    print(
        "IDs:",
        ", ".join(migration_ids),
    )

    print()
    print("Loading embedding model...")
    model = BGEEmbeddingModel()

    results = []
    ranks = []

    for index, item in enumerate(dataset, start=1):
        print(
            f"[{index:02d}/{len(dataset):02d}] "
            f"{item['id']} "
            f"{item['question']}"
        )

        result = retrieve_question(
            model,
            item,
        )

        results.append(result)
        ranks.append(result["gold_rank"])

        print(
            f"    gold_rank = {result['gold_rank']}"
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    metrics = calculate_metrics(ranks)

    # ------------------------------------------------------------------
    # Migration residual details
    # ------------------------------------------------------------------

    migration_details = []

    migration_id_set = set(migration_ids)

    for result in results:
        if result["id"] not in migration_id_set:
            continue

        top10 = result["top10"]

        migration_rank = None

        for row in top10:
            source = normalize_source(
                row["source"]
            )

            if source == normalize_source(
                MIGRATION_SOURCE
            ):
                migration_rank = row["rank"]
                break

        migration_details.append(
            {
                "id": result["id"],
                "question": result["question"],
                "gold_rank": result["gold_rank"],
                "migration_rank": migration_rank,
                "top10": top10,
            }
        )

    # ------------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": "E28-B",
        "name": "migration_section_ablation",
        "dataset": str(DATASET_PATH),
        "question_count": len(dataset),
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "max_chunks_per_source": MAX_CHUNKS_PER_SOURCE,
        "generic_sections": sorted(GENERIC_SECTIONS),
        "representation": {
            "baseline": "E25 C7",
            "migration": "title + content",
            "other_sources": (
                "title + conditional section + content"
            ),
        },
        "metrics": metrics,
        "migration_residual_ids": migration_ids,
        "migration_details": migration_details,
        "questions": results,
    }

    output_path = (
        RESULTS_DIR
        / "e28_migration_section_ablation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("E28-B RESULTS")
    print("=" * 100)

    print(
        f"Questions : {len(dataset)}"
    )

    print(
        f"Hit@1    : {metrics['hit@1']:.4f}"
    )
    print(
        f"Hit@3    : {metrics['hit@3']:.4f}"
    )
    print(
        f"Hit@5    : {metrics['hit@5']:.4f}"
    )
    print(
        f"Hit@10   : {metrics['hit@10']:.4f}"
    )
    print(
        f"Recall@10: {metrics['recall@10']:.4f}"
    )
    print(
        f"MRR      : {metrics['mrr']:.4f}"
    )

    # ------------------------------------------------------------------
    # Migration residual summary
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("MIGRATION RESIDUALS")
    print("=" * 100)

    for item in migration_details:
        print(
            f"{item['id']} | "
            f"gold_rank={item['gold_rank']} | "
            f"migration_rank={item['migration_rank']} | "
            f"{item['question']}"
        )

    print()
    print(
        "Saved to:"
    )
    print(
        output_path
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()