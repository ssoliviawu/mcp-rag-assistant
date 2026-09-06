from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

# ============================================================================
# Project root
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Imports
# ============================================================================

from embedding.model import BGEEmbeddingModel
from retrieval.search import search_chunks

# ============================================================================
# Configuration
# ============================================================================

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


# ============================================================================
# Section logic - exact E25 C7
# ============================================================================

def clean_section(section: str | None) -> str:
    if not section:
        return ""

    return " ".join(str(section).split())


def section_quality(section: str | None) -> bool:
    """
    Exact E25 C7 section-quality logic.

    Generic rules:
        recap
        installation
        requirements

    Other heuristics:
        empty
        <= 1 word
        breadcrumb final component generic
        generic prefix
        > 300 chars
    """

    section = clean_section(section)

    # 1. Empty
    if not section:
        return False

    # 2. Exact generic
    if section.lower() in GENERIC_SECTIONS:
        return False

    # 3. One word
    if len(section.split()) <= 1:
        return False

    # 4. Breadcrumb final component generic
    parts = [part.strip() for part in section.split(">")]
    final_component = parts[-1].lower()

    if final_component in GENERIC_SECTIONS:
        return False

    # 5. Generic prefix
    for generic in GENERIC_SECTIONS:
        if final_component.startswith(generic + " "):
            return False

    # 6. Too long
    if len(section) > 300:
        return False

    return True


# ============================================================================
# Helpers
# ============================================================================

def normalize_source(source: str | None) -> str:
    if not source:
        return ""

    return source.rstrip("/")


def get_source(result: dict) -> str:
    return normalize_source(
        result.get("source")
        or result.get("url")
        or result.get("metadata", {}).get("source")
        or ""
    )


def get_title(result: dict) -> str:
    return (
        result.get("title")
        or result.get("metadata", {}).get("title")
        or ""
    )


def get_section(result: dict) -> str:
    return (
        result.get("section")
        or result.get("metadata", {}).get("section")
        or ""
    )


def get_content(result: dict) -> str:
    return result.get("content") or ""


def get_similarity(result: dict) -> float:
    if result.get("e28_similarity") is not None:
        return float(result["e28_similarity"])

    if result.get("similarity") is not None:
        return float(result["similarity"])

    if result.get("score") is not None:
        return float(result["score"])

    return 0.0


# ============================================================================
# Representation builders
# ============================================================================

def build_c7_representation(result: dict) -> str:
    """
    E25 C7 representation for all normal sources.
    """

    title = get_title(result)
    section = get_section(result)
    content = get_content(result)

    if section_quality(section):
        return (
            f"{title}\n\n"
            f"{clean_section(section)}\n\n"
            f"{content}"
        )

    return f"{title}\n\n{content}"


def build_migration_representation(
    result: dict,
    mode: str,
) -> str:
    """
    Representation used ONLY for migration source.

    Modes:

        full
            title + section + content

        title_content
            title + content

        title
            title only

        content
            content only
    """

    title = get_title(result)
    section = clean_section(get_section(result))
    content = get_content(result)

    if mode == "full":
        if section:
            return (
                f"{title}\n\n"
                f"{section}\n\n"
                f"{content}"
            )

        return f"{title}\n\n{content}"

    if mode == "title_content":
        return f"{title}\n\n{content}"

    if mode == "title":
        return title

    if mode == "content":
        return content

    raise ValueError(
        f"Unknown migration representation mode: {mode}"
    )


# ============================================================================
# Source diversity
# ============================================================================

def apply_source_diversity(
    results: list[dict],
    max_chunks_per_source: int,
) -> list[dict]:
    counts: dict[str, int] = {}
    selected: list[dict] = []

    for result in results:
        source = get_source(result)

        count = counts.get(source, 0)

        if count >= max_chunks_per_source:
            continue

        counts[source] = count + 1
        selected.append(result)

    return selected


# ============================================================================
# Dataset
# ============================================================================

def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = []

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # Same as previous experiments:
            # exclude spec questions q025 / q041
            if item.get("domain") == "spec":
                continue

            dataset.append(item)

    return dataset


# ============================================================================
# Relevant sources
# ============================================================================

def get_relevant_sources(item: dict) -> set[str]:
    sources = item.get("relevant_sources", [])

    if isinstance(sources, str):
        return {
            normalize_source(sources)
        }

    return {
        normalize_source(source)
        for source in sources
        if source
    }


# ============================================================================
# Ranking / metrics
# ============================================================================

def first_relevant_rank(
    results: list[dict],
    relevant_sources: set[str],
) -> int | None:

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if get_source(result) in relevant_sources:
            return rank

    return None


def hit_at_k(
    ranks: list[int | None],
    k: int,
) -> float:

    if not ranks:
        return 0.0

    hits = sum(
        1
        for rank in ranks
        if rank is not None and rank <= k
    )

    return hits / len(ranks)


def mrr(
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
        "mrr": mrr(ranks),
    }


# ============================================================================
# E27 migration residual IDs
# ============================================================================

def load_migration_residual_ids() -> list[str]:

    if not E27_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"E27 input not found: {E27_INPUT_PATH}"
        )

    with E27_INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    ids = []

    for item in data.get("questions", []):

        top10 = item.get("top10", [])

        if not top10:
            continue

        top1 = top10[0]

        source = normalize_source(
            top1.get("source")
            or top1.get("url")
            or ""
        )

        if source == normalize_source(
            MIGRATION_SOURCE
        ):
            ids.append(item["id"])

    return ids


# ============================================================================
# One experiment
# ============================================================================

def run_experiment(
    model: BGEEmbeddingModel,
    dataset: list[dict],
    migration_mode: str,
) -> dict:

    question_results = []
    ranks = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        question_id = item["id"]
        question = item["question"]

        print(
            f"    [{index:02d}/{len(dataset):02d}] "
            f"{question_id}"
        )

        relevant_sources = get_relevant_sources(item)

        # ---------------------------------------------------------------
        # Query embedding
        # ---------------------------------------------------------------

        query_embedding = model.embed_query(
            question
        )

        # ---------------------------------------------------------------
        # Fixed candidate pool
        # ---------------------------------------------------------------

        candidates = search_chunks(
            query_embedding,
            limit=CANDIDATE_LIMIT,
        )

        # ---------------------------------------------------------------
        # Build representations
        # ---------------------------------------------------------------

        texts = []

        for candidate in candidates:

            source = get_source(candidate)

            if source == normalize_source(
                MIGRATION_SOURCE
            ):
                text = build_migration_representation(
                    candidate,
                    migration_mode,
                )
            else:
                text = build_c7_representation(
                    candidate
                )

            texts.append(text)

        # ---------------------------------------------------------------
        # Candidate embeddings
        # ---------------------------------------------------------------

        candidate_embeddings = model.embed_documents(
            texts,
            batch_size=16,
        )

        # Both query and candidate embeddings are normalized.
        similarities = (
            candidate_embeddings @ query_embedding
        )

        reranked = []

        for candidate, similarity in zip(
            candidates,
            similarities,
        ):

            result = dict(candidate)

            result["e28_similarity"] = float(
                similarity
            )

            reranked.append(result)

        reranked.sort(
            key=lambda x: x["e28_similarity"],
            reverse=True,
        )

        # ---------------------------------------------------------------
        # Source diversity
        # ---------------------------------------------------------------

        final_results = apply_source_diversity(
            reranked,
            MAX_CHUNKS_PER_SOURCE,
        )

        final_results = final_results[:TOP_K]

        gold_rank = first_relevant_rank(
            final_results,
            relevant_sources,
        )

        ranks.append(gold_rank)

        question_results.append(
            {
                "id": question_id,
                "question": question,
                "gold_rank": gold_rank,
                "top10": [
                    {
                        "rank": rank,
                        "title": get_title(result),
                        "section": get_section(result),
                        "source": get_source(result),
                        "similarity": get_similarity(
                            result
                        ),
                    }
                    for rank, result in enumerate(
                        final_results,
                        start=1,
                    )
                ],
            }
        )

    return {
        "metrics": calculate_metrics(ranks),
        "questions": question_results,
    }


# ============================================================================
# Migration residual analysis
# ============================================================================

def extract_migration_details(
    experiment_result: dict,
    migration_ids: list[str],
) -> list[dict]:

    migration_id_set = set(
        migration_ids
    )

    details = []

    for question in experiment_result["questions"]:

        if question["id"] not in migration_id_set:
            continue

        migration_rank = None

        for row in question["top10"]:

            if normalize_source(
                row["source"]
            ) == normalize_source(
                MIGRATION_SOURCE
            ):
                migration_rank = row["rank"]
                break

        details.append(
            {
                "id": question["id"],
                "question": question["question"],
                "gold_rank": question["gold_rank"],
                "migration_rank": migration_rank,
            }
        )

    return details


# ============================================================================
# Compare experiments
# ============================================================================

def compare_experiments(
    baseline: dict,
    experiment: dict,
) -> dict:

    baseline_questions = {
        item["id"]: item
        for item in baseline["questions"]
    }

    experiment_questions = {
        item["id"]: item
        for item in experiment["questions"]
    }

    improved = []
    worsened = []
    unchanged = []

    for question_id in baseline_questions:

        baseline_rank = baseline_questions[
            question_id
        ]["gold_rank"]

        experiment_rank = experiment_questions[
            question_id
        ]["gold_rank"]

        if (
            baseline_rank is not None
            and experiment_rank is not None
            and experiment_rank > baseline_rank
        ):
            improved.append(
                {
                    "id": question_id,
                    "baseline_rank": baseline_rank,
                    "experiment_rank": experiment_rank,
                    "delta": (
                        baseline_rank
                        - experiment_rank
                    ),
                }
            )

        elif (
            baseline_rank is not None
            and experiment_rank is not None
            and experiment_rank < baseline_rank
        ):
            worsened.append(
                {
                    "id": question_id,
                    "baseline_rank": baseline_rank,
                    "experiment_rank": experiment_rank,
                    "delta": (
                        baseline_rank
                        - experiment_rank
                    ),
                }
            )

        else:
            unchanged.append(question_id)

    return {
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "counts": {
            "improved": len(improved),
            "worsened": len(worsened),
            "unchanged": len(unchanged),
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 100)
    print(
        "E28-C - MIGRATION REPRESENTATION DECOMPOSITION"
    )
    print("=" * 100)

    print()
    print("Fixed configuration:")
    print(
        f"  Candidate limit       : {CANDIDATE_LIMIT}"
    )
    print(
        f"  Top K                 : {TOP_K}"
    )
    print(
        f"  Max chunks per source: "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )
    print(
        "  Generic sections      : "
        f"{sorted(GENERIC_SECTIONS)}"
    )

    print()
    print("Migration representations:")
    print(
        "  C0 = full C7:             "
        "title + conditional section + content"
    )
    print(
        "  C1 = migration full:      "
        "title + section + content"
    )
    print(
        "  C2 = migration tc:        "
        "title + content"
    )
    print(
        "  C3 = migration title:     "
        "title only"
    )
    print(
        "  C4 = migration content:   "
        "content only"
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset = load_dataset()

    print()
    print(
        f"Questions: {len(dataset)}"
    )

    migration_ids = (
        load_migration_residual_ids()
    )

    print(
        "Migration residuals: "
        f"{len(migration_ids)}"
    )

    print(
        "IDs:",
        ", ".join(migration_ids),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = BGEEmbeddingModel()

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------

    experiments = {}

    modes = {
        "C0_C7_baseline": "c7",
        "C1_migration_full": "full",
        "C2_migration_title_content": "title_content",
        "C3_migration_title": "title",
        "C4_migration_content": "content",
    }

    for experiment_name, mode in modes.items():

        print()
        print("=" * 100)
        print(experiment_name)
        print("=" * 100)

        if mode == "c7":
            # ----------------------------------------------------------
            # Baseline:
            #
            # All sources use C7.
            # ----------------------------------------------------------

            result = run_experiment(
                model=model,
                dataset=dataset,
                migration_mode="full",
            )

            # The run_experiment function normally gives migration
            # special handling. For the baseline we need migration to
            # use C7 too. Since C7's migration section passes the
            # section_quality check, "full" is equivalent here.
            #
            # This is intentional.
            experiments[
                experiment_name
            ] = result

        else:

            result = run_experiment(
                model=model,
                dataset=dataset,
                migration_mode=mode,
            )

            experiments[
                experiment_name
            ] = result

        metrics = result["metrics"]

        print()
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
            f"MRR      : {metrics['mrr']:.4f}"
        )

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    baseline = experiments[
        "C0_C7_baseline"
    ]

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    comparisons = {}

    for name, result in experiments.items():

        if name == "C0_C7_baseline":
            continue

        comparisons[name] = compare_experiments(
            baseline,
            result,
        )

    # ------------------------------------------------------------------
    # Migration residual details
    # ------------------------------------------------------------------

    migration_details = {}

    for name, result in experiments.items():

        migration_details[name] = (
            extract_migration_details(
                result,
                migration_ids,
            )
        )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("E28-C SUMMARY")
    print("=" * 100)

    print()
    print(
        f"{'Experiment':<35}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 85)

    for name, result in experiments.items():

        metrics = result["metrics"]

        print(
            f"{name:<35}"
            f"{metrics['hit@1']:>10.4f}"
            f"{metrics['hit@3']:>10.4f}"
            f"{metrics['hit@5']:>10.4f}"
            f"{metrics['hit@10']:>10.4f}"
            f"{metrics['mrr']:>10.4f}"
        )

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CHANGES VS C7")
    print("=" * 100)

    for name, comparison in comparisons.items():

        metrics = experiments[
            name
        ]["metrics"]

        baseline_metrics = baseline[
            "metrics"
        ]

        print()
        print(name)

        print(
            "  Hit@1 delta    : "
            f"{metrics['hit@1'] - baseline_metrics['hit@1']:+.4f}"
        )

        print(
            "  Hit@3 delta    : "
            f"{metrics['hit@3'] - baseline_metrics['hit@3']:+.4f}"
        )

        print(
            "  Hit@5 delta    : "
            f"{metrics['hit@5'] - baseline_metrics['hit@5']:+.4f}"
        )

        print(
            "  Hit@10 delta   : "
            f"{metrics['hit@10'] - baseline_metrics['hit@10']:+.4f}"
        )

        print(
            "  MRR delta      : "
            f"{metrics['mrr'] - baseline_metrics['mrr']:+.4f}"
        )

        print(
            "  Improved       : "
            f"{comparison['counts']['improved']}"
        )

        print(
            "  Worsened       : "
            f"{comparison['counts']['worsened']}"
        )

        print(
            "  Unchanged      : "
            f"{comparison['counts']['unchanged']}"
        )

    # ------------------------------------------------------------------
    # Migration residual rank table
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("MIGRATION RESIDUAL RANKS")
    print("=" * 100)

    print()
    print(
        f"{'ID':<8}"
        f"{'C0':>8}"
        f"{'C1':>8}"
        f"{'C2':>8}"
        f"{'C3':>8}"
        f"{'C4':>8}"
    )

    print("-" * 48)

    experiment_keys = [
        "C0_C7_baseline",
        "C1_migration_full",
        "C2_migration_title_content",
        "C3_migration_title",
        "C4_migration_content",
    ]

    rank_maps = {}

    for name in experiment_keys:

        rank_maps[name] = {
            item["id"]: item["gold_rank"]
            for item in experiments[name][
                "questions"
            ]
        }

    for question_id in migration_ids:

        print(
            f"{question_id:<8}"
            f"{str(rank_maps['C0_C7_baseline'].get(question_id)):>8}"
            f"{str(rank_maps['C1_migration_full'].get(question_id)):>8}"
            f"{str(rank_maps['C2_migration_title_content'].get(question_id)):>8}"
            f"{str(rank_maps['C3_migration_title'].get(question_id)):>8}"
            f"{str(rank_maps['C4_migration_content'].get(question_id)):>8}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": "E28-C",
        "name": (
            "migration_representation_decomposition"
        ),
        "dataset": str(DATASET_PATH),
        "question_count": len(dataset),
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "max_chunks_per_source": (
            MAX_CHUNKS_PER_SOURCE
        ),
        "generic_sections": sorted(
            GENERIC_SECTIONS
        ),
        "migration_source": MIGRATION_SOURCE,
        "representations": {
            "C0_C7_baseline": (
                "all sources: "
                "title + conditional section + content"
            ),
            "C1_migration_full": (
                "migration: "
                "title + section + content"
            ),
            "C2_migration_title_content": (
                "migration: "
                "title + content"
            ),
            "C3_migration_title": (
                "migration: title only"
            ),
            "C4_migration_content": (
                "migration: content only"
            ),
        },
        "metrics": {
            name: result["metrics"]
            for name, result
            in experiments.items()
        },
        "comparisons_vs_c7": comparisons,
        "migration_residual_ids": migration_ids,
        "migration_details": migration_details,
        "experiments": experiments,
    }

    output_path = (
        RESULTS_DIR
        / "e28_migration_representation_decomposition.json"
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

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print()
    print(
        "Saved to:"
    )
    print(
        output_path
    )


if __name__ == "__main__":
    main()