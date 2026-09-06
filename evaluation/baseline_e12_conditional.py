import json
import re
import statistics
from pathlib import Path

from embedding.model import BGEEmbeddingModel
from db.repository import search_chunks


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

E6_DETAILS_PATH = (
    RESULTS_DIR / "e6_lexical_alpha_details.json"
)

E8_PATH = (
    RESULTS_DIR / "e8_e6_vs_e2_analysis.json"
)

OUTPUT_PATH = (
    RESULTS_DIR / "e12_conditional_lexical_summary.json"
)

CANDIDATE_LIMIT = 100
TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 1

ALPHA = 0.05
SEMANTIC_GAP_THRESHOLD = 0.01
LEXICAL_GAP_THRESHOLD = 0.20


_model = BGEEmbeddingModel()


# ============================================================
# URL normalization
# ============================================================

def normalize_url(url):
    if not url:
        return ""

    return str(url).strip().rstrip("/")


# ============================================================
# Lexical score
# ============================================================

def tokenize(text):
    if not text:
        return set()

    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        text.lower(),
    )

    return {
        token
        for token in tokens
        if len(token) >= 2
    }


def lexical_score(query, content):
    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    content_tokens = tokenize(content)

    matched = query_tokens & content_tokens

    return len(matched) / len(query_tokens)


# ============================================================
# Source diversity
# ============================================================

def apply_source_diversity(
    results,
    top_k=TOP_K,
    max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
):
    output = []
    source_counts = {}

    for result in results:
        source = normalize_url(
            result.get("url", "")
        )

        if (
            source_counts.get(source, 0)
            >= max_chunks_per_source
        ):
            continue

        output.append(result)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(output) >= top_k:
            break

    return output


# ============================================================
# E6 global lexical
# ============================================================

def e6_rank(
    candidates,
    alpha=ALPHA,
):
    scored = []

    for item in candidates:
        similarity = float(
            item.get("similarity", 0.0)
        )

        lexical = float(
            item.get("lexical_score", 0.0)
        )

        final_score = (
            similarity
            + alpha * lexical
        )

        new_item = dict(item)

        new_item["final_score"] = final_score

        scored.append(new_item)

    scored.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return apply_source_diversity(scored)


# ============================================================
# E12 conditional lexical
# ============================================================

def conditional_rank(
    candidates,
    alpha=ALPHA,
    semantic_gap_threshold=SEMANTIC_GAP_THRESHOLD,
    lexical_gap_threshold=LEXICAL_GAP_THRESHOLD,
):
    """
    Conditional lexical correction.

    Candidates are initially ordered by vector similarity.

    For each candidate, compare it with the immediately
    preceding candidate.

    Lexical boost is allowed only when:

        semantic_gap <= 0.01
        AND
        lexical_gap >= 0.20

    Otherwise keep pure semantic score.
    """

    scored = []

    for i, item in enumerate(candidates):

        similarity = float(
            item.get("similarity", 0.0)
        )

        lexical = float(
            item.get("lexical_score", 0.0)
        )

        final_score = similarity
        boosted = False

        if i > 0:

            previous = candidates[i - 1]

            previous_similarity = float(
                previous.get("similarity", 0.0)
            )

            previous_lexical = float(
                previous.get("lexical_score", 0.0)
            )

            semantic_gap = (
                previous_similarity
                - similarity
            )

            lexical_gap = (
                lexical
                - previous_lexical
            )

            if (
                semantic_gap
                <= semantic_gap_threshold
                and lexical_gap
                >= lexical_gap_threshold
            ):
                final_score = (
                    similarity
                    + alpha * lexical
                )

                boosted = True

        new_item = dict(item)

        new_item["final_score"] = final_score
        new_item["conditional_boost"] = boosted

        scored.append(new_item)

    scored.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return apply_source_diversity(scored)


# ============================================================
# Gold helpers
# ============================================================

def get_gold_sources(dataset_item):
    sources = dataset_item.get(
        "relevant_sources",
        [],
    )

    return {
        normalize_url(source)
        for source in sources
        if source
    }


def get_gold_rank(results, gold_sources):
    for rank, item in enumerate(
        results,
        start=1,
    ):
        url = normalize_url(
            item.get("url", "")
        )

        if url in gold_sources:
            return rank

    return None


def hit_at_10(results, gold_sources):
    return get_gold_rank(
        results,
        gold_sources,
    ) is not None


# ============================================================
# Dataset
# ============================================================

def load_dataset():
    data = []

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # Exclude spec questions exactly as previous
            # retrieval evaluation did.
            if item.get("domain") == "spec":
                continue

            data.append(item)

    return data


# ============================================================
# Metrics
# ============================================================

def calculate_mrr(ranks):
    values = []

    for rank in ranks:
        if rank is None:
            values.append(0.0)
        else:
            values.append(1.0 / rank)

    return (
        sum(values) / len(values)
        if values
        else 0.0
    )


def calculate_metrics(rank_map):
    ranks = list(rank_map.values())

    hit1 = sum(
        1
        for rank in ranks
        if rank is not None and rank <= 1
    )

    hit3 = sum(
        1
        for rank in ranks
        if rank is not None and rank <= 3
    )

    hit5 = sum(
        1
        for rank in ranks
        if rank is not None and rank <= 5
    )

    hit10 = sum(
        1
        for rank in ranks
        if rank is not None and rank <= 10
    )

    n = len(ranks)

    return {
        "hit@1": hit1 / n,
        "hit@3": hit3 / n,
        "hit@5": hit5 / n,
        "hit@10": hit10 / n,
        "mrr": calculate_mrr(ranks),
        "questions": n,
        "hits@10": hit10,
        "misses@10": n - hit10,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E12 - CONDITIONAL LEXICAL RETRIEVAL")
    print("=" * 80)

    dataset = load_dataset()

    print()
    print("Questions :", len(dataset))
    print("Candidate :", CANDIDATE_LIMIT)
    print("Top K     :", TOP_K)
    print("Alpha     :", ALPHA)
    print(
        "Semantic gap <= ",
        SEMANTIC_GAP_THRESHOLD,
    )
    print(
        "Lexical gap >= ",
        LEXICAL_GAP_THRESHOLD,
    )

    e6_details = {}

    if E6_DETAILS_PATH.exists():

        with open(
            E6_DETAILS_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            e6_data = json.load(f)

        e6_results = (
            e6_data
            .get("results", {})
            .get("E6_05", [])
        )

        for item in e6_results:
            e6_details[item["id"]] = item

    e8_data = {}

    if E8_PATH.exists():

        with open(
            E8_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            raw_e8 = json.load(f)

        if isinstance(raw_e8, dict):

            if "questions" in raw_e8:

                for item in raw_e8["questions"]:
                    e8_data[item["id"]] = item

    print()
    print(
        "E6_05 details :",
        len(e6_details),
    )

    print(
        "E8 records   :",
        len(e8_data),
    )

    details = []

    e6_rank_map = {}
    e12_rank_map = {}

    improvement = 0
    worsening = 0
    unchanged = 0

    candidate_counts = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        qid = item["id"]
        question = item["question"]

        print(
            f"[{index:02d}/{len(dataset)}] {qid}"
        )

        gold_sources = get_gold_sources(item)

        # ----------------------------------------------------
        # Vector Top100
        # ----------------------------------------------------

        embedding = _model.embed_query(
            question
        )

        candidates = search_chunks(
            embedding,
            limit=CANDIDATE_LIMIT,
        )

        if not candidates:
            print("  WARNING: no candidates")
            continue

        candidate_counts.append(
            len(candidates)
        )

        # ----------------------------------------------------
        # Compute lexical score once
        # ----------------------------------------------------

        enriched = []

        for candidate in candidates:

            new_item = dict(candidate)

            new_item["lexical_score"] = (
                lexical_score(
                    question,
                    candidate.get(
                        "content",
                        "",
                    ),
                )
            )

            enriched.append(new_item)

        # ----------------------------------------------------
        # E6
        # ----------------------------------------------------

        e6_results = e6_rank(
            enriched,
            alpha=ALPHA,
        )

        # ----------------------------------------------------
        # E12
        # ----------------------------------------------------

        e12_results = conditional_rank(
            enriched,
            alpha=ALPHA,
            semantic_gap_threshold=(
                SEMANTIC_GAP_THRESHOLD
            ),
            lexical_gap_threshold=(
                LEXICAL_GAP_THRESHOLD
            ),
        )

        e6_rank_value = get_gold_rank(
            e6_results,
            gold_sources,
        )

        e12_rank_value = get_gold_rank(
            e12_results,
            gold_sources,
        )

        e6_rank_map[qid] = e6_rank_value
        e12_rank_map[qid] = e12_rank_value

        if (
            e12_rank_value is not None
            and e6_rank_value is not None
        ):

            if e12_rank_value < e6_rank_value:
                improvement += 1

            elif e12_rank_value > e6_rank_value:
                worsening += 1

            else:
                unchanged += 1

        elif (
            e12_rank_value is not None
            and e6_rank_value is None
        ):

            improvement += 1

        elif (
            e12_rank_value is None
            and e6_rank_value is not None
        ):

            worsening += 1

        else:
            unchanged += 1

        boosted_count = sum(
            1
            for x in e12_results
            if x.get(
                "conditional_boost",
                False,
            )
        )

        details.append(
            {
                "id": qid,
                "question": question,
                "gold_sources": list(
                    gold_sources
                ),
                "e6": {
                    "gold_rank": e6_rank_value,
                    "hit@10": (
                        e6_rank_value is not None
                        and e6_rank_value <= 10
                    ),
                },
                "e12": {
                    "gold_rank": e12_rank_value,
                    "hit@10": (
                        e12_rank_value is not None
                        and e12_rank_value <= 10
                    ),
                    "conditional_boost_count": (
                        boosted_count
                    ),
                },
            }
        )

    # ========================================================
    # Metrics
    # ========================================================

    e6_metrics = calculate_metrics(
        e6_rank_map
    )

    e12_metrics = calculate_metrics(
        e12_rank_map
    )

    output = {
        "experiment": "E12_conditional_lexical",
        "dataset": DATASET_PATH.name,
        "questions": len(dataset),
        "pipeline": {
            "candidate_limit": CANDIDATE_LIMIT,
            "top_k": TOP_K,
            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),
            "alpha": ALPHA,
            "semantic_gap_threshold": (
                SEMANTIC_GAP_THRESHOLD
            ),
            "lexical_gap_threshold": (
                LEXICAL_GAP_THRESHOLD
            ),
        },
        "metrics": {
            "E6": e6_metrics,
            "E12": e12_metrics,
        },
        "rank_change": {
            "improved": improvement,
            "worsened": worsening,
            "unchanged": unchanged,
        },
        "candidate_statistics": {
            "avg_candidates": (
                statistics.mean(candidate_counts)
                if candidate_counts
                else 0
            ),
            "min_candidates": (
                min(candidate_counts)
                if candidate_counts
                else 0
            ),
            "max_candidates": (
                max(candidate_counts)
                if candidate_counts
                else 0
            ),
        },
        "details": details,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)
    print("E12 RESULTS")
    print("=" * 80)

    print()
    print(
        f"{'Metric':<12}"
        f"{'E6':>12}"
        f"{'E12':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 48)

    for metric in [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "mrr",
    ]:

        e6_value = e6_metrics[metric]
        e12_value = e12_metrics[metric]

        print(
            f"{metric:<12}"
            f"{e6_value:>12.4f}"
            f"{e12_value:>12.4f}"
            f"{e12_value - e6_value:>+12.4f}"
        )

    print()
    print("=" * 80)
    print("RANK CHANGE")
    print("=" * 80)

    print(
        f"Improve : {improvement}"
    )

    print(
        f"Worsen  : {worsening}"
    )

    print(
        f"Same    : {unchanged}"
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print("=" * 80)


if __name__ == "__main__":
    main()