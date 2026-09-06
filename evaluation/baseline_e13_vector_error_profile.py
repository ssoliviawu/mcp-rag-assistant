import json
import re
import statistics
from pathlib import Path

from embedding.model import BGEEmbeddingModel
from db.repository import search_chunks


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

OUTPUT_PATH = (
    RESULTS_DIR / "e13_vector_error_profile.json"
)

CANDIDATE_LIMIT = 100


_model = BGEEmbeddingModel()


# ============================================================
# URL
# ============================================================

def normalize_url(url):
    if not url:
        return ""

    return str(url).strip().rstrip("/")


# ============================================================
# Token / text helpers
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

    return len(
        query_tokens & content_tokens
    ) / len(query_tokens)


def text_length(text):
    if not text:
        return 0

    return len(str(text))


def word_count(text):
    if not text:
        return 0

    return len(
        re.findall(
            r"\b\w+\b",
            str(text),
        )
    )


def code_ratio(text):
    """
    Rough heuristic only.

    We intentionally do NOT attempt AST parsing.
    This is diagnostic, not a semantic classifier.
    """

    if not text:
        return 0.0

    text = str(text)

    code_markers = [
        "```",
        "def ",
        "class ",
        "import ",
        "from ",
        "async def ",
        "await ",
        "return ",
        "self.",
        "@mcp.",
        "@server.",
    ]

    hits = sum(
        1
        for marker in code_markers
        if marker in text
    )

    return min(
        hits / 5.0,
        1.0,
    )


def is_code_heavy(text):
    return code_ratio(text) >= 0.4


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

            # Keep the same evaluation population
            # used by E2-E12.
            if item.get("domain") == "spec":
                continue

            data.append(item)

    return data


# ============================================================
# Gold helpers
# ============================================================

def get_gold_sources(item):
    return {
        normalize_url(source)
        for source in item.get(
            "relevant_sources",
            [],
        )
        if source
    }


def is_gold(result, gold_sources):
    return (
        normalize_url(
            result.get("url", "")
        )
        in gold_sources
    )


# ============================================================
# Rank helpers
# ============================================================

def first_gold_rank(
    results,
    gold_sources,
):
    for rank, result in enumerate(
        results,
        start=1,
    ):

        if is_gold(
            result,
            gold_sources,
        ):
            return rank

    return None


def gold_ranks(
    results,
    gold_sources,
):
    ranks = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if is_gold(
            result,
            gold_sources,
        ):
            ranks.append(rank)

    return ranks


# ============================================================
# Rank bucket
# ============================================================

def rank_bucket(rank):
    if rank is None:
        return "NOT_FOUND"

    if rank == 1:
        return "TOP1"

    if rank <= 3:
        return "TOP2_3"

    if rank <= 10:
        return "TOP4_10"

    if rank <= 20:
        return "TOP11_20"

    if rank <= 50:
        return "TOP21_50"

    return "TOP51_100"


# ============================================================
# Candidate representation
# ============================================================

def summarize_result(
    rank,
    result,
    query,
    gold_sources,
):
    content = result.get(
        "content",
        "",
    )

    similarity = result.get(
        "similarity",
        None,
    )

    if similarity is not None:
        similarity = float(similarity)

    lexical = lexical_score(
        query,
        content,
    )

    return {
        "rank": rank,
        "id": result.get("id"),
        "title": result.get("title"),
        "section": result.get("section"),
        "url": result.get("url"),
        "similarity": similarity,
        "lexical_score": lexical,
        "content_chars": text_length(
            content
        ),
        "content_words": word_count(
            content
        ),
        "code_ratio": code_ratio(
            content
        ),
        "code_heavy": is_code_heavy(
            content
        ),
        "is_gold": is_gold(
            result,
            gold_sources,
        ),
    }


# ============================================================
# Per-question analysis
# ============================================================

def analyze_question(item):
    qid = item["id"]
    question = item["question"]

    gold_sources = get_gold_sources(item)

    embedding = _model.embed_query(
        question
    )

    results = search_chunks(
        embedding,
        limit=CANDIDATE_LIMIT,
    )

    if not results:
        return {
            "id": qid,
            "question": question,
            "gold_sources": list(
                gold_sources
            ),
            "candidate_count": 0,
            "gold_rank": None,
            "rank_bucket": "NOT_FOUND",
            "gold_found": False,
            "error": "NO_RESULTS",
        }

    first_rank = first_gold_rank(
        results,
        gold_sources,
    )

    all_gold_ranks = gold_ranks(
        results,
        gold_sources,
    )

    # --------------------------------------------------------
    # Gold candidates
    # --------------------------------------------------------

    gold_results = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if is_gold(
            result,
            gold_sources,
        ):

            gold_results.append(
                summarize_result(
                    rank,
                    result,
                    question,
                    gold_sources,
                )
            )

    # --------------------------------------------------------
    # Top competitors
    # --------------------------------------------------------

    top_results = []

    for rank, result in enumerate(
        results[:10],
        start=1,
    ):

        top_results.append(
            summarize_result(
                rank,
                result,
                question,
                gold_sources,
            )
        )

    top1 = top_results[0]

    # --------------------------------------------------------
    # Best gold
    # --------------------------------------------------------

    best_gold = None

    if gold_results:

        best_gold = min(
            gold_results,
            key=lambda x: x["rank"],
        )

    # --------------------------------------------------------
    # Semantic gaps
    # --------------------------------------------------------

    semantic_gap_top1_to_gold = None

    if (
        best_gold is not None
        and top1.get("similarity") is not None
        and best_gold.get("similarity") is not None
    ):

        semantic_gap_top1_to_gold = (
            top1["similarity"]
            - best_gold["similarity"]
        )

    lexical_gap_top1_to_gold = None

    if best_gold is not None:

        lexical_gap_top1_to_gold = (
            top1["lexical_score"]
            - best_gold["lexical_score"]
        )

    # --------------------------------------------------------
    # Gold chunk statistics
    # --------------------------------------------------------

    if gold_results:

        gold_lengths = [
            x["content_chars"]
            for x in gold_results
        ]

        gold_words = [
            x["content_words"]
            for x in gold_results
        ]

        gold_code_ratios = [
            x["code_ratio"]
            for x in gold_results
        ]

        best_gold_length = best_gold[
            "content_chars"
        ]

        best_gold_words = best_gold[
            "content_words"
        ]

        best_gold_code_ratio = best_gold[
            "code_ratio"
        ]

    else:

        gold_lengths = []
        gold_words = []
        gold_code_ratios = []

        best_gold_length = None
        best_gold_words = None
        best_gold_code_ratio = None

    # --------------------------------------------------------
    # Top1 vs Gold characteristics
    # --------------------------------------------------------

    if best_gold is not None:

        top1_longer_than_gold = (
            top1["content_chars"]
            > best_gold["content_chars"]
        )

        top1_more_code_than_gold = (
            top1["code_ratio"]
            > best_gold["code_ratio"]
        )

    else:

        top1_longer_than_gold = None
        top1_more_code_than_gold = None

    # --------------------------------------------------------
    # Migration / troubleshooting / low-level etc.
    # --------------------------------------------------------

    top1_url = normalize_url(
        top1.get("url", "")
    )

    top1_characteristics = {
        "migration": (
            "/migration" in top1_url
        ),
        "troubleshooting": (
            "/troubleshooting" in top1_url
        ),
        "low_level": (
            "/low-level-server" in top1_url
        ),
        "structured_output": (
            "/structured-output" in top1_url
        ),
        "deployment": (
            "/run/deploy" in top1_url
        ),
        "code_heavy": top1[
            "code_heavy"
        ],
    }

    return {
        "id": qid,
        "question": question,
        "gold_sources": list(
            gold_sources
        ),
        "candidate_count": len(results),
        "gold_found": first_rank is not None,
        "gold_rank": first_rank,
        "rank_bucket": rank_bucket(
            first_rank
        ),
        "all_gold_ranks": all_gold_ranks,

        "top1": top1,

        "best_gold": best_gold,

        "comparison": {
            "semantic_gap_top1_minus_gold": (
                semantic_gap_top1_to_gold
            ),
            "lexical_gap_top1_minus_gold": (
                lexical_gap_top1_to_gold
            ),
            "top1_longer_than_gold": (
                top1_longer_than_gold
            ),
            "top1_more_code_than_gold": (
                top1_more_code_than_gold
            ),
        },

        "gold_statistics": {
            "gold_count_in_top100": len(
                gold_results
            ),
            "avg_gold_chars": (
                statistics.mean(
                    gold_lengths
                )
                if gold_lengths
                else None
            ),
            "avg_gold_words": (
                statistics.mean(
                    gold_words
                )
                if gold_words
                else None
            ),
            "avg_gold_code_ratio": (
                statistics.mean(
                    gold_code_ratios
                )
                if gold_code_ratios
                else None
            ),
            "best_gold_chars": (
                best_gold_length
            ),
            "best_gold_words": (
                best_gold_words
            ),
            "best_gold_code_ratio": (
                best_gold_code_ratio
            ),
        },

        "top1_characteristics": (
            top1_characteristics
        ),

        "top10": top_results,
        "gold_results": gold_results,
    }


# ============================================================
# Aggregate statistics
# ============================================================

def safe_mean(values):
    values = [
        x
        for x in values
        if x is not None
    ]

    if not values:
        return None

    return statistics.mean(values)


def count_true(records, path):
    count = 0

    for record in records:

        value = record

        try:
            for key in path:
                value = value[key]

            if value is True:
                count += 1

        except (
            KeyError,
            TypeError,
        ):
            pass

    return count


def build_summary(records):

    total = len(records)

    rank_buckets = {
        "TOP1": 0,
        "TOP2_3": 0,
        "TOP4_10": 0,
        "TOP11_20": 0,
        "TOP21_50": 0,
        "TOP51_100": 0,
        "NOT_FOUND": 0,
    }

    for record in records:

        bucket = record[
            "rank_bucket"
        ]

        rank_buckets[bucket] = (
            rank_buckets.get(
                bucket,
                0,
            )
            + 1
        )

    ranks = [
        r["gold_rank"]
        for r in records
        if r["gold_rank"] is not None
    ]

    semantic_gaps = [
        r["comparison"][
            "semantic_gap_top1_minus_gold"
        ]
        for r in records
        if r["comparison"][
            "semantic_gap_top1_minus_gold"
        ] is not None
    ]

    lexical_gaps = [
        r["comparison"][
            "lexical_gap_top1_minus_gold"
        ]
        for r in records
        if r["comparison"][
            "lexical_gap_top1_minus_gold"
        ] is not None
    ]

    # --------------------------------------------------------
    # Top1 source characteristics
    # --------------------------------------------------------

    migration = sum(
        1
        for r in records
        if r["top1_characteristics"][
            "migration"
        ]
    )

    troubleshooting = sum(
        1
        for r in records
        if r["top1_characteristics"][
            "troubleshooting"
        ]
    )

    low_level = sum(
        1
        for r in records
        if r["top1_characteristics"][
            "low_level"
        ]
    )

    structured_output = sum(
        1
        for r in records
        if r["top1_characteristics"][
            "structured_output"
        ]
    )

    deployment = sum(
        1
        for r in records
        if r["top1_characteristics"][
            "deployment"
        ]
    )

    code_heavy = sum(
        1
        for r in records
        if r["top1_characteristics"][
            "code_heavy"
        ]
    )

    # --------------------------------------------------------
    # Gold vs Top1
    # --------------------------------------------------------

    top1_longer = sum(
        1
        for r in records
        if r["comparison"][
            "top1_longer_than_gold"
        ] is True
    )

    top1_more_code = sum(
        1
        for r in records
        if r["comparison"][
            "top1_more_code_than_gold"
        ] is True
    )

    # --------------------------------------------------------
    # Near ties
    # --------------------------------------------------------

    near_tie_001 = sum(
        1
        for gap in semantic_gaps
        if abs(gap) <= 0.01
    )

    near_tie_003 = sum(
        1
        for gap in semantic_gaps
        if abs(gap) <= 0.03
    )

    near_tie_005 = sum(
        1
        for gap in semantic_gaps
        if abs(gap) <= 0.05
    )

    return {
        "total_questions": total,

        "gold_found_in_top100": len(
            ranks
        ),

        "gold_missed_in_top100": (
            total - len(ranks)
        ),

        "gold_rank": {
            "min": min(ranks)
            if ranks
            else None,
            "max": max(ranks)
            if ranks
            else None,
            "avg": safe_mean(ranks),
            "median": (
                statistics.median(ranks)
                if ranks
                else None
            ),
        },

        "rank_distribution": rank_buckets,

        "semantic_gap_top1_minus_gold": {
            "avg": safe_mean(
                semantic_gaps
            ),
            "median": (
                statistics.median(
                    semantic_gaps
                )
                if semantic_gaps
                else None
            ),
            "near_tie_<=_0.01": (
                near_tie_001
            ),
            "near_tie_<=_0.03": (
                near_tie_003
            ),
            "near_tie_<=_0.05": (
                near_tie_005
            ),
        },

        "lexical_gap_top1_minus_gold": {
            "avg": safe_mean(
                lexical_gaps
            ),
            "median": (
                statistics.median(
                    lexical_gaps
                )
                if lexical_gaps
                else None
            ),
        },

        "top1_characteristics": {
            "migration": migration,
            "troubleshooting": (
                troubleshooting
            ),
            "low_level": low_level,
            "structured_output": (
                structured_output
            ),
            "deployment": deployment,
            "code_heavy": code_heavy,
        },

        "top1_vs_gold": {
            "top1_longer_than_gold": (
                top1_longer
            ),
            "top1_more_code_than_gold": (
                top1_more_code
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E13 - VECTOR TOP100 ERROR PROFILE")
    print("=" * 80)

    dataset = load_dataset()

    print()
    print(
        "Questions :",
        len(dataset),
    )

    print(
        "Candidate :",
        CANDIDATE_LIMIT,
    )

    records = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        qid = item["id"]

        print(
            f"[{index:02d}/{len(dataset)}] {qid}"
        )

        record = analyze_question(
            item
        )

        records.append(record)

    summary = build_summary(
        records
    )

    output = {
        "experiment": "E13_vector_error_profile",
        "dataset": DATASET_PATH.name,
        "candidate_limit": CANDIDATE_LIMIT,
        "question_count": len(dataset),
        "summary": summary,
        "questions": records,
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
    # Console summary
    # ========================================================

    print()
    print("=" * 80)
    print("E13 SUMMARY")
    print("=" * 80)

    print()
    print(
        "Gold found in Top100 : "
        f"{summary['gold_found_in_top100']}"
        f"/{summary['total_questions']}"
    )

    print(
        "Gold missed          : "
        f"{summary['gold_missed_in_top100']}"
    )

    print()
    print("Gold rank distribution")
    print("-" * 40)

    for bucket, count in (
        summary[
            "rank_distribution"
        ].items()
    ):

        print(
            f"{bucket:<12} {count}"
        )

    print()
    print("Gold rank")
    print("-" * 40)

    print(
        "Average  :",
        summary["gold_rank"]["avg"],
    )

    print(
        "Median   :",
        summary["gold_rank"]["median"],
    )

    print(
        "Min      :",
        summary["gold_rank"]["min"],
    )

    print(
        "Max      :",
        summary["gold_rank"]["max"],
    )

    print()
    print(
        "Semantic gap "
        "(Top1 similarity - Gold similarity)"
    )

    print("-" * 40)

    gap = summary[
        "semantic_gap_top1_minus_gold"
    ]

    print(
        "Average          :",
        gap["avg"],
    )

    print(
        "Median           :",
        gap["median"],
    )

    print(
        "<= 0.01          :",
        gap["near_tie_<=_0.01"],
    )

    print(
        "<= 0.03          :",
        gap["near_tie_<=_0.03"],
    )

    print(
        "<= 0.05          :",
        gap["near_tie_<=_0.05"],
    )

    print()
    print("Top1 characteristics")
    print("-" * 40)

    for key, value in (
        summary[
            "top1_characteristics"
        ].items()
    ):

        print(
            f"{key:<20} {value}"
        )

    print()
    print("Top1 vs Gold")
    print("-" * 40)

    for key, value in (
        summary[
            "top1_vs_gold"
        ].items()
    ):

        print(
            f"{key:<30} {value}"
        )

    print()
    print("=" * 80)

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print("=" * 80)


if __name__ == "__main__":
    main()