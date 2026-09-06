
"""
Analyze LOW_RANK_HIT cases using actual chunk content from PostgreSQL.

Input:
    evaluation/results/error_analysis.json

Database:
    postgresql://rag:rag@localhost:5432/mcp_rag

Output:
    evaluation/results/low_rank_pattern_analysis.json

Run:
    uv run python -m evaluation.analyze_low_rank_patterns
"""

import json
import os
import re
from pathlib import Path
from collections import Counter

import psycopg


INPUT_FILE = Path(
    "evaluation/results/error_analysis.json"
)

OUTPUT_FILE = Path(
    "evaluation/results/low_rank_pattern_analysis.json"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/mcp_rag",
)


# ============================================================
# Helpers
# ============================================================

def normalize_url(url):
    return (url or "").rstrip("/")


def tokenize(text):
    return set(
        re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*|\b\d+\b",
            (text or "").lower(),
        )
    )


def lexical_overlap(question, content):
    q = tokenize(question)
    c = tokenize(content)

    if not q:
        return 0.0

    return len(q & c) / len(q)


def keyword_hits(question, content):
    stopwords = {
        "how",
        "do",
        "does",
        "did",
        "what",
        "when",
        "where",
        "why",
        "can",
        "could",
        "should",
        "would",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "for",
        "from",
        "and",
        "or",
        "with",
        "as",
        "by",
        "i",
        "it",
        "this",
        "that",
        "my",
        "your",
        "mcp",
    }

    q = tokenize(question)
    c = tokenize(content)

    return sorted(
        token
        for token in q & c
        if len(token) >= 3
        and token not in stopwords
    )


# ============================================================
# Database
# ============================================================

def load_chunk_contents(chunk_ids):
    """
    Load chunk content in one DB query.
    """

    if not chunk_ids:
        return {}

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT id, content
                FROM chunks
                WHERE id = ANY(%s)
                """,
                (list(chunk_ids),),
            )

            rows = cur.fetchall()

    return {
        str(chunk_id): content or ""
        for chunk_id, content in rows
    }


# ============================================================
# Load evaluation results
# ============================================================

def load_low_rank_cases():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data

    elif isinstance(data, dict):
        items = (
            data.get("results")
            or data.get("questions")
            or data.get("details")
            or []
        )

    else:
        items = []

    cases = []

    for item in items:

        # ------------------------------------------------------
        # Correct field for your error_analysis.json
        # ------------------------------------------------------

        if item.get("error_type") != "LOW_RANK_HIT":
            continue

        metrics = item.get(
            "metrics",
            {},
        )

        first_rank = metrics.get(
            "first_relevant_rank"
        )

        if first_rank is None:
            continue

        retrieved_details = item.get(
            "retrieved_details",
            [],
        )

        if not retrieved_details:
            continue

        top1 = retrieved_details[0]

        relevant_sources = {
            normalize_url(source)
            for source in (
                item.get(
                    "relevant_sources",
                    [],
                )
            )
            if source
        }

        gold = None

        for result in retrieved_details:

            source = normalize_url(
                result.get("url")
            )

            if source in relevant_sources:
                gold = result
                break

        if gold is None:
            continue

        cases.append({
            "id": item.get("id"),
            "question": item.get(
                "question",
                "",
            ),
            "first_relevant_rank": first_rank,
            "top1": top1,
            "gold": gold,
        })

    return cases


# ============================================================
# Analyze
# ============================================================

def analyze_case(case, contents):

    question = case["question"]

    top1 = case["top1"]
    gold = case["gold"]

    top1_id = str(
        top1.get("id")
    )

    gold_id = str(
        gold.get("id")
    )

    top1_content = contents.get(
        top1_id,
        "",
    )

    gold_content = contents.get(
        gold_id,
        "",
    )

    top1_source = normalize_url(
        top1.get("url")
    )

    gold_source = normalize_url(
        gold.get("url")
    )

    top1_similarity = top1.get(
        "similarity"
    )

    gold_similarity = gold.get(
        "similarity"
    )

    gap = None

    if (
        top1_similarity is not None
        and gold_similarity is not None
    ):
        gap = (
            top1_similarity
            - gold_similarity
        )

    top1_overlap = lexical_overlap(
        question,
        top1_content,
    )

    gold_overlap = lexical_overlap(
        question,
        gold_content,
    )

    title = (
        top1.get("title") or ""
    ).lower()

    migration = (
        "/migration" in top1_source
    )

    troubleshooting = (
        "/troubleshooting"
        in top1_source
    )

    low_level = (
        "/advanced/low-level-server"
        in top1_source
    )

    code_heavy = (
        "docs_src/" in title
        or "```" in top1_content
        or top1_content.count("\n") >= 12
    )

    # --------------------------------------------------------
    # Diagnostic classification
    # --------------------------------------------------------

    if gap is not None and gap < 0.01:

        category = "NEAR_TIE"

    elif (
        gold_overlap >= 0.25
        and top1_overlap
        < gold_overlap * 0.65
    ):

        category = "GOLD_MORE_DIRECT"

    elif (
        migration
        and gap is not None
        and gap < 0.05
        and top1_overlap < gold_overlap
    ):

        category = "TOP1_SUSPICIOUS"

    else:

        category = (
            "TOP1_ALSO_VALID_OR_AMBIGUOUS"
        )

    return {
        "id": case["id"],

        "question": question,

        "first_relevant_rank":
            case["first_relevant_rank"],

        "category": category,

        "similarity_gap": gap,

        "top1": {
            "id": top1_id,
            "rank": top1.get("rank"),
            "source": top1_source,
            "title": top1.get("title"),
            "section": top1.get("section"),
            "similarity": top1_similarity,

            "lexical_overlap": round(
                top1_overlap,
                4,
            ),

            "keyword_hits": keyword_hits(
                question,
                top1_content,
            ),

            "migration": migration,
            "troubleshooting":
                troubleshooting,
            "low_level_server":
                low_level,
            "code_heavy":
                code_heavy,

            "content": top1_content,
        },

        "gold": {
            "id": gold_id,
            "rank": gold.get("rank"),
            "source": gold_source,
            "title": gold.get("title"),
            "section": gold.get("section"),
            "similarity": gold_similarity,

            "lexical_overlap": round(
                gold_overlap,
                4,
            ),

            "keyword_hits": keyword_hits(
                question,
                gold_content,
            ),

            "content": gold_content,
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    cases = load_low_rank_cases()

    print(
        f"LOW_RANK_HIT cases: {len(cases)}"
    )

    if not cases:
        raise RuntimeError(
            "No LOW_RANK_HIT cases found."
        )

    # --------------------------------------------------------
    # Collect chunk IDs
    # --------------------------------------------------------

    chunk_ids = set()

    for case in cases:

        chunk_ids.add(
            str(case["top1"]["id"])
        )

        chunk_ids.add(
            str(case["gold"]["id"])
        )

    print(
        f"Chunk IDs to load: "
        f"{len(chunk_ids)}"
    )

    # --------------------------------------------------------
    # Load actual content
    # --------------------------------------------------------

    contents = load_chunk_contents(
        chunk_ids
    )

    print(
        f"Chunks loaded from DB: "
        f"{len(contents)}"
    )

    missing = (
        len(chunk_ids)
        - len(contents)
    )

    if missing:
        print(
            f"WARNING: "
            f"{missing} chunks missing"
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    results = [
        analyze_case(
            case,
            contents,
        )
        for case in cases
    ]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    category_counts = Counter(
        x["category"]
        for x in results
    )

    source_counts = Counter(
        x["top1"]["source"]
        for x in results
    )

    migration_count = sum(
        x["top1"]["migration"]
        for x in results
    )

    troubleshooting_count = sum(
        x["top1"]["troubleshooting"]
        for x in results
    )

    low_level_count = sum(
        x["top1"]["low_level_server"]
        for x in results
    )

    code_heavy_count = sum(
        x["top1"]["code_heavy"]
        for x in results
    )

    gaps = [
        x["similarity_gap"]
        for x in results
        if x["similarity_gap"] is not None
    ]

    summary = {
        "total_cases": len(results),

        "category_counts": dict(
            category_counts
        ),

        "top1_source_counts": dict(
            source_counts
        ),

        "top1_characteristics": {
            "migration": migration_count,
            "troubleshooting":
                troubleshooting_count,
            "low_level_server":
                low_level_count,
            "code_heavy":
                code_heavy_count,
        },

        "similarity_gap": {
            "lt_0.01": sum(
                x < 0.01
                for x in gaps
            ),

            "lt_0.03": sum(
                x < 0.03
                for x in gaps
            ),

            "lt_0.05": sum(
                x < 0.05
                for x in gaps
            ),

            "average": (
                round(
                    sum(gaps)
                    / len(gaps),
                    6,
                )
                if gaps
                else None
            ),
        },
    }

    output = {
        "summary": summary,
        "cases": results,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOW-RANK PATTERN ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal: {len(results)}"
    )

    print("\nCATEGORY")

    for key, value in sorted(
        category_counts.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"  {key}: {value}"
        )

    print("\nTOP1 SOURCES")

    for source, count in sorted(
        source_counts.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"  {count:2d}  {source}"
        )

    print("\nCHARACTERISTICS")

    for key, value in (
        summary[
            "top1_characteristics"
        ].items()
    ):
        print(
            f"  {key}: {value}"
        )

    print("\nSIMILARITY GAP")

    gap = summary[
        "similarity_gap"
    ]

    print(
        f"  < 0.01 : {gap['lt_0.01']}"
    )

    print(
        f"  < 0.03 : {gap['lt_0.03']}"
    )

    print(
        f"  < 0.05 : {gap['lt_0.05']}"
    )

    print(
        f"  average: {gap['average']}"
    )

    print("\nCASES")

    for x in results:

        top1 = x["top1"]
        gold = x["gold"]

        print(
            f"\n{x['id']} | "
            f"{x['category']}"
        )

        print(
            f"  Q: {x['question']}"
        )

        print(
            f"  TOP1: "
            f"{top1['source']} | "
            f"sim={top1['similarity']:.4f}"
        )

        print(
            f"  GOLD: "
            f"{gold['source']} | "
            f"rank={gold['rank']} | "
            f"sim={gold['similarity']:.4f}"
        )

        print(
            f"  gap={x['similarity_gap']:.6f}"
            if x["similarity_gap"] is not None
            else "  gap=None"
        )

        print(
            f"  lexical overlap: "
            f"{top1['lexical_overlap']} "
            f"vs "
            f"{gold['lexical_overlap']}"
        )

    print("\n" + "=" * 70)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

