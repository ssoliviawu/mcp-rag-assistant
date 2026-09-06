import json
from pathlib import Path
import os
import psycopg


INPUT_FILE = Path("evaluation/results/error_analysis.json")
OUTPUT_FILE = Path("evaluation/results/low_rank_content_analysis.json")


# ============================================================
# DB CONFIG
# ============================================================

DB_URL = "postgresql://rag:rag@localhost:5432/mcp_rag"


def normalize_url(url):
    if not url:
        return ""
    return url.rstrip("/")


def get_connection():
    return psycopg.connect(DB_URL)


def get_chunk_content(conn, chunk_id):
    """
    Get chunk content directly from PostgreSQL.
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content
            FROM chunks
            WHERE id = %s
            """,
            (chunk_id,),
        )

        row = cur.fetchone()

    if row is None:
        return None

    return {
        "id": row[0],
        "content": row[1],
    }


def main():

    # ============================================================
    # 1. Load error_analysis.json
    # ============================================================

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    low_rank_hits = [
        item
        for item in results
        if item.get("error_type") == "LOW_RANK_HIT"
    ]

    print("=" * 100)
    print("LOW RANK HIT CONTENT ANALYSIS")
    print("=" * 100)
    print(f"Total LOW_RANK_HIT: {len(low_rank_hits)}")

    # ============================================================
    # 2. Connect DB
    # ============================================================

    conn = get_connection()

    output = []

    try:

        # ========================================================
        # 3. Analyze each question
        # ========================================================

        for item in low_rank_hits:

            qid = item["id"]
            question = item["question"]

            relevant_sources = {
                normalize_url(url)
                for url in item.get("relevant_sources", [])
            }

            details = item.get("retrieved_details", [])

            if not details:
                continue

            # ----------------------------------------------------
            # Top1
            # ----------------------------------------------------

            top1 = details[0]

            top1_id = top1.get("id")
            top1_source = normalize_url(top1.get("url"))
            top1_similarity = top1.get("similarity")

            # ----------------------------------------------------
            # Gold
            # ----------------------------------------------------

            gold_details = [
                d
                for d in details
                if normalize_url(d.get("url")) in relevant_sources
            ]

            gold_details.sort(key=lambda x: x["rank"])

            if gold_details:

                gold = gold_details[0]

                gold_id = gold.get("id")
                gold_source = normalize_url(gold.get("url"))
                gold_rank = gold.get("rank")
                gold_similarity = gold.get("similarity")

            else:

                gold = None
                gold_id = None
                gold_source = None
                gold_rank = None
                gold_similarity = None

            # ----------------------------------------------------
            # Load actual contents
            # ----------------------------------------------------

            top1_db = None
            gold_db = None

            if top1_id:
                top1_db = get_chunk_content(
                    conn,
                    top1_id,
                )

            if gold_id:
                gold_db = get_chunk_content(
                    conn,
                    gold_id,
                )

            top1_content = (
                top1_db["content"]
                if top1_db
                else None
            )

            gold_content = (
                gold_db["content"]
                if gold_db
                else None
            )

            # ----------------------------------------------------
            # Similarity gap
            # ----------------------------------------------------

            if (
                top1_similarity is not None
                and gold_similarity is not None
            ):
                similarity_gap = (
                    top1_similarity
                    - gold_similarity
                )
            else:
                similarity_gap = None

            # ----------------------------------------------------
            # Build result
            # ----------------------------------------------------

            result = {
                "id": qid,
                "question": question,

                "top1": {
                    "id": top1_id,
                    "rank": 1,
                    "source": top1_source,
                    "title": top1.get("title"),
                    "section": top1.get("section"),
                    "similarity": top1_similarity,
                    "content": top1_content,
                },

                "gold": {
                    "id": gold_id,
                    "rank": gold_rank,
                    "source": gold_source,
                    "title": (
                        gold.get("title")
                        if gold
                        else None
                    ),
                    "section": (
                        gold.get("section")
                        if gold
                        else None
                    ),
                    "similarity": gold_similarity,
                    "content": gold_content,
                },

                "similarity_gap": similarity_gap,

                "gold_in_top3": (
                    gold_rank is not None
                    and gold_rank <= 3
                ),
            }

            output.append(result)

            # ====================================================
            # Print
            # ====================================================

            print()
            print("=" * 100)
            print(qid)
            print("=" * 100)

            print()
            print("QUESTION:")
            print(question)

            print()
            print("-" * 100)
            print("TOP1")
            print("-" * 100)

            print(f"ID         : {top1_id}")
            print(f"Source     : {top1_source}")
            print(f"Title      : {top1.get('title')}")
            print(f"Section    : {top1.get('section')}")
            print(f"Similarity : {top1_similarity}")

            print()
            print("CONTENT:")
            print(top1_content or "[CONTENT NOT FOUND]")

            print()
            print("-" * 100)
            print("GOLD")
            print("-" * 100)

            print(f"ID         : {gold_id}")
            print(f"Rank       : {gold_rank}")
            print(f"Source     : {gold_source}")
            print(
                f"Title      : "
                f"{gold.get('title') if gold else None}"
            )
            print(
                f"Section    : "
                f"{gold.get('section') if gold else None}"
            )
            print(f"Similarity : {gold_similarity}")
            print(f"Gap        : {similarity_gap}")

            print()
            print("CONTENT:")
            print(gold_content or "[CONTENT NOT FOUND]")

    finally:
        conn.close()

    # ============================================================
    # 4. Save JSON
    # ============================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "total": len(output),
                "results": output,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 100)


if __name__ == "__main__":
    main()