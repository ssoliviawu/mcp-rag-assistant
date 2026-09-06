import json
import re
from pathlib import Path

from db.repository import search_chunks
from embedding.model import BGEEmbeddingModel


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

E13_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e13_vector_error_profile.json"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e16_chunk_content_analysis.json"
)


# ============================================================
# Configuration
# ============================================================

FOCUS_IDS = [
    "q010",
    "q021",
    "q048",
    "q003",
    "q011",
    "q013",
    "q019",
    "q045",
    "q050",
]


# ============================================================
# Helpers
# ============================================================

def normalize_url(url):
    if not url:
        return ""

    url = str(url).strip()

    if url.endswith("/"):
        url = url[:-1]

    return url


def tokenize(text):
    if not text:
        return []

    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        str(text).lower(),
    )


def lexical_overlap(query, text):
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))

    if not query_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


def safe_text(value):
    if value is None:
        return ""

    return str(value)


def make_representations(chunk):
    title = safe_text(chunk.get("title"))
    section = safe_text(chunk.get("section"))
    content = safe_text(chunk.get("content"))

    return {
        "content_only": content,

        "title_content": (
            f"{title}\n\n{content}"
            if title
            else content
        ),

        "section_content": (
            f"{section}\n\n{content}"
            if section
            else content
        ),

        "title_section_content": (
            f"{title}\n\n{section}\n\n{content}"
            if title or section
            else content
        ),
    }


def representation_stats(query, chunk):
    representations = make_representations(chunk)

    result = {}

    for name, text in representations.items():
        result[name] = {
            "chars": len(text),
            "words": len(text.split()),
            "lexical_overlap": lexical_overlap(
                query,
                text,
            ),
        }

    return result


def find_chunk_by_identity(target, candidates):
    """
    Try to identify the exact DB chunk from E13 metadata.

    Priority:
    1. chunk id
    2. URL + title + section
    3. URL + title
    4. URL + section
    """

    target_id = target.get("id")
    target_url = normalize_url(target.get("url"))
    target_title = safe_text(target.get("title"))
    target_section = safe_text(target.get("section"))

    # --------------------------------------------------------
    # 1. ID
    # --------------------------------------------------------

    if target_id is not None:
        for chunk in candidates:
            if str(chunk.get("id")) == str(target_id):
                return chunk, "id"

    # --------------------------------------------------------
    # 2. URL + title + section
    # --------------------------------------------------------

    for chunk in candidates:
        if (
            normalize_url(chunk.get("url")) == target_url
            and safe_text(chunk.get("title")) == target_title
            and safe_text(chunk.get("section")) == target_section
        ):
            return chunk, "url+title+section"

    # --------------------------------------------------------
    # 3. URL + title
    # --------------------------------------------------------

    for chunk in candidates:
        if (
            normalize_url(chunk.get("url")) == target_url
            and safe_text(chunk.get("title")) == target_title
        ):
            return chunk, "url+title"

    # --------------------------------------------------------
    # 4. URL + section
    # --------------------------------------------------------

    for chunk in candidates:
        if (
            normalize_url(chunk.get("url")) == target_url
            and safe_text(chunk.get("section")) == target_section
        ):
            return chunk, "url+section"

    return None, None


def print_chunk(label, chunk, query):
    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    content = safe_text(chunk.get("content"))

    print(f"ID      : {chunk.get('id')}")
    print(f"Title   : {chunk.get('title')}")
    print(f"Section : {chunk.get('section')}")
    print(f"URL     : {chunk.get('url')}")
    print(f"Chars   : {len(content)}")
    print(f"Words   : {len(content.split())}")

    print()
    print("CONTENT")
    print("-" * 80)
    print(content)

    stats = representation_stats(
        query,
        chunk,
    )

    print()
    print("REPRESENTATION STATS")
    print("-" * 80)

    for name, values in stats.items():
        print(
            f"{name:24s} "
            f"chars={values['chars']:5d} "
            f"words={values['words']:4d} "
            f"lexical={values['lexical_overlap']:.3f}"
        )


# ============================================================
# Load E13
# ============================================================

if not E13_PATH.exists():
    raise FileNotFoundError(
        f"E13 result not found:\n{E13_PATH}"
    )

with open(
    E13_PATH,
    "r",
    encoding="utf-8",
) as f:
    e13 = json.load(f)

questions = e13["questions"]

question_map = {
    item["id"]: item
    for item in questions
}


# ============================================================
# Initialize embedding model ONCE
# ============================================================

model = BGEEmbeddingModel()


# ============================================================
# Main analysis
# ============================================================

print("=" * 80)
print("E16 - REAL CHUNK CONTENT ANALYSIS")
print("=" * 80)

print(f"E13 input : {E13_PATH}")
print(f"Questions : {len(questions)}")
print(f"Focus     : {len(FOCUS_IDS)}")
print()


results = []


for qid in FOCUS_IDS:

    if qid not in question_map:
        print(f"[SKIP] {qid} not found in E13")
        continue

    item = question_map[qid]

    query = item["question"]

    e13_top1 = item["top1"]
    e13_gold = item["best_gold"]

    print()
    print("#" * 80)
    print(f"# {qid}")
    print("#" * 80)

    print(f"Question : {query}")
    print(f"Gold rank: {item['gold_rank']}")

    print(
        f"Semantic gap: "
        f"{item['comparison']['semantic_gap_top1_minus_gold']:.6f}"
    )

    print(
        f"Lexical gap: "
        f"{item['comparison']['lexical_gap_top1_minus_gold']:.6f}"
    )

    # --------------------------------------------------------
    # Retrieve same Vector Top100 candidates
    # --------------------------------------------------------

    try:
        query_embedding = model.embed_query(query)

        candidates = search_chunks(
            query_embedding,
            limit=100,
        )

    except Exception as e:
        print()
        print(
            f"[ERROR] DB retrieval failed for {qid}: {e}"
        )
        continue

    # --------------------------------------------------------
    # Match E13 Top1 / Gold to real DB chunks
    # --------------------------------------------------------

    top1_chunk, top1_match = find_chunk_by_identity(
        e13_top1,
        candidates,
    )

    gold_chunk, gold_match = find_chunk_by_identity(
        e13_gold,
        candidates,
    )

    print()
    print(
        f"Top1 identity match : "
        f"{top1_match or 'NOT FOUND'}"
    )

    print(
        f"Gold identity match : "
        f"{gold_match or 'NOT FOUND'}"
    )

    if top1_chunk is None:
        print(
            "[WARNING] Top1 chunk not found in DB"
        )
        continue

    if gold_chunk is None:
        print(
            "[WARNING] Gold chunk not found in DB"
        )
        continue

    # --------------------------------------------------------
    # Print real content
    # --------------------------------------------------------

    print_chunk(
        f"{qid} - TOP1 CHUNK",
        top1_chunk,
        query,
    )

    print_chunk(
        f"{qid} - GOLD CHUNK",
        gold_chunk,
        query,
    )

    # --------------------------------------------------------
    # Representation comparison
    # --------------------------------------------------------

    top1_repr = representation_stats(
        query,
        top1_chunk,
    )

    gold_repr = representation_stats(
        query,
        gold_chunk,
    )

    representation_delta = {}

    for name in top1_repr:
        representation_delta[name] = {
            "char_delta_top1_minus_gold": (
                top1_repr[name]["chars"]
                - gold_repr[name]["chars"]
            ),
            "word_delta_top1_minus_gold": (
                top1_repr[name]["words"]
                - gold_repr[name]["words"]
            ),
            "lexical_delta_top1_minus_gold": (
                top1_repr[name]["lexical_overlap"]
                - gold_repr[name]["lexical_overlap"]
            ),
        }

    results.append(
        {
            "id": qid,
            "question": query,
            "gold_rank": item["gold_rank"],

            "e13": {
                "top1": e13_top1,
                "best_gold": e13_gold,
                "comparison": item["comparison"],
            },

            "top1_identity_match": top1_match,
            "gold_identity_match": gold_match,

            "top1_chunk": {
                "id": top1_chunk.get("id"),
                "title": top1_chunk.get("title"),
                "section": top1_chunk.get("section"),
                "url": top1_chunk.get("url"),
                "content": top1_chunk.get("content"),
                "chars": len(
                    safe_text(
                        top1_chunk.get("content")
                    )
                ),
                "words": len(
                    safe_text(
                        top1_chunk.get("content")
                    ).split()
                ),
            },

            "gold_chunk": {
                "id": gold_chunk.get("id"),
                "title": gold_chunk.get("title"),
                "section": gold_chunk.get("section"),
                "url": gold_chunk.get("url"),
                "content": gold_chunk.get("content"),
                "chars": len(
                    safe_text(
                        gold_chunk.get("content")
                    )
                ),
                "words": len(
                    safe_text(
                        gold_chunk.get("content")
                    ).split()
                ),
            },

            "representation_stats": {
                "top1": top1_repr,
                "gold": gold_repr,
                "delta": representation_delta,
            },
        }
    )


# ============================================================
# Aggregate analysis
# ============================================================

print()
print("=" * 80)
print("E16 - AGGREGATE REPRESENTATION ANALYSIS")
print("=" * 80)


if results:

    representation_names = [
        "content_only",
        "title_content",
        "section_content",
        "title_section_content",
    ]

    for name in representation_names:

        top1_chars = []
        gold_chars = []

        top1_words = []
        gold_words = []

        top1_lexical = []
        gold_lexical = []

        for result in results:

            top1_stats = (
                result["representation_stats"]["top1"][name]
            )

            gold_stats = (
                result["representation_stats"]["gold"][name]
            )

            top1_chars.append(
                top1_stats["chars"]
            )

            gold_chars.append(
                gold_stats["chars"]
            )

            top1_words.append(
                top1_stats["words"]
            )

            gold_words.append(
                gold_stats["words"]
            )

            top1_lexical.append(
                top1_stats["lexical_overlap"]
            )

            gold_lexical.append(
                gold_stats["lexical_overlap"]
            )

        avg_top1_chars = (
            sum(top1_chars)
            / len(top1_chars)
        )

        avg_gold_chars = (
            sum(gold_chars)
            / len(gold_chars)
        )

        avg_top1_words = (
            sum(top1_words)
            / len(top1_words)
        )

        avg_gold_words = (
            sum(gold_words)
            / len(gold_words)
        )

        avg_top1_lexical = (
            sum(top1_lexical)
            / len(top1_lexical)
        )

        avg_gold_lexical = (
            sum(gold_lexical)
            / len(gold_lexical)
        )

        print()
        print(f"[{name}]")

        print(
            f"Avg chars   : "
            f"Top1={avg_top1_chars:.1f} "
            f"Gold={avg_gold_chars:.1f}"
        )

        print(
            f"Avg words   : "
            f"Top1={avg_top1_words:.1f} "
            f"Gold={avg_gold_words:.1f}"
        )

        print(
            f"Avg lexical : "
            f"Top1={avg_top1_lexical:.3f} "
            f"Gold={avg_gold_lexical:.3f}"
        )


# ============================================================
# Title / Section availability
# ============================================================

print()
print("=" * 80)
print("E16 - TITLE / SECTION CONTRIBUTION")
print("=" * 80)

title_nonempty = 0
section_nonempty = 0

for result in results:

    top1 = result["top1_chunk"]

    if safe_text(top1["title"]).strip():
        title_nonempty += 1

    if safe_text(top1["section"]).strip():
        section_nonempty += 1

print(
    f"Top1 chunks with title   : "
    f"{title_nonempty}/{len(results)}"
)

print(
    f"Top1 chunks with section : "
    f"{section_nonempty}/{len(results)}"
)


# ============================================================
# Migration cases
# ============================================================

print()
print("=" * 80)
print("E16 - MIGRATION CASES")
print("=" * 80)

migration_results = [
    result
    for result in results
    if "/migration"
    in normalize_url(
        result["top1_chunk"]["url"]
    ).lower()
]

print(
    f"Migration Top1 cases : "
    f"{len(migration_results)}"
)

for result in migration_results:

    top1 = result["top1_chunk"]
    gold = result["gold_chunk"]

    print()
    print(
        f"{result['id']} | "
        f"Gold rank={result['gold_rank']}"
    )

    print(
        f"TOP1 title   : {top1['title']}"
    )

    print(
        f"TOP1 section : {top1['section']}"
    )

    print(
        f"GOLD title   : {gold['title']}"
    )

    print(
        f"GOLD section : {gold['section']}"
    )


# ============================================================
# Save
# ============================================================

output = {
    "experiment": "E16_real_chunk_content_analysis",
    "input": str(E13_PATH),
    "question_count": len(results),
    "focus_ids": FOCUS_IDS,
    "results": results,
}


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


print()
print("=" * 80)
print("E16 COMPLETE")
print("=" * 80)

print(
    f"Saved: {OUTPUT_PATH}"
)