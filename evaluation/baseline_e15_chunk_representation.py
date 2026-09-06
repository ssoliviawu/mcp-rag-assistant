import json
import statistics
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e13_vector_error_profile.json"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e15_chunk_representation.json"
)


# ============================================================
# Helpers
# ============================================================

def safe_mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else 0.0


def safe_median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else 0.0


def pct(part, total):
    return (part / total * 100) if total else 0.0


def rank_bucket(rank):
    if rank == 1:
        return "TOP1"
    if rank <= 3:
        return "TOP2_3"
    if rank <= 10:
        return "TOP4_10"
    if rank <= 20:
        return "TOP11_20"
    return "TOP21_100"


def source_type(url):
    if not url:
        return "unknown"

    url = url.lower()

    if "/migration" in url:
        return "migration"

    if "/troubleshooting" in url:
        return "troubleshooting"

    if "/advanced/low-level-server" in url:
        return "low_level"

    if "/servers/structured-output" in url:
        return "structured_output"

    if "/servers/" in url:
        return "servers"

    if "/deployment" in url:
        return "deployment"

    if "/run/" in url:
        return "run"

    if "/advanced/" in url:
        return "advanced"

    if "/handlers/" in url:
        return "handlers"

    if "/client/" in url:
        return "client"

    return "other"


def analyze_group(records):
    if not records:
        return {
            "questions": 0,
        }

    length_ratios = []
    word_ratios = []
    code_diffs = []
    semantic_gaps = []
    lexical_gaps = []

    top1_lengths = []
    gold_lengths = []

    top1_words = []
    gold_words = []

    top1_code = []
    gold_code = []

    for r in records:
        top1 = r["top1"]
        gold = r["best_gold"]
        comparison = r["comparison"]

        if gold["content_chars"] > 0:
            length_ratios.append(
                top1["content_chars"] / gold["content_chars"]
            )

        if gold["content_words"] > 0:
            word_ratios.append(
                top1["content_words"] / gold["content_words"]
            )

        code_diffs.append(
            top1["code_ratio"] - gold["code_ratio"]
        )

        semantic_gaps.append(
            comparison["semantic_gap_top1_minus_gold"]
        )

        lexical_gaps.append(
            comparison["lexical_gap_top1_minus_gold"]
        )

        top1_lengths.append(top1["content_chars"])
        gold_lengths.append(gold["content_chars"])

        top1_words.append(top1["content_words"])
        gold_words.append(gold["content_words"])

        top1_code.append(top1["code_ratio"])
        gold_code.append(gold["code_ratio"])

    return {
        "questions": len(records),

        "length": {
            "top1_avg_chars": safe_mean(top1_lengths),
            "gold_avg_chars": safe_mean(gold_lengths),
            "top1_median_chars": safe_median(top1_lengths),
            "gold_median_chars": safe_median(gold_lengths),
            "avg_length_ratio": safe_mean(length_ratios),
            "median_length_ratio": safe_median(length_ratios),
            "top1_longer_count": sum(
                1 for r in records
                if r["comparison"]["top1_longer_than_gold"]
            ),
        },

        "words": {
            "top1_avg_words": safe_mean(top1_words),
            "gold_avg_words": safe_mean(gold_words),
            "top1_median_words": safe_median(top1_words),
            "gold_median_words": safe_median(gold_words),
            "avg_word_ratio": safe_mean(word_ratios),
            "median_word_ratio": safe_median(word_ratios),
        },

        "code": {
            "top1_avg_code_ratio": safe_mean(top1_code),
            "gold_avg_code_ratio": safe_mean(gold_code),
            "avg_code_ratio_difference": safe_mean(code_diffs),
            "top1_code_heavy": sum(
                1 for r in records
                if r["top1"]["code_heavy"]
            ),
            "gold_code_heavy": sum(
                1 for r in records
                if r["best_gold"]["code_heavy"]
            ),
        },

        "ranking": {
            "avg_semantic_gap": safe_mean(semantic_gaps),
            "median_semantic_gap": safe_median(semantic_gaps),
            "avg_lexical_gap": safe_mean(lexical_gaps),
            "median_lexical_gap": safe_median(lexical_gaps),
        },
    }


# ============================================================
# Load E13
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"E13 result not found:\n{INPUT_PATH}"
    )

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = data["questions"]

print("=" * 80)
print("E15 - CHUNK REPRESENTATION ANALYSIS")
print("=" * 80)

print(f"Input     : {INPUT_PATH}")
print(f"Questions : {len(questions)}")
print()


# ============================================================
# Build records
# ============================================================

records = []

for item in questions:
    if not item.get("gold_found"):
        continue

    top1 = item["top1"]
    gold = item["best_gold"]

    record = {
        "id": item["id"],
        "question": item["question"],
        "gold_rank": item["gold_rank"],
        "rank_bucket": item["rank_bucket"],

        "top1_source_type": source_type(top1.get("url", "")),
        "gold_source_type": source_type(gold.get("url", "")),

        "top1": {
            "title": top1.get("title"),
            "section": top1.get("section"),
            "url": top1.get("url"),
            "similarity": top1.get("similarity"),
            "lexical_score": top1.get("lexical_score"),
            "content_chars": top1.get("content_chars", 0),
            "content_words": top1.get("content_words", 0),
            "code_ratio": top1.get("code_ratio", 0),
            "code_heavy": top1.get("code_heavy", False),
        },

        "best_gold": {
            "title": gold.get("title"),
            "section": gold.get("section"),
            "url": gold.get("url"),
            "similarity": gold.get("similarity"),
            "lexical_score": gold.get("lexical_score"),
            "content_chars": gold.get("content_chars", 0),
            "content_words": gold.get("content_words", 0),
            "code_ratio": gold.get("code_ratio", 0),
            "code_heavy": gold.get("code_heavy", False),
        },

        "comparison": item["comparison"],
    }

    records.append(record)


# ============================================================
# 1. Overall representation statistics
# ============================================================

overall = analyze_group(records)

print("1. OVERALL REPRESENTATION")
print("-" * 80)

length = overall["length"]
words = overall["words"]
code = overall["code"]
ranking = overall["ranking"]

print(
    f"Top1 avg chars        : {length['top1_avg_chars']:.1f}"
)
print(
    f"Gold avg chars        : {length['gold_avg_chars']:.1f}"
)
print(
    f"Top1 median chars     : {length['top1_median_chars']:.1f}"
)
print(
    f"Gold median chars     : {length['gold_median_chars']:.1f}"
)
print(
    f"Avg length ratio      : {length['avg_length_ratio']:.2f}x"
)
print(
    f"Median length ratio   : {length['median_length_ratio']:.2f}x"
)
print(
    f"Top1 longer than Gold : "
    f"{length['top1_longer_count']}/{len(records)} "
    f"({pct(length['top1_longer_count'], len(records)):.1f}%)"
)

print()

print(
    f"Top1 avg words        : {words['top1_avg_words']:.1f}"
)
print(
    f"Gold avg words        : {words['gold_avg_words']:.1f}"
)
print(
    f"Avg word ratio        : {words['avg_word_ratio']:.2f}x"
)
print(
    f"Median word ratio     : {words['median_word_ratio']:.2f}x"
)

print()

print(
    f"Top1 avg code ratio   : {code['top1_avg_code_ratio']:.3f}"
)
print(
    f"Gold avg code ratio   : {code['gold_avg_code_ratio']:.3f}"
)
print(
    f"Code ratio difference : "
    f"{code['avg_code_ratio_difference']:+.3f}"
)
print(
    f"Top1 code-heavy       : "
    f"{code['top1_code_heavy']}/{len(records)} "
    f"({pct(code['top1_code_heavy'], len(records)):.1f}%)"
)
print(
    f"Gold code-heavy       : "
    f"{code['gold_code_heavy']}/{len(records)} "
    f"({pct(code['gold_code_heavy'], len(records)):.1f}%)"
)

print()

print(
    f"Avg semantic gap      : {ranking['avg_semantic_gap']:.6f}"
)
print(
    f"Median semantic gap   : {ranking['median_semantic_gap']:.6f}"
)
print(
    f"Avg lexical gap       : {ranking['avg_lexical_gap']:.6f}"
)
print(
    f"Median lexical gap    : {ranking['median_lexical_gap']:.6f}"
)

print()


# ============================================================
# 2. Analyze by Gold rank bucket
# ============================================================

print("2. BY GOLD RANK BUCKET")
print("-" * 80)

buckets = [
    "TOP1",
    "TOP2_3",
    "TOP4_10",
    "TOP11_20",
]

bucket_results = {}

for bucket in buckets:
    group = [
        r for r in records
        if r["rank_bucket"] == bucket
    ]

    result = analyze_group(group)
    bucket_results[bucket] = result

    if not group:
        continue

    print(f"\n[{bucket}]")
    print(f"Questions             : {len(group)}")
    print(
        f"Avg Top1 chars        : "
        f"{result['length']['top1_avg_chars']:.1f}"
    )
    print(
        f"Avg Gold chars        : "
        f"{result['length']['gold_avg_chars']:.1f}"
    )
    print(
        f"Median length ratio   : "
        f"{result['length']['median_length_ratio']:.2f}x"
    )
    print(
        f"Top1 code-heavy       : "
        f"{result['code']['top1_code_heavy']}/{len(group)} "
        f"({pct(result['code']['top1_code_heavy'], len(group)):.1f}%)"
    )
    print(
        f"Gold code-heavy       : "
        f"{result['code']['gold_code_heavy']}/{len(group)} "
        f"({pct(result['code']['gold_code_heavy'], len(group)):.1f}%)"
    )
    print(
        f"Avg semantic gap      : "
        f"{result['ranking']['avg_semantic_gap']:.6f}"
    )
    print(
        f"Median semantic gap   : "
        f"{result['ranking']['median_semantic_gap']:.6f}"
    )


# ============================================================
# 3. Source type analysis
# ============================================================

print()
print("3. TOP1 SOURCE TYPE")
print("-" * 80)

source_groups = defaultdict(list)

for r in records:
    source_groups[r["top1_source_type"]].append(r)

source_results = {}

for source, group in sorted(
    source_groups.items(),
    key=lambda x: len(x[1]),
    reverse=True,
):
    result = analyze_group(group)
    source_results[source] = result

    print(
        f"{source:20s} "
        f"{len(group):2d} "
        f"({pct(len(group), len(records)):5.1f}%) "
        f"avg_len_ratio={result['length']['avg_length_ratio']:.2f}x "
        f"code_heavy={result['code']['top1_code_heavy']}/{len(group)} "
        f"sem_gap={result['ranking']['avg_semantic_gap']:.4f}"
    )


# ============================================================
# 4. Migration vs non-migration
# ============================================================

print()
print("4. MIGRATION VS NON-MIGRATION")
print("-" * 80)

migration_records = [
    r for r in records
    if r["top1_source_type"] == "migration"
]

non_migration_records = [
    r for r in records
    if r["top1_source_type"] != "migration"
]

migration_result = analyze_group(migration_records)
non_migration_result = analyze_group(non_migration_records)

for name, result in [
    ("MIGRATION", migration_result),
    ("NON-MIGRATION", non_migration_result),
]:
    if not result["questions"]:
        continue

    print(f"\n{name}")
    print(f"Questions             : {result['questions']}")
    print(
        f"Avg Top1 chars        : "
        f"{result['length']['top1_avg_chars']:.1f}"
    )
    print(
        f"Avg Gold chars        : "
        f"{result['length']['gold_avg_chars']:.1f}"
    )
    print(
        f"Median length ratio   : "
        f"{result['length']['median_length_ratio']:.2f}x"
    )
    print(
        f"Top1 code-heavy       : "
        f"{result['code']['top1_code_heavy']}/{result['questions']}"
    )
    print(
        f"Avg semantic gap      : "
        f"{result['ranking']['avg_semantic_gap']:.6f}"
    )


# ============================================================
# 5. Code-heavy vs non-code-heavy
# ============================================================

print()
print("5. CODE-HEAVY VS NON-CODE-HEAVY TOP1")
print("-" * 80)

code_heavy_records = [
    r for r in records
    if r["top1"]["code_heavy"]
]

non_code_records = [
    r for r in records
    if not r["top1"]["code_heavy"]
]

code_heavy_result = analyze_group(code_heavy_records)
non_code_result = analyze_group(non_code_records)

for name, result in [
    ("CODE-HEAVY TOP1", code_heavy_result),
    ("NON-CODE-HEAVY TOP1", non_code_result),
]:
    if not result["questions"]:
        continue

    print(f"\n{name}")
    print(f"Questions             : {result['questions']}")
    print(
        f"Avg semantic gap      : "
        f"{result['ranking']['avg_semantic_gap']:.6f}"
    )
    print(
        f"Median semantic gap   : "
        f"{result['ranking']['median_semantic_gap']:.6f}"
    )
    print(
        f"Avg length ratio      : "
        f"{result['length']['avg_length_ratio']:.2f}x"
    )
    print(
        f"Median length ratio   : "
        f"{result['length']['median_length_ratio']:.2f}x"
    )


# ============================================================
# 6. Deep dive: Gold rank 11-20
# ============================================================

print()
print("6. GOLD RANK 11-20 DEEP DIVE")
print("-" * 80)

rank_11_20 = [
    r for r in records
    if r["rank_bucket"] == "TOP11_20"
]

for r in sorted(rank_11_20, key=lambda x: x["gold_rank"]):

    top1 = r["top1"]
    gold = r["best_gold"]
    comparison = r["comparison"]

    print()
    print(
        f"{r['id']} | Gold rank={r['gold_rank']}"
    )
    print(f"Question : {r['question']}")
    print()
    print("TOP1")
    print(
        f"  source     : {r['top1_source_type']}"
    )
    print(
        f"  title      : {top1['title']}"
    )
    print(
        f"  section    : {top1['section']}"
    )
    print(
        f"  similarity : {top1['similarity']:.6f}"
    )
    print(
        f"  lexical    : {top1['lexical_score']:.6f}"
    )
    print(
        f"  chars      : {top1['content_chars']}"
    )
    print(
        f"  words      : {top1['content_words']}"
    )
    print(
        f"  code_ratio : {top1['code_ratio']:.3f}"
    )
    print(
        f"  code_heavy : {top1['code_heavy']}"
    )

    print()
    print("GOLD")
    print(
        f"  source     : {r['gold_source_type']}"
    )
    print(
        f"  title      : {gold['title']}"
    )
    print(
        f"  section    : {gold['section']}"
    )
    print(
        f"  similarity : {gold['similarity']:.6f}"
    )
    print(
        f"  lexical    : {gold['lexical_score']:.6f}"
    )
    print(
        f"  chars      : {gold['content_chars']}"
    )
    print(
        f"  words      : {gold['content_words']}"
    )
    print(
        f"  code_ratio : {gold['code_ratio']:.3f}"
    )
    print(
        f"  code_heavy : {gold['code_heavy']}"
    )

    print()
    print(
        f"Semantic gap : "
        f"{comparison['semantic_gap_top1_minus_gold']:.6f}"
    )
    print(
        f"Lexical gap  : "
        f"{comparison['lexical_gap_top1_minus_gold']:.6f}"
    )


# ============================================================
# 7. Largest length-ratio cases
# ============================================================

print()
print("7. TOP1 MUCH LONGER THAN GOLD")
print("-" * 80)

length_ranked = []

for r in records:
    gold_chars = r["best_gold"]["content_chars"]

    if gold_chars <= 0:
        continue

    ratio = (
        r["top1"]["content_chars"]
        / gold_chars
    )

    length_ranked.append(
        (ratio, r)
    )

length_ranked.sort(
    key=lambda x: x[0],
    reverse=True,
)

for ratio, r in length_ranked[:10]:
    print(
        f"{r['id']:5s} "
        f"ratio={ratio:7.2f}x "
        f"gold_rank={r['gold_rank']:2d} "
        f"top1={r['top1_source_type']:18s} "
        f"top1_chars={r['top1']['content_chars']:5d} "
        f"gold_chars={r['best_gold']['content_chars']:5d}"
    )


# ============================================================
# 8. Largest semantic-gap cases
# ============================================================

print()
print("8. LARGEST SEMANTIC GAP CASES")
print("-" * 80)

semantic_ranked = sorted(
    records,
    key=lambda r: r["comparison"]["semantic_gap_top1_minus_gold"],
    reverse=True,
)

for r in semantic_ranked[:10]:
    print(
        f"{r['id']:5s} "
        f"gap={r['comparison']['semantic_gap_top1_minus_gold']:.6f} "
        f"gold_rank={r['gold_rank']:2d} "
        f"top1={r['top1_source_type']:18s} "
        f"gold={r['gold_source_type']:18s}"
    )


# ============================================================
# 9. Specific hard cases
# ============================================================

print()
print("9. KNOWN HARD CASES")
print("-" * 80)

hard_case_ids = [
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

hard_cases = []

for r in records:
    if r["id"] in hard_case_ids:
        hard_cases.append(r)

for r in hard_cases:
    top1 = r["top1"]
    gold = r["best_gold"]
    comparison = r["comparison"]

    print()
    print(
        f"{r['id']} | Gold rank={r['gold_rank']} | "
        f"Top1={r['top1_source_type']}"
    )
    print(
        f"Question: {r['question']}"
    )
    print(
        f"Top1 chars={top1['content_chars']} | "
        f"Gold chars={gold['content_chars']} | "
        f"ratio="
        f"{top1['content_chars'] / gold['content_chars']:.2f}x"
        if gold["content_chars"] > 0
        else "ratio=N/A"
    )
    print(
        f"Top1 code={top1['code_ratio']:.3f} | "
        f"Gold code={gold['code_ratio']:.3f}"
    )
    print(
        f"Semantic gap="
        f"{comparison['semantic_gap_top1_minus_gold']:.6f} | "
        f"Lexical gap="
        f"{comparison['lexical_gap_top1_minus_gold']:.6f}"
    )


# ============================================================
# 10. Save result
# ============================================================

output = {
    "experiment": "E15_chunk_representation",
    "input": str(INPUT_PATH),
    "question_count": len(records),

    "overall": overall,

    "by_rank_bucket": bucket_results,

    "by_top1_source_type": source_results,

    "migration_vs_non_migration": {
        "migration": migration_result,
        "non_migration": non_migration_result,
    },

    "code_heavy_vs_non_code_heavy": {
        "code_heavy": code_heavy_result,
        "non_code_heavy": non_code_result,
    },

    "gold_rank_11_20": [
        {
            "id": r["id"],
            "question": r["question"],
            "gold_rank": r["gold_rank"],
            "top1_source_type": r["top1_source_type"],
            "gold_source_type": r["gold_source_type"],
            "top1": r["top1"],
            "best_gold": r["best_gold"],
            "comparison": r["comparison"],
        }
        for r in rank_11_20
    ],

    "largest_length_ratio_cases": [
        {
            "id": r["id"],
            "question": r["question"],
            "gold_rank": r["gold_rank"],
            "length_ratio": ratio,
            "top1": r["top1"],
            "best_gold": r["best_gold"],
            "comparison": r["comparison"],
        }
        for ratio, r in length_ranked[:20]
    ],

    "largest_semantic_gap_cases": [
        {
            "id": r["id"],
            "question": r["question"],
            "gold_rank": r["gold_rank"],
            "top1": r["top1"],
            "best_gold": r["best_gold"],
            "comparison": r["comparison"],
        }
        for r in semantic_ranked[:20]
    ],
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print("=" * 80)
print("E15 COMPLETE")
print("=" * 80)
print(f"Saved: {OUTPUT_PATH}")