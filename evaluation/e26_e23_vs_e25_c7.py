import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


# ----------------------------------------------------------------------
# File discovery
# ----------------------------------------------------------------------

def find_json_file(filename_candidates):
    for filename in filename_candidates:
        path = RESULTS_DIR / filename
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find any of:\n"
        + "\n".join(str(RESULTS_DIR / x) for x in filename_candidates)
    )


E23_FILE = find_json_file([
    "e23_conditional_section_details.json",
    "e23_conditional_details.json",
    "e23_section_details.json",
])

E25_FILE = find_json_file([
    "e25_section_rule_combinations_details.json",
])


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


e23_data = load_json(E23_FILE)
e25_data = load_json(E25_FILE)


# ----------------------------------------------------------------------
# Extract question records
# ----------------------------------------------------------------------

def extract_e23_questions(data):
    """
    E23 structure:

    representations
      conditional
        questions
    """

    questions = (
        data["representations"]
        ["conditional"]
        ["questions"]
    )

    return {
        item["id"]: item
        for item in questions
    }


def extract_e25_c7_questions(data):
    """
    E25 structure is expected to contain:

    C7_all_three
      -> questions

    Handles either:
      data["C7_all_three"]["questions"]
    or
      data["representations"]["C7_all_three"]["questions"]
    """

    if "C7_all_three" in data:
        obj = data["C7_all_three"]

    elif (
        "representations" in data
        and "C7_all_three" in data["representations"]
    ):
        obj = data["representations"]["C7_all_three"]

    else:
        raise KeyError(
            "Could not find C7_all_three in E25 details JSON."
        )

    if "questions" not in obj:
        raise KeyError(
            "C7_all_three does not contain a 'questions' field."
        )

    return {
        item["id"]: item
        for item in obj["questions"]
    }


e23_questions = extract_e23_questions(e23_data)
e25_questions = extract_e25_c7_questions(e25_data)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def get_gold_rank(item):
    """
    Support both:
      gold_rank
      first_relevant_rank
    """

    if "gold_rank" in item:
        return item["gold_rank"]

    if "first_relevant_rank" in item:
        return item["first_relevant_rank"]

    return None


def get_top10(item):
    return item.get("top10", [])


def get_top1(item):
    top10 = get_top10(item)

    if not top10:
        return None

    return top10[0]


def get_gold_sources(item):
    return set(item.get("gold_sources", []))


def is_gold_result(result, gold_sources):
    if result is None:
        return False

    return result.get("url") in gold_sources


def build_rank_map(top10):
    return {
        item.get("id"): item
        for item in top10
        if item.get("id") is not None
    }


def calculate_rank_delta(e23_rank, e25_rank):
    """
    Positive = improved
    Negative = worsened

    Example:
        E23 rank 5 -> E25 rank 1
        delta = +4
    """

    if e23_rank is None or e25_rank is None:
        return None

    return e23_rank - e25_rank


def classify_rank_change(e23_rank, e25_rank):
    if e23_rank is None and e25_rank is None:
        return "MISS_BOTH"

    if e23_rank is None and e25_rank is not None:
        return "NEW_HIT"

    if e23_rank is not None and e25_rank is None:
        return "NEW_MISS"

    if e25_rank < e23_rank:
        return "IMPROVED"

    if e25_rank > e23_rank:
        return "WORSENED"

    return "UNCHANGED"


# ----------------------------------------------------------------------
# Alignment sanity check
# ----------------------------------------------------------------------

e23_ids = set(e23_questions)
e25_ids = set(e25_questions)

common_ids = sorted(e23_ids & e25_ids)
only_e23 = sorted(e23_ids - e25_ids)
only_e25 = sorted(e25_ids - e23_ids)


print("=" * 100)
print("E26 - E23 CONDITIONAL vs E25 C7")
print("=" * 100)

print()
print("E23 file :", E23_FILE)
print("E25 file :", E25_FILE)
print()

print("E23 questions :", len(e23_ids))
print("E25 C7 questions :", len(e25_ids))
print("Common questions :", len(common_ids))

if only_e23:
    print("WARNING: only in E23:", only_e23)

if only_e25:
    print("WARNING: only in E25:", only_e25)

if len(common_ids) != 48:
    raise RuntimeError(
        f"Expected 48 common questions, got {len(common_ids)}"
    )


# ----------------------------------------------------------------------
# Per-question comparison
# ----------------------------------------------------------------------

details = []

improved = []
worsened = []
unchanged = []
new_hit = []
new_miss = []

top1_became_gold = []
top1_lost_gold = []
top1_changed = []

for qid in common_ids:

    e23 = e23_questions[qid]
    e25 = e25_questions[qid]

    question = e23.get(
        "question",
        e25.get("question", "")
    )

    e23_rank = get_gold_rank(e23)
    e25_rank = get_gold_rank(e25)

    rank_delta = calculate_rank_delta(
        e23_rank,
        e25_rank,
    )

    change_type = classify_rank_change(
        e23_rank,
        e25_rank,
    )

    e23_top1 = get_top1(e23)
    e25_top1 = get_top1(e25)

    gold_sources = get_gold_sources(e23)

    e23_top1_is_gold = is_gold_result(
        e23_top1,
        gold_sources,
    )

    e25_top1_is_gold = is_gold_result(
        e25_top1,
        gold_sources,
    )

    e23_top1_id = (
        e23_top1.get("id")
        if e23_top1
        else None
    )

    e25_top1_id = (
        e25_top1.get("id")
        if e25_top1
        else None
    )

    top1_changed_flag = (
        e23_top1_id != e25_top1_id
    )

    if change_type == "IMPROVED":
        improved.append(qid)

    elif change_type == "WORSENED":
        worsened.append(qid)

    elif change_type == "UNCHANGED":
        unchanged.append(qid)

    elif change_type == "NEW_HIT":
        new_hit.append(qid)

    elif change_type == "NEW_MISS":
        new_miss.append(qid)

    if not e23_top1_is_gold and e25_top1_is_gold:
        top1_became_gold.append(qid)

    if e23_top1_is_gold and not e25_top1_is_gold:
        top1_lost_gold.append(qid)

    if top1_changed_flag:
        top1_changed.append(qid)

    # Find E23/C7 score for the gold item if present in top10.
    e23_gold_result = None
    e25_gold_result = None

    for result in get_top10(e23):
        if result.get("url") in gold_sources:
            e23_gold_result = result
            break

    for result in get_top10(e25):
        if result.get("url") in gold_sources:
            e25_gold_result = result
            break

    e23_gold_similarity = (
        e23_gold_result.get("similarity")
        if e23_gold_result
        else None
    )

    e25_gold_similarity = (
        e25_gold_result.get("similarity")
        if e25_gold_result
        else None
    )

    similarity_delta = None

    if (
        e23_gold_similarity is not None
        and e25_gold_similarity is not None
    ):
        similarity_delta = (
            e25_gold_similarity
            - e23_gold_similarity
        )

    details.append({
        "id": qid,
        "question": question,

        "e23": {
            "gold_rank": e23_rank,
            "gold_similarity": e23_gold_similarity,
            "top1": e23_top1,
            "top1_is_gold": e23_top1_is_gold,
        },

        "e25_c7": {
            "gold_rank": e25_rank,
            "gold_similarity": e25_gold_similarity,
            "top1": e25_top1,
            "top1_is_gold": e25_top1_is_gold,
        },

        "comparison": {
            "rank_delta": rank_delta,
            "change": change_type,
            "similarity_delta": similarity_delta,
            "top1_changed": top1_changed_flag,
            "top1_became_gold": (
                not e23_top1_is_gold
                and e25_top1_is_gold
            ),
            "top1_lost_gold": (
                e23_top1_is_gold
                and not e25_top1_is_gold
            ),
        },

        "gold_sources": sorted(gold_sources),
    })


# ----------------------------------------------------------------------
# Sort diagnostic views
# ----------------------------------------------------------------------

improved_details = sorted(
    [
        x for x in details
        if x["comparison"]["change"] == "IMPROVED"
    ],
    key=lambda x: (
        -(x["comparison"]["rank_delta"] or 0)
    ),
)

worsened_details = sorted(
    [
        x for x in details
        if x["comparison"]["change"] == "WORSENED"
    ],
    key=lambda x: (
        x["comparison"]["rank_delta"] or 0
    ),
)

top1_changes = [
    x for x in details
    if x["comparison"]["top1_changed"]
]


# ----------------------------------------------------------------------
# Recalculate metrics from gold rank
# ----------------------------------------------------------------------

def calculate_metrics(records):
    ranks = [
        x["e25_c7"]["gold_rank"]
        for x in records
    ]

    n = len(records)

    def hit_at(k):
        return sum(
            1
            for rank in ranks
            if rank is not None and rank <= k
        ) / n

    def mrr():
        total = 0.0

        for rank in ranks:
            if rank is not None:
                total += 1.0 / rank

        return total / n

    return {
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@10": hit_at(10),
        "mrr": mrr(),
    }


def calculate_e23_metrics(records):
    ranks = [
        x["e23"]["gold_rank"]
        for x in records
    ]

    n = len(records)

    def hit_at(k):
        return sum(
            1
            for rank in ranks
            if rank is not None and rank <= k
        ) / n

    total_mrr = sum(
        1.0 / rank
        for rank in ranks
        if rank is not None
    )

    return {
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@10": hit_at(10),
        "mrr": total_mrr / n,
    }


e23_recalculated = calculate_e23_metrics(details)
e25_recalculated = calculate_metrics(details)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

summary = {
    "experiment": "E26_e23_conditional_vs_e25_c7",

    "question_count": len(details),

    "e23": {
        "experiment": "E23_conditional_section",
        "representation": "conditional",
        "metrics_recalculated": e23_recalculated,
    },

    "e25": {
        "experiment": "E25_section_rule_combinations",
        "combination": "C7_all_three",
        "rules": [
            "recap",
            "installation",
            "requirements",
        ],
        "metrics_recalculated": e25_recalculated,
    },

    "rank_change": {
        "improved": len(improved),
        "worsened": len(worsened),
        "unchanged": len(unchanged),
        "new_hit": len(new_hit),
        "new_miss": len(new_miss),
    },

    "top1_change": {
        "changed": len(top1_changed),
        "became_gold": len(top1_became_gold),
        "lost_gold": len(top1_lost_gold),
    },

    "delta": {
        "hit@1": (
            e25_recalculated["hit@1"]
            - e23_recalculated["hit@1"]
        ),
        "hit@3": (
            e25_recalculated["hit@3"]
            - e23_recalculated["hit@3"]
        ),
        "hit@5": (
            e25_recalculated["hit@5"]
            - e23_recalculated["hit@5"]
        ),
        "hit@10": (
            e25_recalculated["hit@10"]
            - e23_recalculated["hit@10"]
        ),
        "mrr": (
            e25_recalculated["mrr"]
            - e23_recalculated["mrr"]
        ),
    },

    "diagnostic": {
        "improved_questions": improved,
        "worsened_questions": worsened,
        "top1_became_gold_questions": top1_became_gold,
        "top1_lost_gold_questions": top1_lost_gold,
    },
}


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

summary_path = (
    RESULTS_DIR
    / "e26_e23_vs_e25_c7_summary.json"
)

details_path = (
    RESULTS_DIR
    / "e26_e23_vs_e25_c7_details.json"
)

with summary_path.open("w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )

with details_path.open("w", encoding="utf-8") as f:
    json.dump(
        details,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ----------------------------------------------------------------------
# Console output
# ----------------------------------------------------------------------

print()
print("=" * 100)
print("E26 SUMMARY")
print("=" * 100)

print()
print("E23 Conditional:")
for key, value in e23_recalculated.items():
    print(f"  {key:<10}: {value:.4f}")

print()
print("E25 C7:")
for key, value in e25_recalculated.items():
    print(f"  {key:<10}: {value:.4f}")

print()
print("Delta: E25 C7 - E23 Conditional")
for key, value in summary["delta"].items():
    print(f"  {key:<10}: {value:+.4f}")

print()
print("Rank changes")
print(f"  Improved  : {len(improved)}")
print(f"  Worsened  : {len(worsened)}")
print(f"  Unchanged : {len(unchanged)}")
print(f"  New hit   : {len(new_hit)}")
print(f"  New miss  : {len(new_miss)}")

print()
print("Top1 changes")
print(f"  Changed      : {len(top1_changed)}")
print(f"  Became gold  : {len(top1_became_gold)}")
print(f"  Lost gold    : {len(top1_lost_gold)}")

print()
print("=" * 100)
print("TOP IMPROVEMENTS")
print("=" * 100)

for item in improved_details[:15]:
    print()
    print(
        f"{item['id']} | "
        f"rank {item['e23']['gold_rank']} "
        f"-> {item['e25_c7']['gold_rank']} "
        f"(delta {item['comparison']['rank_delta']:+d})"
    )
    print(f"Question: {item['question']}")
    print(
        f"E23 Top1: "
        f"{item['e23']['top1']['title'] if item['e23']['top1'] else None}"
    )
    print(
        f"C7  Top1: "
        f"{item['e25_c7']['top1']['title'] if item['e25_c7']['top1'] else None}"
    )

print()
print("=" * 100)
print("TOP REGRESSIONS")
print("=" * 100)

for item in worsened_details[:15]:
    print()
    print(
        f"{item['id']} | "
        f"rank {item['e23']['gold_rank']} "
        f"-> {item['e25_c7']['gold_rank']} "
        f"(delta {item['comparison']['rank_delta']:+d})"
    )
    print(f"Question: {item['question']}")
    print(
        f"E23 Top1: "
        f"{item['e23']['top1']['title'] if item['e23']['top1'] else None}"
    )
    print(
        f"C7  Top1: "
        f"{item['e25_c7']['top1']['title'] if item['e25_c7']['top1'] else None}"
    )

print()
print("=" * 100)
print("TOP1 BECAME GOLD")
print("=" * 100)

for item in details:
    if item["comparison"]["top1_became_gold"]:
        print()
        print(item["id"])
        print(item["question"])
        print(
            f"E23: rank={item['e23']['gold_rank']}, "
            f"Top1={item['e23']['top1']['title']}"
        )
        print(
            f"C7 : rank={item['e25_c7']['gold_rank']}, "
            f"Top1={item['e25_c7']['top1']['title']}"
        )

print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)
print(summary_path)
print(details_path)