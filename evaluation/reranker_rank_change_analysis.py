from pathlib import Path
import re
from collections import Counter


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR /"results" / "reranker_overall.txt"

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "reranker_rank_change_analysis.txt"
)


# ============================================================
# Helpers
# ============================================================

def parse_rank(value):
    value = value.strip().upper()

    if value == "MISS":
        return None

    return int(value)


def calculate_category(vector_rank, reranker_rank):
    """
    Rank change convention:

        reranker - vector

    Negative = improved
    Positive = worsened
    Zero     = unchanged
    """

    if vector_rank is None and reranker_rank is None:
        return "BOTH_MISS"

    if vector_rank is None:
        return "VECTOR_MISS"

    if reranker_rank is None:
        return "RERANKER_MISS"

    change = reranker_rank - vector_rank

    if change < 0:
        return "IMPROVED"

    if change > 0:
        return "WORSENED"

    return "UNCHANGED"


# ============================================================
# Parsing
# ============================================================

def parse_questions(text: str):

    questions = []

    current = None

    lines = text.splitlines()

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        # ====================================================
        # FORMAT 1
        #
        # [49/50] q049: What changed from MCP Python SDK v1 to v2?
        #
        # ====================================================

        match = re.match(
            r"^\[\d+/\d+\]\s*(q\d+)\s*:\s*(.+)$",
            line,
            re.IGNORECASE,
        )

        if match:

            if current is not None:
                questions.append(current)

            current = {
                "id": match.group(1).lower(),
                "question": match.group(2).strip(),
                "vector_rank": None,
                "reranker_rank": None,
                "rank_change": None,
                "category": None,
            }

            continue

        # ====================================================
        # FORMAT 2
        #
        # q049 | What changed from MCP Python SDK v1 to v2?
        #
        # ====================================================

        match = re.match(
            r"^(q\d+)\s*\|\s*(.+)$",
            line,
            re.IGNORECASE,
        )

        if match:

            if current is not None:
                questions.append(current)

            current = {
                "id": match.group(1).lower(),
                "question": match.group(2).strip(),
                "vector_rank": None,
                "reranker_rank": None,
                "rank_change": None,
                "category": None,
            }

            continue

        # ====================================================
        # No question yet
        # ====================================================

        if current is None:
            continue

        # ====================================================
        # FORMAT 1 rank line
        #
        # UNCHANGED Vector=1 Reranker=1
        #
        # Also supports:
        #
        # WORSENED Vector=6 Reranker=7
        # IMPROVED Vector=7 Reranker=3
        # ====================================================

        vector_match = re.search(
            r"\bVector\s*=\s*(MISS|\d+)\b",
            line,
            re.IGNORECASE,
        )

        reranker_match = re.search(
            r"\bReranker\s*=\s*(MISS|\d+)\b",
            line,
            re.IGNORECASE,
        )

        if vector_match:

            current["vector_rank"] = parse_rank(
                vector_match.group(1)
            )

        if reranker_match:

            current["reranker_rank"] = parse_rank(
                reranker_match.group(1)
            )

        # ====================================================
        # FORMAT 2 rank lines
        #
        # Vector rank    : 1
        # Reranker rank  : 1
        # Rank change    : 0
        # ====================================================

        vector_match_old = re.search(
            r"\bVector\s+rank\s*:\s*(MISS|\d+)\b",
            line,
            re.IGNORECASE,
        )

        reranker_match_old = re.search(
            r"\bReranker\s+rank\s*:\s*(MISS|\d+)\b",
            line,
            re.IGNORECASE,
        )

        change_match_old = re.search(
            r"\bRank\s+change\s*:\s*([+-]?\d+)",
            line,
            re.IGNORECASE,
        )

        if vector_match_old:

            current["vector_rank"] = parse_rank(
                vector_match_old.group(1)
            )

        if reranker_match_old:

            current["reranker_rank"] = parse_rank(
                reranker_match_old.group(1)
            )

        if change_match_old:

            current["rank_change"] = int(
                change_match_old.group(1)
            )

        # ====================================================
        # Once both ranks are available, calculate category
        # ====================================================

        vector_rank = current["vector_rank"]
        reranker_rank = current["reranker_rank"]

        if (
            vector_rank is not None
            or reranker_rank is not None
        ):

            current["category"] = calculate_category(
                vector_rank,
                reranker_rank,
            )

            # If both ranks are available, always calculate
            # rank change ourselves.
            if (
                vector_rank is not None
                and reranker_rank is not None
            ):

                current["rank_change"] = (
                    reranker_rank - vector_rank
                )

        # ====================================================
        # If both are MISS
        # ====================================================

        elif (
            vector_rank is None
            and reranker_rank is None
        ):

            # Only classify as BOTH_MISS if the line actually
            # contains rank information.
            if (
                "Vector" in line
                or "Reranker" in line
            ):
                current["category"] = "BOTH_MISS"

    # ========================================================
    # Save final question
    # ========================================================

    if current is not None:
        questions.append(current)

    return questions


# ============================================================
# Statistics
# ============================================================

def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def analyze(questions):

    return {
        "total": len(questions),

        "improved": [
            q for q in questions
            if q["category"] == "IMPROVED"
        ],

        "worsened": [
            q for q in questions
            if q["category"] == "WORSENED"
        ],

        "unchanged": [
            q for q in questions
            if q["category"] == "UNCHANGED"
        ],

        "vector_miss": [
            q for q in questions
            if q["category"] == "VECTOR_MISS"
        ],

        "reranker_miss": [
            q for q in questions
            if q["category"] == "RERANKER_MISS"
        ],

        "both_miss": [
            q for q in questions
            if q["category"] == "BOTH_MISS"
        ],
    }


# ============================================================
# Output helpers
# ============================================================

def write_question_list(
    f,
    title,
    questions,
    reverse=False,
):
    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n")

    if not questions:
        f.write("None.\n")
        return

    sorted_questions = sorted(
        questions,
        key=lambda q: (
            q["rank_change"]
            if q["rank_change"] is not None
            else 0
        ),
        reverse=reverse,
    )

    for q in sorted_questions:

        vector_display = (
            q["vector_rank"]
            if q["vector_rank"] is not None
            else "MISS"
        )

        reranker_display = (
            q["reranker_rank"]
            if q["reranker_rank"] is not None
            else "MISS"
        )

        f.write(
            f"\n{q['id']} | {q['question']}\n"
        )

        f.write(
            f"Vector rank   : {vector_display}\n"
        )

        f.write(
            f"Reranker rank : {reranker_display}\n"
        )

        if q["rank_change"] is not None:
            f.write(
                f"Rank change   : "
                f"{q['rank_change']:+d}\n"
            )

        f.write(
            f"Category      : {q['category']}\n"
        )

# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("RERANKER RANK CHANGE ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    print(f"Input file : {INPUT_FILE}")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Read file
    #
    # Try UTF-16 first because your previous file appears
    # to use UTF-16.
    # --------------------------------------------------------

    try:

        text = INPUT_FILE.read_text(
            encoding="utf-16"
        )

    except UnicodeError:

        print(
            "UTF-16 read failed, trying UTF-8..."
        )

        text = INPUT_FILE.read_text(
            encoding="utf-8"
        )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    questions = parse_questions(text)

    print(
        f"Total lines     : "
        f"{len(text.splitlines())}"
    )

    print(
        f"Parsed questions: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    print()
    print("First parsed records:")
    print("-" * 80)

    for q in questions[:10]:

        print(
            f"{q['id']} | "
            f"V={q['vector_rank']} "
            f"R={q['reranker_rank']} "
            f"change={q['rank_change']} "
            f"{q['category']} | "
            f"{q['question']}"
        )

    print("-" * 80)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not questions:

        print()
        print("ERROR: No questions were parsed.")
        print()
        print("First 30 raw lines:")
        print("-" * 80)

        for i, line in enumerate(
            text.splitlines()[:30],
            start=1,
        ):

            print(
                f"{i:03d}: {repr(line)}"
            )

        print("-" * 80)

        raise RuntimeError(
            "No question records were parsed."
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = analyze(questions)

    improved = stats["improved"]
    worsened = stats["worsened"]
    unchanged = stats["unchanged"]
    vector_miss = stats["vector_miss"]
    reranker_miss = stats["reranker_miss"]
    both_miss = stats["both_miss"]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    most_improved = sorted(
        improved,
        key=lambda q: q["rank_change"],
    )

    most_worsened = sorted(
        worsened,
        key=lambda q: q["rank_change"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Valid rank changes
    # --------------------------------------------------------

    valid = [
        q["rank_change"]
        for q in questions
        if q["rank_change"] is not None
    ]

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    distribution = Counter(valid)

    # --------------------------------------------------------
    # Top 1
    # --------------------------------------------------------

    vector_top1 = [
        q for q in questions
        if q["vector_rank"] == 1
    ]

    reranker_top1 = [
        q for q in questions
        if q["reranker_rank"] == 1
    ]

    lost_top1 = [
        q for q in vector_top1
        if q["reranker_rank"] != 1
    ]

    gained_top1 = [
        q for q in reranker_top1
        if q["vector_rank"] != 1
    ]

    # --------------------------------------------------------
    # Suspicious
    # --------------------------------------------------------

    suspicious = [
        q for q in questions
        if (
            q["rank_change"] is not None
            and q["rank_change"] >= 3
        )
    ]

    # --------------------------------------------------------
    # Q013
    # --------------------------------------------------------

    q013 = next(
        (
            q
            for q in questions
            if q["id"] == "q013"
        ),
        None,
    )

    # ========================================================
    # Write output
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        f.write("=" * 80 + "\n")
        f.write("RERANKER RANK CHANGE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        f.write("OVERALL\n")
        f.write("-" * 80 + "\n")

        f.write(
            f"Total questions : "
            f"{len(questions)}\n"
        )

        f.write(
            f"Improved        : "
            f"{len(improved)}\n"
        )

        f.write(
            f"Unchanged       : "
            f"{len(unchanged)}\n"
        )

        f.write(
            f"Worsened        : "
            f"{len(worsened)}\n"
        )

        f.write(
            f"Vector miss     : "
            f"{len(vector_miss)}\n"
        )

        f.write(
            f"Reranker miss   : "
            f"{len(reranker_miss)}\n"
        )

        f.write(
            f"Both miss       : "
            f"{len(both_miss)}\n"
        )

        # ----------------------------------------------------
        # Rank change
        # ----------------------------------------------------

        f.write("\n")

        f.write(
            f"Average rank change : "
            f"{average(valid):+.3f}\n"
        )

        f.write(
            "Negative = improved\n"
        )

        f.write(
            "Positive = worsened\n"
        )

        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        f.write("\n")
        f.write("RANK CHANGE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")

        for change in sorted(distribution):

            f.write(
                f"{change:+d} : "
                f"{distribution[change]}\n"
            )

        # ----------------------------------------------------
        # Most improved
        # ----------------------------------------------------

        write_question_list(
            f,
            "MOST IMPROVED",
            most_improved[:10],
        )

        # ----------------------------------------------------
        # Most worsened
        # ----------------------------------------------------

        write_question_list(
            f,
            "MOST WORSENED",
            most_worsened[:15],
            reverse=True,
        )

        # ----------------------------------------------------
        # Large improvements
        # ----------------------------------------------------

        large_improvements = [
            q for q in improved
            if q["rank_change"] <= -3
        ]

        write_question_list(
            f,
            "LARGE IMPROVEMENTS (>= 3 RANKS)",
            large_improvements,
        )

        # ----------------------------------------------------
        # Large degradations
        # ----------------------------------------------------

        large_degradations = [
            q for q in worsened
            if q["rank_change"] >= 3
        ]

        write_question_list(
            f,
            "LARGE DEGRADATIONS (>= 3 RANKS)",
            large_degradations,
            reverse=True,
        )

        # ====================================================
        # TOP-1
        # ====================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("TOP-1 EFFECT\n")
        f.write("=" * 80 + "\n")

        f.write(
            f"Vector top-1 questions   : "
            f"{len(vector_top1)}\n"
        )

        f.write(
            f"Reranker top-1 questions : "
            f"{len(reranker_top1)}\n"
        )

        f.write(
            f"Lost vector top-1        : "
            f"{len(lost_top1)}\n"
        )

        f.write(
            f"Gained reranker top-1   : "
            f"{len(gained_top1)}\n"
        )

        if lost_top1:

            f.write("\n")
            f.write("QUESTIONS THAT LOST TOP-1\n")
            f.write("-" * 80 + "\n")

            for q in sorted(
                lost_top1,
                key=lambda x: x["rank_change"],
                reverse=True,
            ):

                f.write(
                    f"{q['id']} | "
                    f"V={q['vector_rank']} "
                    f"R={q['reranker_rank']} "
                    f"change={q['rank_change']:+d} | "
                    f"{q['question']}\n"
                )

        if gained_top1:

            f.write("\n")
            f.write("QUESTIONS THAT GAINED TOP-1\n")
            f.write("-" * 80 + "\n")

            for q in sorted(
                gained_top1,
                key=lambda x: x["rank_change"],
            ):

                f.write(
                    f"{q['id']} | "
                    f"V={q['vector_rank']} "
                    f"R={q['reranker_rank']} "
                    f"change={q['rank_change']:+d} | "
                    f"{q['question']}\n"
                )

        # ====================================================
        # Suspicious
        # ====================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SUSPICIOUS CASES\n")
        f.write("=" * 80 + "\n")

        f.write(
            f"Questions worsened by >=3 ranks: "
            f"{len(suspicious)}\n"
        )

        for q in sorted(
            suspicious,
            key=lambda x: x["rank_change"],
            reverse=True,
        ):

            f.write("\n")

            f.write(
                f"{q['id']}\n"
            )

            f.write(
                f"Question      : "
                f"{q['question']}\n"
            )

            f.write(
                f"Vector rank   : "
                f"{q['vector_rank']}\n"
            )

            f.write(
                f"Reranker rank : "
                f"{q['reranker_rank']}\n"
            )

            f.write(
                f"Rank change   : "
                f"{q['rank_change']:+d}\n"
            )

        # ====================================================
        # Q013
        # ====================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Q013 CHECK\n")
        f.write("=" * 80 + "\n")

        if q013:

            f.write(
                f"Question      : "
                f"{q013['question']}\n"
            )

            f.write(
                f"Vector rank   : "
                f"{q013['vector_rank']}\n"
            )

            f.write(
                f"Reranker rank : "
                f"{q013['reranker_rank']}\n"
            )

            f.write(
                f"Rank change   : "
                f"{q013['rank_change']:+d}\n"
                if q013["rank_change"] is not None
                else "Rank change   : N/A\n"
            )

            f.write(
                f"Category      : "
                f"{q013['category']}\n"
            )

        else:

            f.write(
                "q013 not found.\n"
            )

        # ====================================================
        # Final interpretation
        # ====================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("FINAL INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        if len(worsened) > len(improved):

            f.write(
                "1. Reranker currently worsens more "
                "questions than it improves.\n"
            )

        elif len(improved) > len(worsened):

            f.write(
                "1. Reranker improves more questions "
                "than it worsens.\n"
            )

        else:

            f.write(
                "1. Reranker has a balanced "
                "improve/worsen distribution.\n"
            )

        f.write(
            f"2. Improved: {len(improved)}, "
            f"Worsened: {len(worsened)}, "
            f"Unchanged: {len(unchanged)}.\n"
        )

        f.write(
            f"3. Average rank change: "
            f"{average(valid):+.3f}.\n"
        )

        if len(lost_top1) > len(gained_top1):

            f.write(
                "4. Reranker loses more vector "
                "top-1 results than it gains.\n"
            )

        elif len(gained_top1) > len(lost_top1):

            f.write(
                "4. Reranker gains more top-1 results "
                "than it loses.\n"
            )

        else:

            f.write(
                "4. Reranker has balanced top-1 "
                "gains and losses.\n"
            )

        if suspicious:

            f.write(
                f"5. There are {len(suspicious)} "
                f"large degradation cases (>=3 ranks). "
                f"These should be investigated individually.\n"
            )

        else:

            f.write(
                "5. No large degradation cases detected.\n"
            )

        f.write("\n")

        f.write(
            "IMPORTANT:\n"
            "This analysis only measures rank movement.\n"
            "A reranker moving a relevant document down is "
            "a retrieval-quality regression even if the "
            "document remains inside top-10.\n"
        )

        f.write("\n")

        f.write(
            "NEXT STEP:\n"
            "Inspect the large degradation cases and compare "
            "their query/document characteristics against "
            "successful reranker cases.\n"
        )

    # ========================================================
    # Done
    # ========================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Analysis written to:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print(
        f"Improved  : {len(improved)}"
    )

    print(
        f"Unchanged : {len(unchanged)}"
    )

    print(
        f"Worsened  : {len(worsened)}"
    )

    print(
        f"Vector miss   : {len(vector_miss)}"
    )

    print(
        f"Reranker miss : {len(reranker_miss)}"
    )

    print(
        f"Average change: "
        f"{average(valid):+.3f}"
    )


if __name__ == "__main__":
    main()