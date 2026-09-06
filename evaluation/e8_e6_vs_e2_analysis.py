import json
from pathlib import Path
from collections import Counter


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

E2_DETAILS_PATH = RESULTS_DIR / "retrieval_matrix_details.json"
E6_DETAILS_PATH = RESULTS_DIR / "e6_lexical_alpha_details.json"

OUTPUT_PATH = RESULTS_DIR / "e8_e6_vs_e2_analysis.json"

E2_NAME = "E2"
E6_NAME = "E6_05"
ALPHA = 0.05


# ============================================================
# JSON
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Helpers
# ============================================================

def normalize_source(value):
    if not value:
        return ""

    return str(value).rstrip("/")


def get_question_id(item):
    return item.get("id")


def get_question(item):
    return item.get("question", "")


def get_gold_sources(item):
    return {
        normalize_source(x)
        for x in item.get("relevant_sources", [])
        if x
    }


# ============================================================
# E2
# ============================================================

def extract_e2_results(item):
    """
    E2 details only store retrieved_sources.

    Example:

    "retrieved_sources": [
        "url1",
        "url2",
        ...
    ]
    """

    sources = item.get("retrieved_sources", [])

    results = []

    for rank, source in enumerate(
        sources,
        start=1,
    ):
        results.append(
            {
                "rank": rank,
                "url": normalize_source(source),
                "id": normalize_source(source),
                "title": None,
                "similarity": None,
                "lexical_score": None,
                "final_score": None,
            }
        )

    return results


# ============================================================
# E6
# ============================================================

def extract_e6_results(item):
    """
    E6 stores detailed retrieval results.
    """

    for key in (
        "retrieved_details",
        "retrieved",
        "results",
        "top10",
    ):

        value = item.get(key)

        if isinstance(value, list):
            return value

    return []


def get_e6_source(result):
    return normalize_source(
        result.get("url")
        or result.get("source")
        or result.get("source_url")
    )


def get_result_id(result):
    if result.get("id") is not None:
        return str(result["id"])

    return "|".join(
        [
            get_e6_source(result),
            str(result.get("title", "")),
            str(result.get("section", "")),
        ]
    )


def get_similarity(result):
    value = result.get("similarity")

    if value is None:
        value = result.get("score")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_lexical(result):
    value = result.get("lexical_score")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_final_score(result):
    value = result.get("final_score")

    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    similarity = get_similarity(result)
    lexical = get_lexical(result)

    if similarity is None:
        return None

    if lexical is None:
        return similarity

    return similarity + ALPHA * lexical


# ============================================================
# Rank helpers
# ============================================================

def get_first_gold_rank(results, gold_sources):
    for result in results:

        source = normalize_source(
            result.get("url", "")
        )

        if source in gold_sources:
            return result["rank"]

    return None


def get_gold_results(results, gold_sources):

    output = []

    for result in results:

        source = normalize_source(
            result.get("url", "")
        )

        if source not in gold_sources:
            continue

        output.append(
            {
                "rank": result.get("rank"),
                "id": result.get("id"),
                "title": result.get("title"),
                "url": source,
                "similarity": result.get(
                    "similarity"
                ),
                "lexical_score": result.get(
                    "lexical_score"
                ),
                "final_score": result.get(
                    "final_score"
                ),
            }
        )

    return output


# ============================================================
# Candidate set
# ============================================================

def get_candidate_ids(results):

    return {
        normalize_source(
            result.get("url", "")
        )
        for result in results
        if result.get("url")
    }


def compare_candidate_sets(
    e2_results,
    e6_results,
):

    e2_sources = get_candidate_ids(
        e2_results
    )

    e6_sources = get_candidate_ids(
        e6_results
    )

    return {
        "same": e2_sources == e6_sources,
        "e2_count": len(e2_sources),
        "e6_count": len(e6_sources),
        "entered": sorted(
            e6_sources - e2_sources
        ),
        "removed": sorted(
            e2_sources - e6_sources
        ),
    }


# ============================================================
# Lexical causal analysis
# ============================================================

def find_e6_gold(
    e6_results,
    gold_sources,
):

    for result in e6_results:

        if (
            get_e6_source(result)
            in gold_sources
        ):
            return result

    return None


def find_e6_top_non_gold(
    e6_results,
    gold_sources,
):

    for result in e6_results:

        if (
            get_e6_source(result)
            not in gold_sources
        ):
            return result

    return None


def classify_case(
    e2_rank,
    e6_rank,
    e2_results,
    e6_results,
    gold_sources,
):

    e2_hit = (
        e2_rank is not None
        and e2_rank <= 10
    )

    e6_hit = (
        e6_rank is not None
        and e6_rank <= 10
    )

    # ========================================================
    # E2 miss -> E6 hit
    # ========================================================

    if not e2_hit and e6_hit:

        gold = find_e6_gold(
            e6_results,
            gold_sources,
        )

        if gold is None:
            return "NEW_HIT_UNKNOWN"

        gold_similarity = get_similarity(
            gold
        )

        gold_lexical = get_lexical(
            gold
        )

        competitor = find_e6_top_non_gold(
            e6_results,
            gold_sources,
        )

        if competitor is not None:

            competitor_similarity = (
                get_similarity(
                    competitor
                )
            )

            competitor_lexical = (
                get_lexical(
                    competitor
                )
            )

            if (
                gold_similarity is not None
                and gold_lexical is not None
                and competitor_similarity is not None
                and competitor_lexical is not None
            ):

                gold_final = (
                    gold_similarity
                    + ALPHA * gold_lexical
                )

                competitor_final = (
                    competitor_similarity
                    + ALPHA
                    * competitor_lexical
                )

                # True lexical reversal:
                #
                # Vector says competitor > gold
                # E6 says gold > competitor

                if (
                    competitor_similarity
                    > gold_similarity
                    and gold_final
                    > competitor_final
                ):
                    return (
                        "TRUE_LEXICAL_RESCUE"
                    )

        return "RANK_CHANGE_RESCUE"

    # ========================================================
    # E2 hit -> E6 miss
    # ========================================================

    if e2_hit and not e6_hit:
        return "LEXICAL_REGRESSION"

    # ========================================================
    # Both hit
    # ========================================================

    if e2_hit and e6_hit:

        if e6_rank < e2_rank:
            return "GOLD_RANK_IMPROVED"

        if e6_rank > e2_rank:
            return "GOLD_RANK_WORSENED"

        return "UNCHANGED"

    # ========================================================
    # Both miss
    # ========================================================

    return "BOTH_MISS"


# ============================================================
# Per-question analysis
# ============================================================

def analyze_question(
    e2_item,
    e6_item,
):

    question_id = get_question_id(
        e2_item
    )

    question = get_question(
        e2_item
    )

    gold_sources = (
        get_gold_sources(e2_item)
        or get_gold_sources(e6_item)
    )

    e2_results = extract_e2_results(
        e2_item
    )

    e6_results = extract_e6_results(
        e6_item
    )

    e2_gold_rank = get_first_gold_rank(
        e2_results,
        gold_sources,
    )

    e6_gold_rank = get_first_gold_rank(
        e6_results,
        gold_sources,
    )

    e2_hit = (
        e2_gold_rank is not None
        and e2_gold_rank <= 10
    )

    e6_hit = (
        e6_gold_rank is not None
        and e6_gold_rank <= 10
    )

    candidate_changes = (
        compare_candidate_sets(
            e2_results,
            e6_results,
        )
    )

    classification = classify_case(
        e2_gold_rank,
        e6_gold_rank,
        e2_results,
        e6_results,
        gold_sources,
    )

    # --------------------------------------------------------
    # Rank change
    # --------------------------------------------------------

    rank_change = None

    if (
        e2_gold_rank is not None
        and e6_gold_rank is not None
    ):

        rank_change = (
            e2_gold_rank
            - e6_gold_rank
        )

    # --------------------------------------------------------
    # E2 top1
    # --------------------------------------------------------

    e2_top1 = None

    if e2_results:

        result = e2_results[0]

        e2_top1 = {
            "rank": 1,
            "url": result["url"],
            "is_gold": (
                result["url"]
                in gold_sources
            ),
        }

    # --------------------------------------------------------
    # E6 top1
    # --------------------------------------------------------

    e6_top1 = None

    if e6_results:

        result = e6_results[0]

        e6_top1 = {
            "rank": result.get(
                "rank",
                1,
            ),
            "id": get_result_id(
                result
            ),
            "title": result.get(
                "title"
            ),
            "url": get_e6_source(
                result
            ),
            "similarity": get_similarity(
                result
            ),
            "lexical_score": get_lexical(
                result
            ),
            "final_score": get_final_score(
                result
            ),
            "is_gold": (
                get_e6_source(result)
                in gold_sources
            ),
        }

    return {
        "id": question_id,
        "question": question,

        "gold_sources": sorted(
            gold_sources
        ),

        "e2": {
            "gold_rank": e2_gold_rank,
            "hit@10": e2_hit,
            "top1": e2_top1,
            "gold_results": get_gold_results(
                e2_results,
                gold_sources,
            ),
        },

        "e6": {
            "gold_rank": e6_gold_rank,
            "hit@10": e6_hit,
            "top1": e6_top1,
            "gold_results": get_gold_results(
                e6_results,
                gold_sources,
            ),
        },

        "rank_change": rank_change,

        "candidate_changes": {
            "same_candidate_set": (
                candidate_changes["same"]
            ),
            "e2_count": (
                candidate_changes["e2_count"]
            ),
            "e6_count": (
                candidate_changes["e6_count"]
            ),
            "entered": (
                candidate_changes["entered"]
            ),
            "removed": (
                candidate_changes["removed"]
            ),
        },

        "classification": classification,
    }


# ============================================================
# Summary
# ============================================================

def build_summary(analyses):

    total = len(analyses)

    e2_hits = sum(
        item["e2"]["hit@10"]
        for item in analyses
    )

    e6_hits = sum(
        item["e6"]["hit@10"]
        for item in analyses
    )

    new_hits = sum(
        (
            not item["e2"]["hit@10"]
            and item["e6"]["hit@10"]
        )
        for item in analyses
    )

    lost_hits = sum(
        (
            item["e2"]["hit@10"]
            and not item["e6"]["hit@10"]
        )
        for item in analyses
    )

    both_hit = sum(
        (
            item["e2"]["hit@10"]
            and item["e6"]["hit@10"]
        )
        for item in analyses
    )

    both_miss = sum(
        (
            not item["e2"]["hit@10"]
            and not item["e6"]["hit@10"]
        )
        for item in analyses
    )

    classifications = Counter(
        item["classification"]
        for item in analyses
    )

    improved = []
    worsened = []

    for item in analyses:

        change = item["rank_change"]

        if change is None:
            continue

        if change > 0:
            improved.append(change)

        elif change < 0:
            worsened.append(
                abs(change)
            )

    same_candidate_set = sum(
        item["candidate_changes"][
            "same_candidate_set"
        ]
        for item in analyses
    )

    return {
        "questions": total,

        "e2": {
            "hits@10": e2_hits,
            "misses@10": (
                total - e2_hits
            ),
            "hit@10": (
                e2_hits / total
                if total
                else 0
            ),
        },

        "e6": {
            "hits@10": e6_hits,
            "misses@10": (
                total - e6_hits
            ),
            "hit@10": (
                e6_hits / total
                if total
                else 0
            ),
        },

        "delta": {
            "additional_hits": (
                e6_hits - e2_hits
            ),
            "hit@10": (
                (e6_hits - e2_hits)
                / total
                if total
                else 0
            ),
        },

        "transition": {
            "new_hits": new_hits,
            "lost_hits": lost_hits,
            "both_hit": both_hit,
            "both_miss": both_miss,
        },

        "rank_change": {
            "improved_count": len(
                improved
            ),
            "worsened_count": len(
                worsened
            ),
            "avg_improvement": (
                sum(improved)
                / len(improved)
                if improved
                else 0
            ),
            "avg_worsening": (
                sum(worsened)
                / len(worsened)
                if worsened
                else 0
            ),
        },

        "classification": dict(
            classifications
        ),

        "candidate_set": {
            "same": same_candidate_set,
            "different": (
                total
                - same_candidate_set
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E8 - E2 vs E6_05")
    print("=" * 80)

    e2_data = load_json(
        E2_DETAILS_PATH
    )

    e6_data = load_json(
        E6_DETAILS_PATH
    )

    # ========================================================
    # E2
    #
    # IMPORTANT:
    # Only E2 is selected.
    # E0/E1/E3a/E3/E4/E5 are ignored.
    # ========================================================

    if "results" not in e2_data:
        raise ValueError(
            "retrieval_matrix_details.json "
            "has no 'results'"
        )

    if E2_NAME not in e2_data["results"]:
        raise ValueError(
            f"{E2_NAME} not found"
        )

    e2_questions = e2_data[
        "results"
    ][E2_NAME]

    if not isinstance(
        e2_questions,
        list,
    ):
        raise ValueError(
            "E2 is not a list"
        )

    # ========================================================
    # E6
    # ========================================================

    if isinstance(
        e6_data,
        dict,
    ):

        if (
            "results" in e6_data
            and isinstance(
                e6_data["results"],
                dict,
            )
            and E6_NAME
            in e6_data["results"]
        ):

            e6_questions = (
                e6_data["results"][
                    E6_NAME
                ]
            )

        elif (
            "results" in e6_data
            and isinstance(
                e6_data["results"],
                list,
            )
        ):

            e6_questions = (
                e6_data["results"]
            )

        else:

            # Most likely structure:
            # {"E6_02": [...],
            #  "E6_05": [...],
            #  "E6_10": [...]}

            if (
                E6_NAME
                in e6_data
            ):

                e6_questions = (
                    e6_data[E6_NAME]
                )

            else:
                raise ValueError(
                    "Cannot locate E6_05"
                )

    elif isinstance(
        e6_data,
        list,
    ):

        e6_questions = e6_data

    else:

        raise ValueError(
            "Unsupported E6 JSON structure"
        )

    # ========================================================
    # Maps
    # ========================================================

    e2_map = {
        item["id"]: item
        for item in e2_questions
        if item.get("id")
    }

    e6_map = {
        item["id"]: item
        for item in e6_questions
        if item.get("id")
    }

    common_ids = sorted(
        set(e2_map)
        & set(e6_map)
    )

    print()
    print(
        f"E2 questions : "
        f"{len(e2_map)}"
    )

    print(
        f"E6 questions : "
        f"{len(e6_map)}"
    )

    print(
        f"Common       : "
        f"{len(common_ids)}"
    )

    if not common_ids:
        raise ValueError(
            "No common question IDs"
        )

    # ========================================================
    # Analyze
    # ========================================================

    analyses = []

    for question_id in common_ids:

        analyses.append(
            analyze_question(
                e2_map[question_id],
                e6_map[question_id],
            )
        )

    summary = build_summary(
        analyses
    )

    # ========================================================
    # Save
    # ========================================================

    output = {
        "experiment": "E8_E2_vs_E6",

        "e2_source": str(
            E2_DETAILS_PATH
        ),

        "e2_experiment": E2_NAME,

        "e6_source": str(
            E6_DETAILS_PATH
        ),

        "e6_experiment": E6_NAME,

        "alpha": ALPHA,

        "summary": summary,

        "questions": analyses,
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

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"E2 Hit@10 : "
        f"{summary['e2']['hit@10']:.4f} "
        f"({summary['e2']['hits@10']}/"
        f"{summary['questions']})"
    )

    print(
        f"E6 Hit@10 : "
        f"{summary['e6']['hit@10']:.4f} "
        f"({summary['e6']['hits@10']}/"
        f"{summary['questions']})"
    )

    print(
        f"Delta     : "
        f"{summary['delta']['hit@10']:+.4f} "
        f"({summary['delta']['additional_hits']:+d})"
    )

    print()
    print("Transition")
    print("-" * 40)

    print(
        f"E2 miss -> E6 hit : "
        f"{summary['transition']['new_hits']}"
    )

    print(
        f"E2 hit  -> E6 miss: "
        f"{summary['transition']['lost_hits']}"
    )

    print(
        f"Both hit           : "
        f"{summary['transition']['both_hit']}"
    )

    print(
        f"Both miss          : "
        f"{summary['transition']['both_miss']}"
    )

    print()
    print("Rank change")
    print("-" * 40)

    print(
        f"Improved : "
        f"{summary['rank_change']['improved_count']}"
    )

    print(
        f"Worsened : "
        f"{summary['rank_change']['worsened_count']}"
    )

    print()
    print("Classification")
    print("-" * 40)

    for category, count in sorted(
        summary["classification"].items(),
        key=lambda x: (-x[1], x[0]),
    ):

        print(
            f"{category:<28} {count}"
        )

    print()
    print("Candidate set")
    print("-" * 40)

    print(
        f"Same candidate set : "
        f"{summary['candidate_set']['same']}/"
        f"{summary['questions']}"
    )

    print(
        f"Different          : "
        f"{summary['candidate_set']['different']}/"
        f"{summary['questions']}"
    )

    print()
    print("=" * 80)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()