import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

CHUNKS_FILE = Path("data/chunks.jsonl")

E23_FILE = Path(
    "evaluation/results/e23_conditional_section_details.json"
)

E25_FILE = Path(
    "evaluation/results/e25_section_rule_combinations_details.json"
)


# ============================================================
# Target chunks for q031
# ============================================================

TARGET_IDS = {
    "mcp-python-sdk--3",
    "mcp-python-sdk--4",
    "mcp-python-sdk-servers-resources-8",
}


# ============================================================
# E23 generic sections
# ============================================================

E23_GENERIC_SECTIONS = {
    "recap",
    "requirements",
    "create it",
    "try it",
    "example",
    "installation",
    "getting started",
    "introduction",
    "overview",
    "summary",
    "conclusion",
    "references",
    "contents",
    "table of contents",
}


# ============================================================
# Helpers
# ============================================================

def clean_section(section):
    if section is None:
        return ""

    return " ".join(str(section).split())


def section_quality(section, generic_sections):
    """
    Reproduce the E23/E25 section-quality logic.

    Returns:
        (is_informative, reason)
    """

    section = clean_section(section)

    if not section:
        return False, "empty section"

    lowered = section.lower()

    # Exact generic section
    if lowered in generic_sections:
        return False, "exact generic section"

    # <= 1 word
    if len(section.split()) <= 1:
        return False, "<= 1 word"

    # Breadcrumb final component
    parts = [
        part.strip()
        for part in section.split(">")
    ]

    final_component = parts[-1].strip().lower()

    if final_component in generic_sections:
        return False, "breadcrumb final component is generic"

    # Generic prefix
    for generic in generic_sections:
        if final_component.startswith(generic + " "):
            return False, f"generic prefix: {generic}"

    # Too long
    if len(section) > 300:
        return False, "> 300 chars"

    return True, "informative"


def build_title_content(chunk):
    title = chunk.get("title") or ""
    content = chunk.get("content") or ""

    return f"{title}\n\n{content}"


def build_title_section_content(chunk):
    title = chunk.get("title") or ""
    section = clean_section(chunk.get("section"))
    content = chunk.get("content") or ""

    return f"{title}\n\n{section}\n\n{content}"


def build_conditional(chunk, generic_sections):
    section = chunk.get("section")

    informative, reason = section_quality(
        section,
        generic_sections,
    )

    if informative:
        representation = build_title_section_content(chunk)
    else:
        representation = build_title_content(chunk)

    return representation, informative, reason


def load_chunks():
    chunks = {}

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            chunk_id = item.get("id")

            if chunk_id in TARGET_IDS:
                chunks[chunk_id] = item

    return chunks


def load_json(path):
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def find_q031(data):
    """
    Find q031 in different possible result structures.
    """

    if not data:
        return []

    results = []

    # E23 structure:
    #
    # {
    #   "representations": {
    #       "conditional": {
    #           "questions": [...]
    #       }
    #   }
    # }
    representations = data.get("representations")

    if isinstance(representations, dict):
        for representation_name, representation_data in representations.items():

            if not isinstance(representation_data, dict):
                continue

            questions = representation_data.get("questions", [])

            for question in questions:
                if question.get("id") == "q031":
                    results.append(
                        {
                            "representation": representation_name,
                            "data": question,
                        }
                    )

    # Generic fallback
    questions = data.get("questions")

    if isinstance(questions, list):
        for question in questions:
            if question.get("id") == "q031":
                results.append(
                    {
                        "representation": question.get(
                            "representation",
                            "unknown",
                        ),
                        "data": question,
                    }
                )

    return results


def build_rank_map(q031_results):
    rank_map = {}

    for result in q031_results:
        representation = result["representation"]
        question = result["data"]

        for item in question.get("top10", []):
            chunk_id = item.get("id")

            if chunk_id in TARGET_IDS:
                rank_map[
                    (
                        representation,
                        chunk_id,
                    )
                ] = {
                    "rank": item.get("rank"),
                    "similarity": item.get("similarity"),
                    "title": item.get("title"),
                    "section": item.get("section"),
                    "url": item.get("url"),
                }

    return rank_map


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("Q031 CHUNK-LEVEL AUDIT")
    print("=" * 80)

    print()
    print("Question:")
    print("How do I make an MCP resource?")

    print()
    print("Target chunks:")
    for chunk_id in sorted(TARGET_IDS):
        print(f"  - {chunk_id}")

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print()
    print("=" * 80)
    print("1. CHUNK DATA FROM data/chunks.jsonl")
    print("=" * 80)

    print(
        f"\nFound {len(chunks)} / {len(TARGET_IDS)} target chunks."
    )

    missing = TARGET_IDS - set(chunks.keys())

    if missing:
        print("\nMISSING:")
        for chunk_id in sorted(missing):
            print(f"  - {chunk_id}")

    # --------------------------------------------------------
    # Load E23
    # --------------------------------------------------------

    e23 = load_json(E23_FILE)
    e23_results = find_q031(e23)
    e23_rank_map = build_rank_map(e23_results)

    # --------------------------------------------------------
    # Load E25
    # --------------------------------------------------------

    e25 = load_json(E25_FILE)
    e25_results = find_q031(e25)
    e25_rank_map = build_rank_map(e25_results)

    # --------------------------------------------------------
    # Print every target chunk
    # --------------------------------------------------------

    for chunk_id in sorted(TARGET_IDS):

        chunk = chunks.get(chunk_id)

        print()
        print()
        print("#" * 80)
        print(f"CHUNK: {chunk_id}")
        print("#" * 80)

        if chunk is None:
            print("NOT FOUND")
            continue

        title = chunk.get("title") or ""
        section = clean_section(chunk.get("section"))
        url = chunk.get("url") or ""
        content = chunk.get("content") or ""

        print()
        print("TITLE")
        print("-" * 80)
        print(title)

        print()
        print("SECTION")
        print("-" * 80)
        print(section)

        print()
        print("URL")
        print("-" * 80)
        print(url)

        print()
        print("CONTENT")
        print("-" * 80)
        print(content)

        # ----------------------------------------------------
        # Section quality
        # ----------------------------------------------------

        informative, reason = section_quality(
            section,
            E23_GENERIC_SECTIONS,
        )

        print()
        print("E23 SECTION QUALITY")
        print("-" * 80)
        print(
            f"informative = {informative}"
        )
        print(
            f"reason      = {reason}"
        )

        # ----------------------------------------------------
        # Representations
        # ----------------------------------------------------

        title_content = build_title_content(chunk)

        title_section_content = build_title_section_content(
            chunk
        )

        conditional, conditional_informative, conditional_reason = (
            build_conditional(
                chunk,
                E23_GENERIC_SECTIONS,
            )
        )

        print()
        print("TITLE + CONTENT")
        print("-" * 80)
        print(title_content)

        print()
        print("TITLE + SECTION + CONTENT")
        print("-" * 80)
        print(title_section_content)

        print()
        print("E23 CONDITIONAL REPRESENTATION")
        print("-" * 80)
        print(conditional)

        # ----------------------------------------------------
        # E23 ranking information
        # ----------------------------------------------------

        print()
        print("E23 RANK / SIMILARITY")
        print("-" * 80)

        found_e23 = False

        for representation in [
            "title_content",
            "title_section_content",
            "conditional",
        ]:

            key = (
                representation,
                chunk_id,
            )

            info = e23_rank_map.get(key)

            if info:
                found_e23 = True

                print(
                    f"{representation:25s} "
                    f"rank={info['rank']} "
                    f"similarity={info['similarity']}"
                )

        if not found_e23:
            print("Not present in E23 Top10.")

        # ----------------------------------------------------
        # E25 ranking information
        # ----------------------------------------------------

        print()
        print("E25 RANK / SIMILARITY")
        print("-" * 80)

        found_e25 = False

        for representation in sorted(
            {
                key[0]
                for key in e25_rank_map.keys()
                if key[1] == chunk_id
            }
        ):

            key = (
                representation,
                chunk_id,
            )

            info = e25_rank_map.get(key)

            if info:
                found_e25 = True

                print(
                    f"{representation:25s} "
                    f"rank={info['rank']} "
                    f"similarity={info['similarity']}"
                )

        if not found_e25:
            print("No E25 ranking entry found.")

    # --------------------------------------------------------
    # Special comparison
    # --------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("2. KEY COMPARISON FOR q031")
    print("=" * 80)

    print()
    print("E23 generic sections:")
    print(
        "  recap, requirements, create it, try it, example, "
        "installation, ..."
    )

    print()
    print(
        "C7 generic sections:"
    )
    print(
        "  recap, installation, requirements"
    )

    print()
    print(
        "Therefore:"
    )

    for chunk_id in sorted(TARGET_IDS):

        chunk = chunks.get(chunk_id)

        if not chunk:
            continue

        section = clean_section(
            chunk.get("section")
        )

        e23_informative, e23_reason = section_quality(
            section,
            E23_GENERIC_SECTIONS,
        )

        c7_generic = {
            "recap",
            "installation",
            "requirements",
        }

        c7_informative, c7_reason = section_quality(
            section,
            c7_generic,
        )

        print()
        print(f"{chunk_id}")
        print(f"  section = {section}")
        print(
            f"  E23 -> informative={e23_informative}, "
            f"reason={e23_reason}"
        )
        print(
            f"  C7  -> informative={c7_informative}, "
            f"reason={c7_reason}"
        )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()