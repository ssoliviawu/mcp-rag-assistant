import re


HEADING_PATTERN = re.compile(
    r"^(#{1,3})\s+(.+?)\s*$",
    re.MULTILINE,
)


def chunk_markdown(
    text: str,
    max_chars: int = 3000,
) -> list[dict]:

    matches = list(
        HEADING_PATTERN.finditer(text)
    )

    sections = []

    heading_stack: list[str] = []

    for i, match in enumerate(matches):

        level = len(match.group(1))
        title = match.group(2).strip()

        # Remove headings from deeper levels
        heading_stack = heading_stack[: level - 1]

        heading_stack.append(title)

        section = " > ".join(heading_stack)

        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        content = text[start:end].strip()

        if content:
            sections.append(
                {
                    "title": title,
                    "section": section,
                    "content": content,
                }
            )

    chunks = []

    for section in sections:

        content = section["content"]

        if len(content) <= max_chars:

            chunks.append(section)

            continue

        paragraphs = re.split(
            r"\n\s*\n",
            content,
        )

        current = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            candidate = (
                f"{current}\n\n{paragraph}"
                if current
                else paragraph
            )

            if (
                current
                and len(candidate) > max_chars
            ):
                chunks.append(
                    {
                        "title": section["title"],
                        "section": section["section"],
                        "content": current,
                    }
                )

                current = paragraph

            else:
                current = candidate

        if current:
            chunks.append(
                {
                    "title": section["title"],
                    "section": section["section"],
                    "content": current,
                }
            )

    return chunks