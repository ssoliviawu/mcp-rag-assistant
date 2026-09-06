import re


def clean_markdown(text: str) -> str:
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces
    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()