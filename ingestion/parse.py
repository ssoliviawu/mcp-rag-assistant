from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify


def parse_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main")

    if main is None:
        main = soup.body

    if main is None:
        raise ValueError(f"No main content found in {path}")

    return markdownify(str(main))

if __name__ == "__main__":
    path = Path("data/raw/mcp-python-sdk.html")

    markdown = parse_html(path)

    output = Path("data/mcp-python-sdk.md")

    output.write_text(
        markdown,
        encoding="utf-8",
    )

    print(f"Saved to {output}")