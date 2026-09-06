from pathlib import Path
from urllib.parse import urlparse

import requests


def download_markdown(
    url: str,
    output_dir: Path,
) -> Path:
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "mcp-rag-assistant/0.1"
        },
    )
    response.raise_for_status()

    parsed = urlparse(url)

    # /get-started/index.md
    # → get-started/index.md
    relative_path = parsed.path.lstrip("/")

    output_path = output_dir / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        response.text,
        encoding="utf-8",
    )

    return output_path