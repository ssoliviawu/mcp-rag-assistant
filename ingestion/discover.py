import re
from pathlib import Path

import requests

from ingestion.download import download_markdown


def download_index(url: str) -> str:
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "mcp-rag-assistant/0.1"
        },
    )

    response.raise_for_status()

    return response.text


def extract_urls(text: str) -> list[str]:
    urls = re.findall(
        r"\[[^\]]+\]\((https?://[^)]+)\)",
        text,
    )

    return list(dict.fromkeys(urls))


if __name__ == "__main__":

    index_url = (
        "https://py.sdk.modelcontextprotocol.io/llms.txt"
    )

    text = download_index(index_url)

    urls = extract_urls(text)

    print(f"Found {len(urls)} URLs")

    output_dir = Path(
        "data/raw/mcp-python-sdk"
    )

    for i, url in enumerate(urls, start=1):

        try:
            path = download_markdown(
                url,
                output_dir,
            )

            print(
                f"[{i}/{len(urls)}] "
                f"Downloaded {path}"
            )

        except requests.RequestException as e:

            print(
                f"[{i}/{len(urls)}] "
                f"FAILED {url}: {e}"
            )