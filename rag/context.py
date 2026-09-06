def build_context(
    results,
) -> str:

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        context_parts.append(
            f"""[Source {i}]
Title: {result["title"]}
Section: {result["section"]}
URL: {result["url"]}
Similarity: {result["similarity"]:.4f}

{result["content"]}
"""
        )

    return "\n\n".join(
        context_parts
    )