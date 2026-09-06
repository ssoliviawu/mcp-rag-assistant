
from retrieval.search import vector_search
from retrieval.reranker import rerank


QUERY = "What happens when an MCP tool returns a primitive type?"

TARGET = (
    "https://py.sdk.modelcontextprotocol.io/"
    "servers/structured-output/"
)


def normalize_url(url):
    if not url:
        return ""
    return url.rstrip("/")


print("=" * 80)
print("Q013 RERANK DEBUG")
print("=" * 80)


# ============================================================
# 1. Vector
# ============================================================

vector_results = vector_search(
    QUERY,
    limit=50,
)

print("\n[1] VECTOR")

target_vector = None

for rank, result in enumerate(
    vector_results,
    start=1,
):

    if normalize_url(result.get("url")) == normalize_url(TARGET):

        if target_vector is None:
            target_vector = (rank, result)

        print(
            f"TARGET VECTOR RANK = {rank}"
        )

        print(
            f"similarity = "
            f"{result.get('similarity')}"
        )

        print(
            f"title = "
            f"{result.get('title')}"
        )

        print(
            f"section = "
            f"{result.get('section')}"
        )

        print(
            "\nCONTENT:\n"
        )

        print(
            result.get("content", "")
        )

        print("\n" + "-" * 80)


# ============================================================
# 2. Rerank
# ============================================================

reranked = rerank(
    QUERY,
    vector_results,
)

print("\n[2] RERANK")

target_rerank = None

for rank, result in enumerate(
    reranked,
    start=1,
):

    if normalize_url(result.get("url")) == normalize_url(TARGET):

        if target_rerank is None:
            target_rerank = (rank, result)

        print(
            f"TARGET RERANK RANK = {rank}"
        )

        print(
            f"reranker_score = "
            f"{result.get('reranker_score')}"
        )

        print(
            f"similarity = "
            f"{result.get('similarity')}"
        )

        print(
            f"title = "
            f"{result.get('title')}"
        )

        print(
            f"section = "
            f"{result.get('section')}"
        )

        print(
            "\nCONTENT:\n"
        )

        print(
            result.get("content", "")
        )

        print("\n" + "-" * 80)


# ============================================================
# 3. Top reranked candidates
# ============================================================

print("\n[3] TOP 5 RERANKED CANDIDATES")

for rank, result in enumerate(
    reranked[:5],
    start=1,
):

    print()
    print(
        f"RANK {rank}"
    )

    print(
        f"URL: {result.get('url')}"
    )

    print(
        f"reranker_score: "
        f"{result.get('reranker_score')}"
    )

    print(
        f"similarity: "
        f"{result.get('similarity')}"
    )

    print(
        f"title: "
        f"{result.get('title')}"
    )

    print(
        f"section: "
        f"{result.get('section')}"
    )

    print(
        "\nCONTENT:\n"
    )

    print(
        result.get("content", "")
    )

    print("\n" + "-" * 80)


# ============================================================
# 4. Summary
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if target_vector:
    print(
        f"Vector target rank: "
        f"{target_vector[0]}"
    )
else:
    print("Vector target: NOT FOUND")

if target_rerank:
    print(
        f"Reranker target rank: "
        f"{target_rerank[0]}"
    )
    print(
        f"Reranker score: "
        f"{target_rerank[1].get('reranker_score')}"
    )
else:
    print("Reranker target: NOT FOUND")

