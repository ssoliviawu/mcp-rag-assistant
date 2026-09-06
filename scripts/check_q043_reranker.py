from retrieval.search import vector_search
from retrieval.reranker import rerank


QUERY = (
    "Why would I use an MCP dependency "
    "instead of calling a helper directly?"
)


# ============================================================
# Vector Top 200
# ============================================================

results = vector_search(
    QUERY,
    limit=200,
)

print("=" * 100)
print("VECTOR")
print("=" * 100)

for rank, result in enumerate(
    results,
    start=1,
):

    if result["id"] in {
        "mcp-python-sdk-handlers-dependencies-4",
        "mcp-python-sdk-handlers-dependencies-11",
    }:

        print(
            f"Vector rank={rank} | "
            f"similarity={result['similarity']:.6f} | "
            f"{result['id']}"
        )


# ============================================================
# Rerank
# ============================================================

reranked = rerank(
    QUERY,
    results,
)


print()
print("=" * 100)
print("RERANK TOP 20")
print("=" * 100)

for rank, result in enumerate(
    reranked[:20],
    start=1,
):

    print(
        f"{rank:3d} | "
        f"rerank={result['reranker_score']:.6f} | "
        f"vector={result['similarity']:.6f} | "
        f"{result['id']} | "
        f"{result['title']}"
    )


# ============================================================
# Target chunks
# ============================================================

print()
print("=" * 100)
print("TARGETS")
print("=" * 100)

targets = {
    "mcp-python-sdk-handlers-dependencies-4",
    "mcp-python-sdk-handlers-dependencies-11",
}

for rank, result in enumerate(
    reranked,
    start=1,
):

    if result["id"] in targets:

        print(
            f"{result['id']}"
        )

        print(
            f"Rerank rank : {rank}"
        )

        print(
            f"Rerank score: "
            f"{result['reranker_score']:.6f}"
        )

        print(
            f"Vector sim  : "
            f"{result['similarity']:.6f}"
        )

        print()
        print(result["content"])
        print("-" * 100)