from retrieval.search import vector_search
from retrieval.reranker import rerank


QUERY = "What happens when an MCP tool returns a primitive type?"

TARGET = "https://py.sdk.modelcontextprotocol.io/servers/structured-output/"


def normalize(url):
    if not url:
        return ""

    return url.strip().rstrip("/") + "/"


# ============================================================
# 1. Vector candidates
# ============================================================

candidates = vector_search(
    query=QUERY,
    limit=100,
)

target = normalize(TARGET)


print("=" * 80)
print("Q013 - VECTOR → RERANKER ANALYSIS")
print("=" * 80)

print(f"Query: {QUERY}")
print(f"Vector candidates: {len(candidates)}")
print()


# ============================================================
# 2. Find target Vector rank
# ============================================================

vector_rank = None

for rank, result in enumerate(
    candidates,
    start=1,
):

    source = (
        result.get("url")
        or result.get("source")
    )

    if normalize(source) == target:

        vector_rank = rank
        break


print("=" * 80)
print("VECTOR RESULT")
print("=" * 80)

print(
    f"Target Vector rank: {vector_rank}"
)

print()


# ============================================================
# 3. Rerank
# ============================================================

reranked = rerank(
    QUERY,
    candidates,
)


# ============================================================
# 4. Find target Reranker rank
# ============================================================

reranker_rank = None
target_result = None

# Vector rank -> zero-based original index
target_original_index = None

if vector_rank is not None:
    target_original_index = vector_rank - 1


for rank, result in enumerate(
    reranked,
    start=1,
):

    if result.get("_original_index") == target_original_index:

        reranker_rank = rank
        target_result = result
        break

print("=" * 80)
print("RERANKER RESULT")
print("=" * 80)

print(
    f"Target Reranker rank: {reranker_rank}"
)

if target_result:

    print(
    f"Target original index: "
    f"{target_result.get('_original_index')}"
   )

    print(
        f"Target reranker score: "
        f"{target_result.get('reranker_score')}"
    )

print()


# ============================================================
# 5. Rank change
# ============================================================

if vector_rank is not None and reranker_rank is not None:

    rank_change = (
        vector_rank
        - reranker_rank
    )

    print("=" * 80)
    print("RANK CHANGE")
    print("=" * 80)

    print(
        f"Vector rank    : {vector_rank}"
    )

    print(
        f"Reranker rank  : {reranker_rank}"
    )

    print(
        f"Rank change    : {rank_change:+d}"
    )

    if reranker_rank > vector_rank:

        print()
        print(
            "RERANKER MOVED TARGET DOWN"
        )

    elif reranker_rank < vector_rank:

        print()
        print(
            "RERANKER MOVED TARGET UP"
        )

    else:

        print()
        print(
            "RANK UNCHANGED"
        )

print()


# ============================================================
# 6. Show top 20 reranked results
# ============================================================

print("=" * 80)
print("RERANKED TOP 20")
print("=" * 80)


for rank, result in enumerate(
    reranked[:20],
    start=1,
):

    source = (
        result.get("url")
        or result.get("source")
    )

    title = result.get(
        "title",
        "",
    )

    score = result.get(
        "reranker_score"
    )

    marker = ""

    if normalize(source) == target:
        marker = "  <<< TARGET"

    print(
        f"[{rank:02d}] "
        f"score={score: .6f} "
        f"{title}"
        f"{marker}"
    )

    print(
        f"      {source}"
    )