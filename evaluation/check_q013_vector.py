from retrieval.search import vector_search


QUERY = "What happens when an MCP tool returns a primitive type?"

TARGET = "https://py.sdk.modelcontextprotocol.io/servers/structured-output/"


def normalize(url):
    if not url:
        return ""

    return url.strip().rstrip("/") + "/"


results = vector_search(
    query=QUERY,
    limit=100,
)


target = normalize(TARGET)


print("=" * 80)
print("Q013 - VECTOR TOP 100")
print("=" * 80)

print(f"Query: {QUERY}")
print(f"Candidate count: {len(results)}")
print()


found = False


for rank, result in enumerate(results, start=1):

    source = (
        result.get("url")
        or result.get("source")
    )

    title = result.get(
        "title",
        "",
    )

    normalized_source = normalize(
        source
    )

    if normalized_source == target:

        found = True

        print("=" * 80)
        print("TARGET FOUND")
        print("=" * 80)

        print(f"Vector rank : {rank}")
        print(f"Title       : {title}")
        print(f"Source      : {source}")

        if "score" in result:
            print(
                f"Score       : {result['score']}"
            )

        if "distance" in result:
            print(
                f"Distance    : {result['distance']}"
            )

        print()
        print("Content preview:")
        print(
            result.get(
                "content",
                "",
            )[:3000]
        )

        break


if not found:

    print("=" * 80)
    print("TARGET NOT FOUND")
    print("=" * 80)

    print(
        "The Ground Truth source is NOT "
        "inside Vector Top-100."
    )

    print()
    print("Top 20 candidates:")

    for rank, result in enumerate(
        results[:20],
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

        print()
        print(f"[{rank}] {title}")
        print(f"     {source}")