from rag.retriever import retrieve
from rag.context import build_context
from rag.prompt import SYSTEM_PROMPT, build_prompt


def test_prompt():

    query = "What are MCP resources?"

    results = retrieve(
        query,
        limit=5,
    )

    context = build_context(results)

    prompt = build_prompt(
        query,
        context,
    )

    assert "What are MCP resources?" in prompt
    assert "resource" in prompt.lower()
    assert "[Source 1]" in prompt

    print("\n")
    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(SYSTEM_PROMPT)

    print("\n")
    print("=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(prompt)