import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Configuration
# ============================================================

def get_environment_config() -> tuple[str, str | None, str]:
    """
    Load and validate OpenAI configuration from environment
    variables.

    Returns:
        api_key:
            OpenAI API key.

        base_url:
            Optional custom OpenAI-compatible API endpoint.

        model:
            Model name.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    if not model:
        raise RuntimeError(
            "OPENAI_MODEL is not set."
        )

    return api_key, base_url, model


# ============================================================
# 3. MCP Server Configuration
# ============================================================

def create_mcp_server_params() -> StdioServerParameters:
    """
    Create the configuration used by the MCP client
    to start the MCP server.
    """

    return StdioServerParameters(
        command="uv",
        args=[
            "run",
            "python",
            "-m",
            "mcp_server.server",
        ],
    )


# ============================================================
# 4. OpenAI Client
# ============================================================

def create_openai_client(
    api_key: str,
    base_url: str | None,
) -> OpenAI:
    """
    Create an OpenAI client.
    """

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


# ============================================================
# 5. MCP Tool Discovery
# ============================================================

async def get_mcp_tools(
    session: ClientSession,
):
    """
    Ask the MCP server for the tools it provides.
    """

    tools_result = await session.list_tools()

    print()
    print("=" * 80)
    print("Available MCP Tools")
    print("=" * 80)

    for tool in tools_result.tools:

        print(
            f"Name: {tool.name}"
        )

        print(
            f"Description: "
            f"{tool.description or ''}"
        )

        print(
            f"Input schema: "
            f"{tool.input_schema}"
        )

        print("-" * 80)

    return tools_result


# ============================================================
# 6. MCP -> OpenAI Tool Conversion
# ============================================================

def convert_mcp_tools_to_openai(
    tools_result,
) -> list[dict[str, Any]]:
    """
    Convert MCP tools into the format expected by
    OpenAI Chat Completions API.
    """

    openai_tools: list[dict[str, Any]] = []

    for tool in tools_result.tools:

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (
                        tool.description or ""
                    ),
                    "parameters": tool.input_schema,
                },
            }
        )

    return openai_tools


# ============================================================
# 7. Print MCP Tools
# ============================================================

def print_mcp_tools(
    tools_result,
) -> None:
    """
    Display all tools exposed by the MCP server.
    """

    print()
    print("=" * 80)
    print("Available MCP Tools")
    print("=" * 80)

    for tool in tools_result.tools:

        print(
            f"Name: {tool.name}"
        )

        print(
            f"Description: "
            f"{tool.description or ''}"
        )

        print(
            f"Input schema: "
            f"{tool.input_schema}"
        )

        print("-" * 80)


# ============================================================
# 8. MCP Result -> Text
# ============================================================

def mcp_result_to_text(
    result,
) -> str:
    """
    Convert MCP CallToolResult into plain text
    that can be passed back to the LLM.
    """

    text_parts: list[str] = []

    for content in result.content:

        if hasattr(content, "text"):

            text_parts.append(
                content.text
            )

        else:

            text_parts.append(
                str(content)
            )

    return "\n".join(text_parts)


# ============================================================
# 9. Execute One MCP Tool
# ============================================================

async def execute_mcp_tool(
    session: ClientSession,
    tool_call,
) -> str:
    """
    Execute one MCP tool requested by the LLM.
    """

    tool_name = tool_call.function.name

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    try:

        arguments = json.loads(
            tool_call.function.arguments
        )

    except json.JSONDecodeError as exc:

        error_message = (
            f"Invalid JSON arguments for "
            f"tool '{tool_name}': {exc}"
        )

        print()
        print("=" * 80)
        print("MCP Tool Error")
        print("=" * 80)
        print(error_message)

        return error_message

    # --------------------------------------------------------
    # Print tool request
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LLM requested MCP Tool")
    print("=" * 80)

    print(
        f"Tool: {tool_name}"
    )

    print(
        f"Arguments: {arguments}"
    )

    # --------------------------------------------------------
    # Call MCP tool
    # --------------------------------------------------------

    try:

        result = await session.call_tool(
            tool_name,
            arguments=arguments,
        )

    except Exception as exc:

        error_message = (
            f"Error executing MCP tool "
            f"'{tool_name}': {exc}"
        )

        print()
        print("=" * 80)
        print("MCP Tool Error")
        print("=" * 80)
        print(error_message)

        return error_message

    # --------------------------------------------------------
    # Convert result to text
    # --------------------------------------------------------

    tool_text = mcp_result_to_text(
        result
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("MCP Tool Result")
    print("=" * 80)

    print(tool_text)

    return tool_text


# ============================================================
# 10. Agent Loop
# ============================================================

async def agent_loop(
    client: OpenAI,
    session: ClientSession,
    openai_tools: list[dict[str, Any]],
    question: str,
    model: str,
) -> str:
    """
    Main LLM <-> MCP tool-calling loop.

    Flow:

        User
          ↓
        LLM
          ↓
        Tool Call
          ↓
        MCP Server
          ↓
        Tool Result
          ↓
        LLM
          ↓
        Final Answer
    """

    # --------------------------------------------------------
    # Initial conversation
    # --------------------------------------------------------

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": question,
        }
    ]

    # --------------------------------------------------------
    # Agent loop
    # --------------------------------------------------------

    while True:

        # ====================================================
        # 1. Ask LLM
        # ====================================================

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # ====================================================
        # 2. No tool call -> final answer
        # ====================================================

        if not message.tool_calls:

            final_answer = (
                message.content or ""
            )

            print()
            print("=" * 80)
            print("Final Answer")
            print("=" * 80)

            print(
                final_answer
            )

            return final_answer

        # ====================================================
        # 3. Add assistant tool-call message
        # ====================================================

        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": (
                            tool_call.function.name
                        ),
                        "arguments": (
                            tool_call.function.arguments
                        ),
                    },
                }
                for tool_call in message.tool_calls
            ],
        }

        messages.append(
            assistant_message
        )

        # ====================================================
        # 4. Execute MCP tools
        # ====================================================

        for tool_call in message.tool_calls:

            tool_text = await execute_mcp_tool(
                session=session,
                tool_call=tool_call,
            )

            # =================================================
            # 5. Add MCP result to conversation
            # =================================================

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_text,
                }
            )


# ============================================================
# 11. Main
# ============================================================

async def main():

    # ========================================================
    # 1. Load configuration
    # ========================================================

    api_key, base_url, model = (
        get_environment_config()
    )

    print()
    print("=" * 80)
    print("MCP RAG Assistant")
    print("=" * 80)

    print(
        f"Model: {model}"
    )

    if base_url:

        print(
            f"Base URL: {base_url}"
        )

    # ========================================================
    # 2. Create OpenAI client
    # ========================================================

    client = create_openai_client(
        api_key=api_key,
        base_url=base_url,
    )

    # ========================================================
    # 3. Create MCP Server parameters
    # ========================================================

    server_params = (
        create_mcp_server_params()
    )

    # ========================================================
    # 4. Start MCP Server
    # ========================================================

    async with stdio_client(
        server_params
    ) as (
        read_stream,
        write_stream,
    ):

        # ====================================================
        # 5. Create MCP Session
        # ====================================================

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            # =================================================
            # 6. Initialize MCP session
            # =================================================

            await session.initialize()

            # =================================================
            # 7. Discover MCP tools
            # =================================================

            tools_result = (
                await get_mcp_tools(
                    session
                )
            )

            # =================================================
            # 8. Convert MCP tools -> OpenAI tools
            # =================================================

            openai_tools = (
                convert_mcp_tools_to_openai(
                    tools_result
                )
            )

            # =================================================
            # 9. Ask user
            # =================================================

            question = input(
                "\nQuestion: "
            )

            # =================================================
            # 10. Run Agent
            # =================================================

            await agent_loop(
                client=client,
                session=session,
                openai_tools=openai_tools,
                question=question,
                model=model,
            )


# ============================================================
# 12. Entry Point
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )