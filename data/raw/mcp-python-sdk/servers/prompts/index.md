# Prompts

A **prompt** is a message template the user picks.

Tools are for the model. A prompt is the opposite: the user chooses one from a menu in their client (a slash command, a button), fills in its arguments, and the rendered messages go into the conversation as if they had typed them.

You declare one by putting `@mcp.prompt()` on a function that returns the text.

## Your first prompt

```python title="server.py" hl_lines="6-9"
# docs_src/prompts/tutorial001.py
from mcp.server import MCPServer

mcp = MCPServer("Code Helper")


@mcp.prompt()
def review_code(code: str) -> str:
    """Review a piece of code."""
    return f"Please review this code:\n\n{code}"
```

The SDK reads the same three things it reads from a tool:

* The **name** is the function name: `review_code`.
* The **description** the client shows is the docstring: `Review a piece of code.`
* The **arguments** come from the parameters. `code` has no default, so it's required.

That is what a client gets back from `prompts/list`:

```json
{
  "name": "review_code",
  "description": "Review a piece of code.",
  "arguments": [
    {"name": "code", "required": true}
  ]
}
```

There is no JSON Schema here. Prompt arguments are a flat list of **named string values**: a form a person fills in, not a payload a model constructs.

### Rendering it

The client renders the template with `prompts/get`, passing the arguments. Your function runs and the `str` you return becomes **one user message**:

```json
{
  "description": "Review a piece of code.",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Please review this code:\n\ndef add(a, b): return a + b"
      }
    }
  ],
  "resultType": "complete"
}
```

That is the entire life of a prompt: listed by name, rendered on demand, dropped into the chat.

!!! check
    `required` is enforced before your function runs. Render `review_code` without `code` and the
    request itself fails with a JSON-RPC error (code `-32603`):

    ```text
    mcp.shared.exceptions.MCPError: Internal server error
    ```

    There is no tool-style error result to hand back to a model, because no model is in the loop:
    the call raises. The reason (`Missing required arguments: {'code'}`) lands in your server's log.

### Try it

Run the server with the MCP Inspector:

```console
uv run mcp dev server.py
```

Open the **Prompts** tab and select `review_code`. The Inspector draws a form with one required `code` field. Fill it in, render it, and you get back exactly the user message above.

## More than one message

A code review is one message. A debugging session is a conversation, and a prompt can seed the whole thing.

Return a list of messages instead of a `str`:

```python title="server.py" hl_lines="2 13-20"
# docs_src/prompts/tutorial002.py
from mcp.server import MCPServer
from mcp.server.mcpserver.prompts.base import AssistantMessage, Message, UserMessage

mcp = MCPServer("Code Helper")


@mcp.prompt()
def review_code(code: str) -> str:
    """Review a piece of code."""
    return f"Please review this code:\n\n{code}"


@mcp.prompt()
def debug_error(error: str) -> list[Message]:
    """Start a debugging conversation."""
    return [
        UserMessage("I'm seeing this error:"),
        UserMessage(error),
        AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]
```

* `UserMessage` and `AssistantMessage` come from `mcp.server.mcpserver.prompts.base`. Hand them a `str` and they wrap it in `TextContent` for you. The role is the class name.
* `Message` is their common base. Use it as the return annotation.

Rendering `debug_error` now produces three messages, in order:

```json
{
  "description": "Start a debugging conversation.",
  "messages": [
    {"role": "user", "content": {"type": "text", "text": "I'm seeing this error:"}},
    {"role": "user", "content": {"type": "text", "text": "TypeError: 'int' object is not iterable"}},
    {
      "role": "assistant",
      "content": {"type": "text", "text": "I'll help debug that. What have you tried so far?"}
    }
  ],
  "resultType": "complete"
}
```

Notice the last one. Pre-filling an `assistant` turn is how you steer the model's *next* reply without making the user type the steering themselves.

## Titles and argument descriptions

`review_code` is a function name, not a label. Give the client something better to put on the button, and describe each argument so the form explains itself:

```python title="server.py" hl_lines="10-13"
# docs_src/prompts/tutorial003.py
from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer

mcp = MCPServer("Code Helper")


@mcp.prompt(title="Code review")
def review_code(
    code: Annotated[str, Field(description="The code to review.")],
    language: Annotated[str, Field(description="The language the code is written in.")] = "python",
) -> str:
    """Review a piece of code."""
    return f"Please review this {language} code:\n\n{code}"
```

* `title="Code review"` is the human-readable name, exactly like a tool's `title`.
* `Annotated[str, Field(description=...)]` is the same pattern **[Tools](https://py.sdk.modelcontextprotocol.io/servers/tools/index.md)** uses to describe a tool's parameters. Here the description lands on the argument instead of in a schema.
* `language` has a default, so it stops being required.

The `prompts/list` entry now carries everything a client needs to draw a good form:

```json
{
  "name": "review_code",
  "title": "Code review",
  "description": "Review a piece of code.",
  "arguments": [
    {"name": "code", "description": "The code to review.", "required": true},
    {"name": "language", "description": "The language the code is written in.", "required": false}
  ]
}
```

!!! info
    If you have read **[Tools](https://py.sdk.modelcontextprotocol.io/servers/tools/index.md)**, you already know everything up to this point. Same decorator, same
    docstring-as-description, same `Annotated`/`Field`. The only things that change are who
    triggers it (the user) and where the result goes (into the conversation).

## More than text

`UserMessage` and `AssistantMessage` also accept a content block, or an `Image` / `Audio` helper, wherever they accept a `str`. Two cases come up in prompts: attaching a document and attaching a picture.

### Embedding a file

```python title="server.py" hl_lines="5 12 21 23"
# docs_src/prompts/tutorial004.py
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver import Message, UserMessage
from mcp.types import EmbeddedResource, TextResourceContents

mcp = MCPServer("Code Helper")

STYLE_GUIDE_FILE = Path(__file__).parent / "style-guide.md"  # or the path to your file on disk


@mcp.resource("style://python", mime_type="text/markdown")
def style_guide() -> str:
    """The team's Python style guide."""
    return STYLE_GUIDE_FILE.read_text(encoding="utf-8")


@mcp.prompt()
def review_code(code: str) -> list[Message]:
    """Review a piece of code against the team style guide."""
    guide = TextResourceContents(uri="style://python", mime_type="text/markdown", text=style_guide())
    return [
        UserMessage(EmbeddedResource(resource=guide)),
        UserMessage(f"Review this code against the style guide above:\n\n{code}"),
    ]
```

* The style guide is a resource at `style://python` (**[Resources](https://py.sdk.modelcontextprotocol.io/servers/resources/index.md)** covers those), read from a `style-guide.md` next to `server.py`. Put any Markdown file there.
* `EmbeddedResource(resource=TextResourceContents(...))`, both from `mcp.types`, carries the file with its URI and MIME type as the first message; the request that refers to it follows as plain text.
* Embedding, rather than pasting the guide into the f-string, lets the client show it as an attachment and reopen `style://python` later, and the model receives the file verbatim. For a binary file use `BlobResourceContents` with a base64 `blob`.

Rendered, the first message's `content` is a `resource` block:

```json
{"type": "resource", "resource": {"uri": "style://python", "mimeType": "text/markdown", "text": "* Prefer early returns.\n..."}}
```

### Attaching an image

```python title="server.py" hl_lines="4 15"
# docs_src/prompts/tutorial005.py
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver import Image, Message, UserMessage

mcp = MCPServer("Code Helper")

DIAGRAM_FILE = Path(__file__).parent / "architecture.png"  # or the path to your file on disk


@mcp.prompt()
def explain_component(component: str) -> list[Message]:
    """Explain one component using the architecture diagram."""
    return [
        UserMessage(Image(path=DIAGRAM_FILE)),
        UserMessage(f"Where does {component} sit in this architecture, and what does it talk to?"),
    ]
```

* `Image` is the helper from **[Images, audio & icons](https://py.sdk.modelcontextprotocol.io/servers/media/index.md)**. `UserMessage` converts it to an `ImageContent` block (the file base64-encoded, MIME type guessed from `.png`) when the prompt renders; `Audio` becomes an `AudioContent` the same way.
* Put any PNG named `architecture.png` beside `server.py`. Prompt arguments are strings, so the picture always comes from the server; `component` only supplies the words.

```json
{"type": "image", "data": "iVBORw0KGgoAAAANSUhEUg...", "mimeType": "image/png"}
```

## Changing the list at runtime

Prompts can be added while clients are connected, for example to let a user save an instruction as a menu entry of their own. Register the prompt, then notify:

```python title="server.py" hl_lines="5 23-27"
# docs_src/prompts/tutorial006.py
from contextlib import suppress

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.prompts import Prompt

mcp = MCPServer("Code Helper")


@mcp.prompt()
def review_code(code: str) -> str:
    """Review a piece of code."""
    return f"Please review this code:\n\n{code}"


@mcp.tool()
async def save_template(name: str, instruction: str, ctx: Context) -> str:
    """Save an instruction as a prompt the user can pick from the menu."""

    def template(code: str) -> str:
        return f"{instruction}\n\n{code}"

    with suppress(ValueError):  # replace an existing entry of the same name
        mcp.remove_prompt(name)
    mcp.add_prompt(Prompt.from_function(template, name=name, description=instruction))
    await ctx.notify_prompts_changed()
    await ctx.session.send_prompt_list_changed()
    return f"Saved '{name}' to the prompt menu."
```

* `mcp.add_prompt(Prompt.from_function(fn, name=..., description=...))` registers a function exactly as `@mcp.prompt()` would, and `mcp.remove_prompt(name)` is the reverse. `add_prompt` keeps an existing entry of the same name rather than overwrite it, so the tool removes any old one first to make saving a replace. `prompts/list` reflects the change immediately.
* `await ctx.notify_prompts_changed()` sends `notifications/prompts/list_changed` to every `2026-07-28` client listening on a `subscriptions/listen` stream (**[Subscriptions](https://py.sdk.modelcontextprotocol.io/handlers/subscriptions/index.md)**). `await ctx.session.send_prompt_list_changed()` sends it to the calling client when that client is pre-2026 (**[Serving legacy clients](https://py.sdk.modelcontextprotocol.io/run/legacy-clients/index.md)**). Call both; each does nothing when there is nobody to tell.
* A client that receives the notification calls `prompts/list` again. In the Python `Client` that is `async with client.listen(prompts_list_changed=True) as sub:`, which yields a `PromptsListChanged` event.

## Recap

* `@mcp.prompt()` on a function makes it a prompt. Name from the function, description from the docstring.
* Prompts are **user-controlled**: the client lists them, the user picks one and fills in the arguments.
* Arguments are a flat list of named strings (no schema). A parameter with a default is optional.
* Return a `str` and it becomes one user message. Return a list of `UserMessage` / `AssistantMessage` to seed a multi-turn conversation.
* `title=` and `Field(description=...)` are what a client puts in its UI.
* A missing required argument fails the whole request. There is no per-prompt error result.
* Wrap an `EmbeddedResource` or an `Image` in a `UserMessage` to attach a document or a picture.
* Add or remove prompts at runtime with `mcp.add_prompt(...)` / `mcp.remove_prompt(...)`, then `await ctx.notify_prompts_changed()` and `await ctx.session.send_prompt_list_changed()`.

Server-side autocomplete for a prompt's (or a resource template's) arguments is **[Completions](https://py.sdk.modelcontextprotocol.io/servers/completions/index.md)**.
