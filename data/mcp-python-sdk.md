MCP Python SDK

[modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk "Go to repository")

* MCP Python SDK

  MCP Python SDK



  On this page
  + [Requirements](#requirements)
  + [Installation](#installation)
  + [Example](#example)

    - [Create it](#create-it)
    - [Run it](#run-it)
    - [Try it](#try-it)
    - [Recap](#recap)
  + [Where to go next](#where-to-go-next)
* [What's new in v2](./whats-new/)
* [Get started](./get-started/)

  Get started
  + [Installation](./get-started/installation/)
  + [First steps](./get-started/first-steps/)
  + [Connect to a real host](./get-started/real-host/)
  + [Testing](./get-started/testing/)
* [Servers](./servers/)

  Servers
  + [Tools](./servers/tools/)
  + [Structured Output](./servers/structured-output/)
  + [Resources](./servers/resources/)
  + [URI templates](./servers/uri-templates/)
  + [Prompts](./servers/prompts/)
  + [Completions](./servers/completions/)
  + [Images, audio & icons](./servers/media/)
  + [Handling errors](./servers/handling-errors/)
* [Inside your handler](./handlers/)

  Inside your handler
  + [The Context](./handlers/context/)
  + [Dependencies](./handlers/dependencies/)
  + [Lifespan](./handlers/lifespan/)
  + [Elicitation](./handlers/elicitation/)
  + [Multi-round-trip requests](./handlers/multi-round-trip/)
  + [Sampling and roots](./handlers/sampling-and-roots/)
  + [Progress](./handlers/progress/)
  + [Logging](./handlers/logging/)
  + [Subscriptions](./handlers/subscriptions/)
* [Running your server](./run/)

  Running your server
  + [Add to an existing app](./run/asgi/)
  + [Deploy & scale](./run/deploy/)
  + [Authorization](./run/authorization/)
  + [OpenTelemetry](./run/opentelemetry/)
  + [Serving legacy clients](./run/legacy-clients/)
* [Clients](./client/)

  Clients
  + [Callbacks](./client/callbacks/)
  + [Transports](./client/transports/)
  + [OAuth](./client/oauth-clients/)
  + [Identity assertion](./client/identity-assertion/)
  + [Multiple servers](./client/session-groups/)
  + [Subscriptions](./client/subscriptions/)
  + [Caching](./client/caching/)
* [Protocol versions](./protocol-versions/)
* [Deprecated features](./deprecated/)
* [Advanced](./advanced/)

  Advanced
  + [The low-level Server](./advanced/low-level-server/)
  + [Pagination](./advanced/pagination/)
  + [Middleware](./advanced/middleware/)
  + [Extensions](./advanced/extensions/)
  + [MCP Apps](./advanced/apps/)
* [Troubleshooting](./troubleshooting/)
* [Migration Guide](./migration/)
* API Reference




  API Reference
  + [mcp](./api/mcp/)
  + [mcp\_types](./api/mcp_types/)

On this page

* [Requirements](#requirements)
* [Installation](#installation)
* [Example](#example)

  + [Create it](#create-it)
  + [Run it](#run-it)
  + [Try it](#try-it)
  + [Recap](#recap)
* [Where to go next](#where-to-go-next)

MCP Python SDK
==============

This documents v2, the current stable release line

New to v2, or coming from v1? **[What's new in v2](whats-new/)** is the five-minute tour of what changed, and the **[Migration Guide](migration/)** covers every breaking change.
Still on v1.x? Its documentation lives at the [v1.x docs](https://py.sdk.modelcontextprotocol.io/v1/).
Something rough or confusing? [Tell us](https://github.com/modelcontextprotocol/python-sdk/issues/new?template=v2-feedback.yaml).

The **Model Context Protocol (MCP)** lets applications provide context to LLMs in a standardized way, separating the concern of *providing* context from the LLM interaction itself.

This is the official Python SDK for it. With it you can:

* **Build MCP servers** that expose tools, resources, and prompts to any MCP host.
* **Build MCP clients** that connect to any MCP server.
* Speak every standard transport: stdio, Streamable HTTP, and SSE.

Requirements
------------

Python 3.10+.

Installation
------------

uvpip

```
uv add "mcp[cli]"
```

```
pip install "mcp[cli]"
```

The `[cli]` extra gives you the `mcp` command; you'll want it for development.
See [Installation](get-started/installation/) for what each dependency is for.

Example
-------

### Create it

Create a file `server.py`:

server.py

```
from mcp.server import MCPServer

mcp = MCPServer("Demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"
```

That's a complete MCP server.

It exposes one **tool**, `add`, and one templated **resource**, `greeting://{name}`.

### Run it

```
uv run mcp dev server.py
```

This starts your server and opens the [MCP Inspector](https://github.com/modelcontextprotocol/inspector), an interactive UI for poking at it. Open the URL it prints.

Note

The Inspector is a Node.js app, so `mcp dev` needs `npx` on your `PATH`.

### Try it

In the Inspector, go to **Tools** and call `add` with `a=1`, `b=2`.

You get `3` back. ✨

The Inspector built that form (a required integer field for `a`, another for `b`) from your type hints. So will Claude, and every other MCP host.

Now go to **Resources** and read `greeting://World`:

```
Hello, World!
```

### Recap

Look again at what you did **not** write:

* No JSON Schema. `a: int, b: int` *is* the schema.
* No request parsing, no serialization, no validation code.
* No protocol handling at all.

You wrote two Python functions with type hints and a docstring. The SDK does the rest.

Where to go next
----------------

* **[Get started](get-started/)** takes you from install to a working, tested server.
* Building an application that *uses* MCP servers? Start with **[Clients](client/)**.
* Already have a FastAPI or Starlette app? **[Add to an existing app](run/asgi/)** mounts an MCP server inside it.
* Hunting an exact error message? **[Troubleshooting](troubleshooting/)** is keyed by the verbatim text.
* Wondering what changed in v2? **[What's new in v2](whats-new/)** is the five-minute tour.
* Migrating from v1? Start with the **[Migration Guide](migration/)**.
* Hunting for an exact signature? The **[API Reference](api/mcp/)** is generated from the source.
* Reading with an LLM? This documentation is also published in the [llms.txt](https://llmstxt.org/) format:
  [llms.txt](https://py.sdk.modelcontextprotocol.io/llms.txt) is an index of the pages, and
  [llms-full.txt](https://py.sdk.modelcontextprotocol.io/llms-full.txt) contains every page in a single file.

Back to top