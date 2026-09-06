# Client transports

Every `Client` talks to its server over a **transport**: the thing that actually carries the messages.

You never configure one separately. `Client` takes a single positional argument and works the transport out from its type.

The *server* side of each (what `mcp.run()` does and what you deploy) is **[Running your server](https://py.sdk.modelcontextprotocol.io/run/index.md)**.

## In memory

Pass the server object itself:

```python title="client.py" hl_lines="14"
# docs_src/client_transports/tutorial001.py
from mcp import Client
from mcp.server import MCPServer

mcp = MCPServer("Bookshop")


@mcp.tool()
def search_books(query: str) -> str:
    """Search the catalog by title or author."""
    return f"Found 3 books matching {query!r}."


async def main() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("search_books", {"query": "dune"})
        print(result.structured_content)
```

No subprocess, no port, no bytes on a wire. The client and the server are two objects in the same process, and the call still goes through the real protocol layer: `search_books` is listed, validated and invoked exactly as it would be over HTTP.

That makes it two things at once:

* **A test harness.** Every example in this documentation is exercised this way, and the **[Testing](https://py.sdk.modelcontextprotocol.io/get-started/testing/index.md)** page builds the whole pattern around it.
* **An embedding API.** An application that constructs the server doesn't need a network hop to call its tools.

## Streamable HTTP

Pass a URL string and you get **Streamable HTTP**, the transport you deploy behind:

```python title="client.py" hl_lines="5"
# docs_src/client_transports/tutorial002.py
from mcp import Client


async def main() -> None:
    async with Client("http://localhost:8000/mcp") as client:
        result = await client.list_tools()
        print([tool.name for tool in result.tools])
```

That is the whole production client. `Client` wraps the URL in `streamable_http_client(...)` for you, on top of an `httpx2.AsyncClient` configured the way MCP needs: `follow_redirects=True`, a 30-second timeout for connect/write/pool, and a 300-second read timeout because the server may hold a response stream open.

!!! check
    A `Client` you have constructed is **not** connected. Construction only picks the transport;
    `async with` is what opens it. Reach for the connection before entering and the SDK tells you so:

    ```text
    RuntimeError: Client must be used within an async context manager
    ```

    Nothing was resolved, fetched or spawned when you wrote `Client("http://...")`. That line is free.

### Bring your own `httpx2.AsyncClient`

The moment you need an `Authorization` header, a cookie, a proxy, mTLS, or a different timeout, build the `httpx2.AsyncClient` yourself and hand it to `streamable_http_client`:

```python title="client.py" hl_lines="8-14"
# docs_src/client_transports/tutorial003.py
import httpx2

from mcp import Client
from mcp.client.streamable_http import streamable_http_client


async def main() -> None:
    async with httpx2.AsyncClient(
        headers={"Authorization": "Bearer ..."},
        timeout=httpx2.Timeout(30.0, read=300.0),
        follow_redirects=True,
    ) as http_client:
        transport = streamable_http_client("http://localhost:8000/mcp", http_client=http_client)
        async with Client(transport) as client:
            result = await client.list_tools()
            print([tool.name for tool in result.tools])
```

Two things to notice:

* You own the `httpx2.AsyncClient`, so **you** enter and exit it. The SDK never closes a client it didn't create.
* `streamable_http_client(url, http_client=...)` returns a transport, and `Client(transport)` accepts it like anything else.

One TLS note: `httpx2` verifies certificates against the operating system trust store (via
[`truststore`](https://pypi.org/project/truststore/)), not a bundled CA list. In an environment with
no usable system CA store (some minimal containers), set the standard `SSL_CERT_FILE`/`SSL_CERT_DIR`
environment variables or pass an explicit `verify=ssl_context` to your `httpx2.AsyncClient`
(background in
[`httpx` and `httpx-sse` replaced by `httpx2`](https://py.sdk.modelcontextprotocol.io/migration/index.md#httpx-and-httpx-sse-replaced-by-httpx2)).

!!! warning
    `streamable_http_client` used to take `headers=` and `timeout=` directly. It does not any more:
    its only parameters are `url`, `http_client` and `terminate_on_close`. Reach for `headers=` out
    of habit and you get:

    ```text
    TypeError: streamable_http_client() got an unexpected keyword argument 'headers'
    ```

    Everything HTTP-shaped now lives on the one `httpx2.AsyncClient` you pass in.

!!! info
    `httpx2` keeps the familiar `httpx` API, so if you know `httpx` you already know how to do auth,
    proxies, event hooks, retries and connection limits here. The SDK adds nothing on top and takes
    nothing away. It is also where OAuth plugs in:
    `httpx2.AsyncClient(auth=OAuthClientProvider(...))`. That whole flow is **[OAuth clients](https://py.sdk.modelcontextprotocol.io/client/oauth-clients/index.md)**.

## stdio

A **stdio** server is a subprocess. The client launches it, writes JSON-RPC to its stdin and reads JSON-RPC from its stdout. It is how a desktop host runs a server on your machine: a host *is* this code plus a UI, and **[Connect to a real host](https://py.sdk.modelcontextprotocol.io/get-started/real-host/index.md)** is the same relationship seen from the host's side, as a config file.

Describe the process with `StdioServerParameters` and hand it to `Client`:

```python title="client.py" hl_lines="3-7 11"
# docs_src/client_transports/tutorial004.py
from mcp import Client, StdioServerParameters

server = StdioServerParameters(
    command="uv",
    args=["run", "server.py"],
    env={"BOOKSHOP_API_KEY": "secret"},
)


async def main() -> None:
    async with Client(server) as client:
        result = await client.list_tools()
        print([tool.name for tool in result.tools])
```

Entering the block spawns the process. Leaving it shuts the subprocess down: close stdin, wait, kill if it lingers. You never clean it up yourself.

The child's stderr goes to yours. To send it somewhere else, build the transport yourself with `stdio_client` (from `mcp`) and pass that instead: `Client(stdio_client(server, errlog=log_file))`.

!!! warning
    The child does **not** inherit your environment. It gets a minimal allow-list (`HOME`, `LOGNAME`,
    `PATH`, `SHELL`, `TERM` and `USER` on POSIX) so nothing sensitive leaks into a process you may
    not have written.

    A server that needs an API key won't find it there. Pass it explicitly with `env=`; those
    variables are merged on top of the allow-list. That is what `BOOKSHOP_API_KEY` is doing above.

## SSE

`sse_client(url)`, from `mcp.client.sse`, is the HTTP transport that Streamable HTTP superseded. Wrap it the same way, `Client(sse_client("http://localhost:8000/sse"))`, to talk to a server that still speaks it, and don't build anything new on it.

## The `Transport` protocol

To `Client`, all of the above are the same thing.

A **transport** is any async context manager that yields a `(read, write)` pair of message streams: formally, the `Transport` protocol in `mcp.client`. `Client` resolves its argument by type: a server object connects in-process, a `str` becomes `streamable_http_client(url)`, a `StdioServerParameters` becomes `stdio_client(params)`, and anything else is entered as a transport directly. That last rule is why `stdio_client(...)`, `streamable_http_client(...)` and `sse_client(...)` all drop into the same slot, and why you can write your own.

## Recap

* `Client(mcp)` (the server object) connects in memory. Use it for tests and for embedding.
* `Client("http://.../mcp")` (a URL) connects over Streamable HTTP, the production transport.
* Headers, auth, proxies and timeouts belong on an `httpx2.AsyncClient` you pass to `streamable_http_client(url, http_client=...)`. There is no `headers=` keyword.
* stdio is `Client(StdioServerParameters(...))`. Wrap it in `stdio_client(...)` yourself only to redirect the child's stderr.
* The subprocess gets an allow-listed environment, not yours; `env=` adds to it.
* A transport is anything you can `async with x as (read, write)`. `Client` hands anything that isn't a server object, a URL or `StdioServerParameters` straight to that protocol.
* Constructing a `Client` picks the transport. `async with` opens it.

Once the transport is open the two sides have to agree on a protocol version. You normally never think about it; when you do, **[Protocol versions](https://py.sdk.modelcontextprotocol.io/protocol-versions/index.md)** is the page.
