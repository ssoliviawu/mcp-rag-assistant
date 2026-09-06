# Extensions

An **extension** is an opt-in bundle of MCP behaviour behind one identifier.

On a server it can contribute tools, resources, and new request methods, and it can wrap
`tools/call`. On a client it can claim extra `tools/call` result shapes and observe vendor
notifications. Each side advertises under its own `capabilities.extensions`, and nothing
changes for anyone who didn't ask for it. That is the contract ([SEP-2133](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2133)), and
it has one golden rule: **extensions are off by default**.

## Using an extension

Pass instances at construction:

```python title="server.py"
# docs_src/extensions/tutorial001.py
from mcp.server.apps import Apps
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("demo", extensions=[Apps()])
```

Done. The server now advertises `io.modelcontextprotocol/ui` under
`capabilities.extensions` and serves everything the extension contributes.

`Apps` is the built-in reference extension, and it gets its own page: **[MCP Apps](https://py.sdk.modelcontextprotocol.io/advanced/apps/index.md)**.

!!! note
    Extensions are fixed at construction. There is no `add_extension` to call later:
    a server's capability map should not change while clients are connected to it.

The capability map rides `server/discover`, which is a **2026-07-28** path. A legacy
`initialize` handshake has nowhere to put it, so a legacy client simply doesn't see
the extension. Design for that: an extension *augments* a server, it must not be the
only way the server is usable.

## Writing your own

Subclass `Extension` and override only what you need. Every method has a default.

### The identifier

```python
# docs_src/extensions/tutorial002.py
from mcp.server.extension import Extension


class Stamps(Extension):
    identifier = "com.example/stamps"
```

The identifier is a `vendor-prefix/name` string following the spec's `_meta` key
grammar: dot-separated labels (each starts with a letter, ends with a letter or
digit), a slash, then the name. It is validated **when the class is defined**, so a
typo doesn't wait for a server to boot:

```text
TypeError: Stamps.identifier must be a `vendor-prefix/name` string
(reverse-DNS prefix required), got 'stamps'
```

Use a domain you control as the prefix. `io.modelcontextprotocol/*` is for extensions
specified by the MCP project itself.

### Contributing tools

The smallest useful extension is one tool and a settings map:

```python title="server.py" hl_lines="17 19-20 22-23 26"
# docs_src/extensions/tutorial003.py
from collections.abc import Sequence
from typing import Any

from mcp import Client
from mcp.server.extension import Extension, ToolBinding
from mcp.server.mcpserver import MCPServer


def stamp(text: str) -> str:
    """Stamp a message with the office seal."""
    return f"[stamped] {text}"


class Stamps(Extension):
    """A purely additive extension: one tool, one capability entry."""

    identifier = "com.example/stamps"

    def settings(self) -> dict[str, Any]:
        return {"sealed": True}

    def tools(self) -> Sequence[ToolBinding]:
        return [ToolBinding(fn=stamp)]


mcp = MCPServer("post-office", extensions=[Stamps()])


async def main() -> None:
    async with Client(mcp) as client:
        print(client.server_capabilities.extensions)
        # {'com.example/stamps': {'sealed': True}}
        result = await client.call_tool("stamp", {"text": "hello"})
        print(result.content)
        # [TextContent(text='[stamped] hello')]
```

* `tools()` returns `ToolBinding`s. The server registers each one exactly as if you
  had called `mcp.add_tool(...)` yourself: same schema generation, same `Context`
  injection, same everything.
* `settings()` is the value advertised at `capabilities.extensions["com.example/stamps"]`.
  Return `{}` (the default) to advertise the extension with no settings.
* The extension never receives the server. It declares contributions as data;
  `MCPServer` consumes them. There is no `self.server` to mutate.

And `main()` is the proof, an in-memory client straight against `mcp`:

```python title="server.py" hl_lines="29-34"
# docs_src/extensions/tutorial003.py
from collections.abc import Sequence
from typing import Any

from mcp import Client
from mcp.server.extension import Extension, ToolBinding
from mcp.server.mcpserver import MCPServer


def stamp(text: str) -> str:
    """Stamp a message with the office seal."""
    return f"[stamped] {text}"


class Stamps(Extension):
    """A purely additive extension: one tool, one capability entry."""

    identifier = "com.example/stamps"

    def settings(self) -> dict[str, Any]:
        return {"sealed": True}

    def tools(self) -> Sequence[ToolBinding]:
        return [ToolBinding(fn=stamp)]


mcp = MCPServer("post-office", extensions=[Stamps()])


async def main() -> None:
    async with Client(mcp) as client:
        print(client.server_capabilities.extensions)
        # {'com.example/stamps': {'sealed': True}}
        result = await client.call_tool("stamp", {"text": "hello"})
        print(result.content)
        # [TextContent(text='[stamped] hello')]
```

### Serving your own methods

An extension can register **new request methods**: its own verbs, served next to the
spec's:

```python title="server.py" hl_lines="16-22 31 40-48"
# docs_src/extensions/tutorial004.py
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field

import mcp.types as types
from mcp import Client
from mcp.client import advertise
from mcp.server.context import ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import MCPServer, require_client_extension

EXTENSION_ID = "com.example/search"


class SearchParams(types.RequestParams):
    query: str
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(types.Result):
    items: list[str]


class SearchRequest(types.Request[SearchParams, Literal["com.example/search"]]):
    method: Literal["com.example/search"] = "com.example/search"
    params: SearchParams


async def search(ctx: ServerRequestContext[Any, Any], params: SearchParams) -> SearchResult:
    require_client_extension(ctx, EXTENSION_ID)
    return SearchResult(items=[f"{params.query}-{n}" for n in range(params.limit)])


class Search(Extension):
    """An extension that serves its own request method."""

    identifier = EXTENSION_ID

    def methods(self) -> Sequence[MethodBinding]:
        return [
            MethodBinding(
                "com.example/search",
                SearchParams,
                search,
                protocol_versions=frozenset({"2026-07-28"}),
            )
        ]


mcp = MCPServer("catalog", extensions=[Search()])


async def main() -> None:
    async with Client(mcp, extensions=[advertise(EXTENSION_ID)]) as client:
        request = SearchRequest(params=SearchParams(query="mcp", limit=3))
        result = await client.session.send_request(request, SearchResult)
        print(result.items)
        # ['mcp-0', 'mcp-1', 'mcp-2']
```

* `SearchParams` subclasses `RequestParams`, so the 2026 `_meta` envelope parses
  uniformly and your handler gets validated params, never a raw dict. Bound what
  the client controls: `Field(ge=1, le=100)` rejects an absurd `limit` before
  your code allocates anything for it.
* `require_client_extension(ctx, EXTENSION_ID)` is the gate: a client that did not
  declare the extension gets the `-32021` (missing required client capability) error,
  with the machine-readable `requiredCapabilities` payload the spec asks for.
* `protocol_versions=frozenset({"2026-07-28"})` pins the method to one wire version.
  At any other version the client gets `METHOD_NOT_FOUND`, exactly as if the method
  didn't exist there. For that client, it doesn't.

Methods are **strictly additive**. The SDK enforces this at construction, not at
runtime:

* A `MethodBinding` for a spec-defined method (`tools/list`, `completion/complete`, ...)
  raises `ValueError` when the binding is constructed. Core verbs belong to the server.
* Two extensions binding the same method raise when the second one registers.
  Last-write-wins is how plugins corrupt each other; we don't do that.
* An empty `protocol_versions` set raises too: a method that can never be served
  is a bug, not a configuration.

### The client side

The same file's `main()` is the whole client story, both halves of it:

```python title="server.py" hl_lines="54-58"
# docs_src/extensions/tutorial004.py
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field

import mcp.types as types
from mcp import Client
from mcp.client import advertise
from mcp.server.context import ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import MCPServer, require_client_extension

EXTENSION_ID = "com.example/search"


class SearchParams(types.RequestParams):
    query: str
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(types.Result):
    items: list[str]


class SearchRequest(types.Request[SearchParams, Literal["com.example/search"]]):
    method: Literal["com.example/search"] = "com.example/search"
    params: SearchParams


async def search(ctx: ServerRequestContext[Any, Any], params: SearchParams) -> SearchResult:
    require_client_extension(ctx, EXTENSION_ID)
    return SearchResult(items=[f"{params.query}-{n}" for n in range(params.limit)])


class Search(Extension):
    """An extension that serves its own request method."""

    identifier = EXTENSION_ID

    def methods(self) -> Sequence[MethodBinding]:
        return [
            MethodBinding(
                "com.example/search",
                SearchParams,
                search,
                protocol_versions=frozenset({"2026-07-28"}),
            )
        ]


mcp = MCPServer("catalog", extensions=[Search()])


async def main() -> None:
    async with Client(mcp, extensions=[advertise(EXTENSION_ID)]) as client:
        request = SearchRequest(params=SearchParams(query="mcp", limit=3))
        result = await client.session.send_request(request, SearchResult)
        print(result.items)
        # ['mcp-0', 'mcp-1', 'mcp-2']
```

* `Client(..., extensions=[advertise(EXTENSION_ID)])` declares the extension. The
  declarations become `ClientCapabilities.extensions`: on a 2026-07-28 connection
  the map travels in the per-request `_meta` envelope, so the server sees it on
  **every** request; on a legacy connection it rides the `initialize` handshake.
  Server code doesn't care which: `require_client_extension(ctx, ...)` and
  `ctx.session.check_client_capability(...)` read the right source on both paths.
* Vendor methods drop one layer to `client.session.send_request(...)`; `Client`
  only grows first-class methods for spec verbs. `send_request` accepts any
  `Request` subclass, so the vendor request passes as-is.

### Intercepting `tools/call`

The one interceptive hook. Override `intercept_tool_call` to observe, short-circuit,
or veto a tool call:

```python title="server.py" hl_lines="17-24"
# docs_src/extensions/tutorial005.py
import logging
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.extension import Extension
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolRequestParams

logger = logging.getLogger(__name__)


class AuditLog(Extension):
    """Observe every tools/call without touching its result."""

    identifier = "com.example/audit"

    async def intercept_tool_call(
        self,
        params: CallToolRequestParams,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        logger.info("tool %r called", params.name)
        return await call_next(ctx)


mcp = MCPServer("audited", extensions=[AuditLog()])


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

* `params` is the validated `CallToolRequestParams`: you get `params.name` and
  `params.arguments` without touching raw JSON. It is also what decides which
  tool call runs: passing a rewritten context through `call_next` changes what
  the handler observes on `ctx`, not the tool invocation. Wire-level request
  rewriting belongs to [Middleware](https://py.sdk.modelcontextprotocol.io/advanced/middleware/index.md).
* `call_next(ctx)` runs the rest of the chain and returns the handler's result.
  Return it unchanged (observe), return something else (replace), or raise an
  `MCPError` (refuse). Whatever you return is serialized like any handler
  result, including the 2026-era `serverInfo` identity stamp, so a
  short-circuiting interceptor never produces an anonymous or off-schema
  response.
* With several extensions, interceptors nest in registration order: the first
  extension in `extensions=[...]` is outermost.
* The default implementation is a pass-through, and a server whose extensions never
  override this hook keeps the bare `tools/call` handler untouched. You don't
  pay for what you don't use.

The hook wraps `tools/call` and nothing else. For every-message concerns, use
[Middleware](https://py.sdk.modelcontextprotocol.io/advanced/middleware/index.md). That is what it is for.

## Using a client extension

A **client extension** is the same contract from the consuming side: a bundle of
client-side behaviour behind one identifier. Pass instances to
`Client(extensions=[...])` and call tools normally:

```python title="client.py" hl_lines="66-68"
# docs_src/extensions/tutorial006.py
from collections.abc import Sequence
from typing import Any, Literal

import mcp.types as types
from mcp import Client
from mcp.client import ClaimContext, ClientExtension, ResultClaim
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.extension import Extension
from mcp.server.mcpserver import MCPServer, require_client_extension

EXTENSION_ID = "com.example/receipts"


class ReceiptResult(types.Result):
    """The claimed result shape; `result_type` pins the wire tag."""

    result_type: Literal["receipt"] = "receipt"
    receipt_token: str


class ReceiptIssuer(Extension):
    """Server half: answers `buy` with a receipt instead of a final result."""

    identifier = EXTENSION_ID

    async def intercept_tool_call(
        self,
        params: types.CallToolRequestParams,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        if params.name != "buy":
            return await call_next(ctx)
        require_client_extension(ctx, EXTENSION_ID)
        return {"resultType": "receipt", "receiptToken": "r-117"}


class Receipts(ClientExtension):
    """Client half: claims the `receipt` shape and supplies the code that finishes it."""

    identifier = EXTENSION_ID

    def claims(self) -> Sequence[ResultClaim[Any]]:
        return [ResultClaim(result_type="receipt", model=ReceiptResult, resolve=self._redeem)]

    async def _redeem(self, claimed: ReceiptResult, ctx: ClaimContext) -> types.CallToolResult:
        return await ctx.session.call_tool("redeem", {"token": claimed.receipt_token})


mcp = MCPServer("shop", extensions=[ReceiptIssuer()])


@mcp.tool()
def buy(item: str) -> types.CallToolResult:
    """Buy an item."""
    raise NotImplementedError  # ReceiptIssuer answers `buy` before the tool runs


@mcp.tool()
def redeem(token: str) -> str:
    """Exchange a receipt token for the goods."""
    return f"goods for {token}"


async def main() -> None:
    async with Client(mcp, extensions=[Receipts()]) as client:
        result = await client.call_tool("buy", {"item": "lamp"})
        print(result.content)
        # [TextContent(text='goods for r-117')]
```

`call_tool("buy", ...)` returns a plain `CallToolResult`, like every other call. What
the extension changed: the server may now answer `buy` with a `receipt` **result
shape** instead of a final result, and `Receipts` finishes it (here by redeeming the
receipt with a follow-up call) before `call_tool` returns. Nothing about the call
site moves.

Drop the extension and none of this exists: the server's gate refuses a client
that did not declare it (error -32021), and a claimed shape from a server that
skips the gate fails validation, exactly as the spec requires for an
unrecognized `resultType`. Off by default, on both ends of the wire.

To advertise an identifier with **no** client-side behaviour (the server gates on
the capability, the client does nothing, as in the search client above), use
`advertise()`:

```python
from mcp.client import advertise

client = Client(mcp, extensions=[advertise("com.example/search")])
```

## Writing a client extension

Subclass `ClientExtension` and override only what you need. Three contribution
kinds, each with a default: `settings()`, `claims()`, and `notifications()`.

```python title="client.py" hl_lines="17-18 43-44 46-47"
# docs_src/extensions/tutorial006.py
from collections.abc import Sequence
from typing import Any, Literal

import mcp.types as types
from mcp import Client
from mcp.client import ClaimContext, ClientExtension, ResultClaim
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.extension import Extension
from mcp.server.mcpserver import MCPServer, require_client_extension

EXTENSION_ID = "com.example/receipts"


class ReceiptResult(types.Result):
    """The claimed result shape; `result_type` pins the wire tag."""

    result_type: Literal["receipt"] = "receipt"
    receipt_token: str


class ReceiptIssuer(Extension):
    """Server half: answers `buy` with a receipt instead of a final result."""

    identifier = EXTENSION_ID

    async def intercept_tool_call(
        self,
        params: types.CallToolRequestParams,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        if params.name != "buy":
            return await call_next(ctx)
        require_client_extension(ctx, EXTENSION_ID)
        return {"resultType": "receipt", "receiptToken": "r-117"}


class Receipts(ClientExtension):
    """Client half: claims the `receipt` shape and supplies the code that finishes it."""

    identifier = EXTENSION_ID

    def claims(self) -> Sequence[ResultClaim[Any]]:
        return [ResultClaim(result_type="receipt", model=ReceiptResult, resolve=self._redeem)]

    async def _redeem(self, claimed: ReceiptResult, ctx: ClaimContext) -> types.CallToolResult:
        return await ctx.session.call_tool("redeem", {"token": claimed.receipt_token})


mcp = MCPServer("shop", extensions=[ReceiptIssuer()])


@mcp.tool()
def buy(item: str) -> types.CallToolResult:
    """Buy an item."""
    raise NotImplementedError  # ReceiptIssuer answers `buy` before the tool runs


@mcp.tool()
def redeem(token: str) -> str:
    """Exchange a receipt token for the goods."""
    return f"goods for {token}"


async def main() -> None:
    async with Client(mcp, extensions=[Receipts()]) as client:
        result = await client.call_tool("buy", {"item": "lamp"})
        print(result.content)
        # [TextContent(text='goods for r-117')]
```

* The identifier follows the same grammar as the server's, validated when the class
  is defined.
* `claims()` returns `ResultClaim`s: a wire tag, the model that parses it, and the
  resolver that finishes it. The model must pin the tag with
  `result_type: Literal["receipt"]` and must not subclass the verb's core result
  types; both are enforced when the claim is constructed. Vendor fields like
  `receipt_token` ride the wire as-is: a substituted shape reaches the client
  verbatim.
* The resolver receives the parsed model and a `ClaimContext`; `ctx.session` is the
  same public handle as `client.session`, so follow-ups are ordinary session calls.
  It returns the verb's normal `CallToolResult`.
* `settings()` is the value advertised at `ClientCapabilities.extensions[identifier]`,
  read once at `Client` construction.

`notifications()` declares vendor server notifications to observe:

```python
def notifications(self) -> Sequence[NotificationBinding[Any]]:
    return [NotificationBinding(method="notifications/receipts", params_type=ReceiptEvent, handler=self.on_receipt)]
```

The handler receives validated params one at a time, in dispatch order. It observes; it cannot veto
or reply.

Two quiet rules. Claims are active on 2026-07-28 connections only, and the capability
ad follows them: on a legacy connection the claims dissolve and the identifier drops
out of the ad with them, so the client never advertises an extension whose shapes it
would reject. And when you want the claimed shape yourself instead of the resolver,
call `client.session.call_tool(..., allow_claimed=True)`; without that flag, a
claimed shape reaching a session-tier caller raises `UnexpectedClaimedResult`.

### Extension verbs

An extension's own request methods need no client-side registration. A vendor request
type subclasses `mcp.types.Request` and goes through `client.session.send_request`,
as in [Serving your own methods](#serving-your-own-methods). One addition: when a
params key must ride the `Mcp-Name` header (extension specs such as tasks require
this for their verbs), the request type declares `name_param`:

```python title="client.py" hl_lines="22-25 46-47"
# docs_src/extensions/tutorial007.py
from collections.abc import Sequence
from typing import Any, Literal

import mcp.types as types
from mcp import Client
from mcp.client import advertise
from mcp.server.context import ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import MCPServer

EXTENSION_ID = "com.example/jobs"


class JobParams(types.RequestParams):
    job_id: str


class JobStatus(types.Result):
    status: str


class JobStatusRequest(types.Request[JobParams, Literal["com.example/jobs.status"]]):
    method: Literal["com.example/jobs.status"] = "com.example/jobs.status"
    params: JobParams
    name_param = "jobId"  # params["jobId"] rides the Mcp-Name header


async def job_status(ctx: ServerRequestContext[Any, Any], params: JobParams) -> JobStatus:
    return JobStatus(status=f"{params.job_id} is running")


class Jobs(Extension):
    """An extension whose verb names its subject, so the header can route on it."""

    identifier = EXTENSION_ID

    def methods(self) -> Sequence[MethodBinding]:
        return [MethodBinding("com.example/jobs.status", JobParams, job_status)]


mcp = MCPServer("worker", extensions=[Jobs()])


async def main() -> None:
    async with Client(mcp, extensions=[advertise(EXTENSION_ID)]) as client:
        request = JobStatusRequest(params=JobParams(job_id="job-7"))
        result = await client.session.send_request(request, JobStatus)
        print(result.status)
        # job-7 is running
```

The session mirrors `params["jobId"]` into `Mcp-Name` on every send path, and a
missing value fails loudly rather than silently omitting a required header.

## What an extension cannot do

The contribution surface is **closed** on purpose. On the server: settings, tools,
resources, methods, one `tools/call` interceptor. On the client: settings, result
claims, notification bindings. An extension cannot:

* **Reach into the host.** It declares data; it holds no server or client reference.
* **Replace core behaviour.** Spec methods and core result tags are rejected at
  construction (`initialize` is reserved by the runner outright); a notification
  binding shadowed by core vocabulary goes quiet with a warning instead.
* **Register late.** After `MCPServer(...)` or `Client(...)` returns, the extension
  set is what it is.

If you are fighting these walls, you are not writing an extension. You are writing
a fork. The walls are the feature: a user reading `extensions=[Apps(), Stamps()]`
knows *everything* those two can have touched.
