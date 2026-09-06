# Subscriptions

A server's catalog is not fixed. Tools appear at runtime, and the content behind a resource URI changes.

**Subscriptions** are how a client hears about it. The client sends one `subscriptions/listen` request, and the response to that request *is* the stream: it stays open and carries the change notifications the client asked for.

## Publish it from the tool

Your side of it is one line: publish the change.

```python title="server.py" hl_lines="20 32"
# docs_src/subscriptions/tutorial001.py
from mcp.server.mcpserver import Context, MCPServer

mcp = MCPServer("Sprint Board")

BOARDS = {
    "sprint": {"design": False, "build": False, "ship": False},
    "backlog": {"tidy docs": False},
}


@mcp.resource("board://{name}")
def board(name: str) -> str:
    tasks = BOARDS[name]
    return "\n".join(f"[{'x' if done else ' '}] {task}" for task, done in tasks.items())


@mcp.tool()
async def complete_task(board: str, task: str, ctx: Context) -> str:
    BOARDS[board][task] = True
    await ctx.notify_resource_updated(f"board://{board}")
    return f"{task}: done"


def sprint_report() -> str:
    done = sum(done for tasks in BOARDS.values() for done in tasks.values())
    return f"{done} task(s) done"


@mcp.tool()
async def enable_reports(ctx: Context) -> str:
    mcp.add_tool(sprint_report)
    await ctx.notify_tools_changed()
    return "reporting is live"
```

* `await ctx.notify_resource_updated("board://sprint")` reaches every open stream that subscribed to that URI. Nobody else.
* `await ctx.notify_tools_changed()` reaches every stream that asked for tool-list changes. A client that receives it calls `tools/list` again, and now sees `sprint_report`.
* The siblings are `notify_prompts_changed()` and `notify_resources_changed()`.
* No subscribers, no work. Publishing to an idle server is a no-op, so you never check whether anyone is listening. You state what changed.

`MCPServer` serves `subscriptions/listen` for you. The wire obligations (the acknowledgment as the first frame, per-stream filtering, the subscription id on every frame) are the SDK's job.

!!! check
    On the wire, a stream whose filter named `board://sprint` looks like this after `complete_task` runs:

    ```json
    {"method": "notifications/subscriptions/acknowledged",
     "params": {"notifications": {"resourceSubscriptions": ["board://sprint"]}, "_meta": {"io.modelcontextprotocol/subscriptionId": "listen-1"}}}

    {"method": "notifications/resources/updated",
     "params": {"uri": "board://sprint", "_meta": {"io.modelcontextprotocol/subscriptionId": "listen-1"}}}
    ```

    Note what the update does *not* carry: the board. Every frame carries the listen request's JSON-RPC id under `_meta`, and that id is the subscription id. The client mints it: the Python `Client` uses strings like `"listen-1"`; other clients may use integers.

## Only what was asked for

The filter is a contract. A stream that requested tool-list changes and one resource URI receives those two kinds and nothing else. Publish a prompt change and that stream stays silent.

`MCPServer` matches resource URIs as exact strings, so a stream that named `board://sprint` hears nothing about `board://sprint/tasks/1`. The spec lets a server report a change on a sub-resource of a subscribed URI; `MCPServer` never does, but clients are built to expect it.

Two things the stream is *not*:

* **It is not a replay log.** A dropped stream is gone, and events published while nobody was connected are not queued. Clients re-listen and refetch.
* **It is not the 2025 path.** Clients that called `resources/subscribe` are served by `ctx.session.send_resource_updated(uri)`. The `notify_*` methods reach `subscriptions/listen` streams only.

## Deciding who may watch

By default every requested kind and URI is honored: any caller may watch any URI you publish. Nothing consults your read handler, because nobody is reading — a caller your `files://{name}` handler would turn away can still open a stream on `files://payroll.csv` and learn that it changed, and when. It never learns content, and it cannot probe what exists, because an unknown URI is honored too and simply never fires. Narrow but real, so gate it before you publish per-user URIs from a multi-tenant server.

The gate is a middleware. It sees the `subscriptions/listen` request before the SDK acknowledges it and refuses when the caller asks for anything they may not read:

```python title="server.py" hl_lines="19-26 29"
# docs_src/subscriptions/tutorial006.py
from mcp_types import INVALID_REQUEST, SubscriptionsListenRequestParams

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError

# Who may see each file. Replace this table with a database or your RBAC system.
ACCESS = {
    "files://report.pdf": {"alice", "bob"},
    "files://payroll.csv": {"carol"},
}


def can_access(user: str | None, uri: str) -> bool:
    return user is not None and user in ACCESS.get(uri, set())


async def gate_subscriptions(ctx: ServerRequestContext, call_next: CallNext) -> HandlerResult:
    if ctx.method == "subscriptions/listen":
        params = SubscriptionsListenRequestParams.model_validate(ctx.params or {}, by_name=False)
        token = get_access_token()
        user = token.subject if token else None
        if not all(can_access(user, uri) for uri in params.notifications.resource_subscriptions or ()):
            raise MCPError(INVALID_REQUEST, "not permitted to watch the requested resources")
    return await call_next(ctx)


mcp = MCPServer("Reports", middleware=[gate_subscriptions])


@mcp.resource("files://{name}")
def file(name: str) -> str:
    uri = f"files://{name}"
    token = get_access_token()
    if not can_access(token.subject if token else None, uri):
        raise MCPError(INVALID_REQUEST, f"Unknown resource: {uri}")
    return f"contents of {name}"
```

* `ctx.params` is the raw request, so the middleware validates it into `SubscriptionsListenRequestParams` itself and reads the filter the client asked for.
* Refusal is a raised `MCPError` before `call_next(ctx)`: the client gets that error and no stream, and the connection carries on. Keep the message uniform, naming no URI, so a refusal never confirms which URIs are protected.
* One `can_access(user, uri)` answers both questions. The resource handler asks it on `resources/read`; the middleware asks it on `subscriptions/listen`. Swap the table for a database or your RBAC system and both stay in step.
* The decision holds for the stream's lifetime. There is no per-event re-check, so if a caller's access can lapse mid-stream (an expiring token), end that caller's connection when it does.

The full middleware contract, including what else it wraps and why it is marked provisional, is on **[Middleware](https://py.sdk.modelcontextprotocol.io/advanced/middleware/index.md)**.

## The client end

Here is a client on the other side of that stream, following the board:

```python title="client.py" hl_lines="15"
# docs_src/subscriptions/tutorial003.py
from mcp import Client
from mcp.client.subscriptions import ResourceUpdated, ToolsListChanged
from mcp.types import TextResourceContents

BOARD = "board://sprint"


async def read_board(client: Client, uri: str = BOARD) -> str:
    [contents] = (await client.read_resource(uri)).contents
    assert isinstance(contents, TextResourceContents)
    return contents.text


async def follow_board(client: Client) -> None:
    async with client.listen(tools_list_changed=True, resource_subscriptions=[BOARD]) as sub:
        async for event in sub:
            match event:
                case ResourceUpdated(uri=uri):
                    print(await read_board(client, uri))
                case ToolsListChanged():
                    tools = await client.list_tools()
                    print("tools:", [tool.name for tool in tools.tools])
                case _:
                    pass  # kinds the filter did not ask for never arrive


async def main() -> None:
    async with Client("http://localhost:8000/mcp") as client:
        await follow_board(client)
```

Entering `client.listen(...)` sends the request and waits for your acknowledgment, so the stream is live when the block starts, and each typed event is a cue to refetch, never a payload. That is the whole contract in one screen. Everything else about the client end lives on its own page: watching beside a main flow, stream endings, and re-listening. See **[Subscriptions](https://py.sdk.modelcontextprotocol.io/client/subscriptions/index.md)** under *Clients*.

## Scaling past one process

Publishes travel from your handler to the open streams over a `SubscriptionBus`. The default is in-memory: one process, every stream in it. That is the right answer until you run replicas behind a load balancer, because then a client's stream is pinned to one replica, and a publish on another replica has to reach it.

That seam is yours to implement: two methods over your pub/sub backend.

```python
from collections.abc import Callable

from redis.asyncio import Redis

from mcp.server.mcpserver import MCPServer
from mcp.server.subscriptions import ServerEvent  # SubscriptionBus is a Protocol: no base class


class RedisSubscriptionBus:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._listeners: dict[object, Callable[[ServerEvent], None]] = {}

    async def publish(self, event: ServerEvent) -> None:
        await self._redis.publish("mcp-events", encode(event))  # to every replica

    def subscribe(self, listener: Callable[[ServerEvent], None]) -> Callable[[], None]:
        token = object()
        self._listeners[token] = listener

        def unsubscribe() -> None:
            self._listeners.pop(token, None)

        return unsubscribe


mcp = MCPServer("Sprint Board", subscriptions=RedisSubscriptionBus(redis))
```

`encode` is yours, and so is the reader task on each replica that decodes arriving messages and calls every registered listener. Listeners are synchronous, must not raise, and run on the server's event loop.

The bus carries typed `ServerEvent` values, four small dataclasses, never JSON-RPC. Stamping, filtering, and stream lifecycles stay in the SDK, so a bus implementation cannot break the protocol. It can only move events between processes.

To publish from outside a request, construct the bus yourself so you hold the reference. `MCPServer` builds one internally when you pass nothing, and does not expose it.

```python
from mcp.server.subscriptions import InMemorySubscriptionBus, ToolsListChanged

bus = InMemorySubscriptionBus()
mcp = MCPServer("Sprint Board", subscriptions=bus)


async def tools_reloaded() -> None:
    await bus.publish(ToolsListChanged())  # from a lifespan task, a webhook, anywhere
```

## The low-level composition

Down on the low-level `Server` there is no pre-wired anything, and the same parts assemble in three lines:

```python title="server.py" hl_lines="8-9 47"
# docs_src/subscriptions/tutorial002.py
from typing import Any

import mcp.types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ResourceUpdated

bus = InMemorySubscriptionBus()
listen_handler = ListenHandler(bus)

BOARD = {"design": False, "build": False}

COMPLETE_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"task": {"type": "string"}},
    "required": ["task"],
}


async def read_resource(
    ctx: ServerRequestContext[Any], params: types.ReadResourceRequestParams
) -> types.ReadResourceResult:
    board = "\n".join(f"[{'x' if done else ' '}] {task}" for task, done in BOARD.items())
    return types.ReadResourceResult(contents=[types.TextResourceContents(uri=params.uri, text=board)])


async def list_tools(
    ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[types.Tool(name="complete_task", description="Mark a task done.", input_schema=COMPLETE_TASK_SCHEMA)]
    )


async def call_tool(ctx: ServerRequestContext[Any], params: types.CallToolRequestParams) -> types.CallToolResult:
    args = params.arguments or {}
    BOARD[args["task"]] = True
    await bus.publish(ResourceUpdated(uri="board://sprint"))
    return types.CallToolResult(content=[types.TextContent(type="text", text="done")])


server = Server(
    "sprint-board",
    on_read_resource=read_resource,
    on_list_tools=list_tools,
    on_call_tool=call_tool,
    on_subscriptions_listen=listen_handler,
)
```

* You own the bus, so you publish to it directly: `await bus.publish(ResourceUpdated(uri=...))`. Put it wherever your handlers can reach it: module scope here, the lifespan in a bigger app.
* `ListenHandler(bus)` is the same handler `MCPServer` registers, and `on_subscriptions_listen=` is an ordinary handler slot. Put your own callable in that slot for different semantics, and the spec obligations move to you: acknowledge first, stamp every frame with the subscription id, deliver nothing outside the filter.
* `ListenHandler.close()` ends every open stream gracefully. Each one receives the listen request's result as its final frame, which is the spec's way of saying the server ended the subscription deliberately. It returns before those streams finish flushing, so give them a moment before you tear the transport down. Without it, streams end when the client disconnects.

## Recap

* A client opts in with one `subscriptions/listen` request, and the response is the stream. Serving it is built in.
* You publish with `ctx.notify_*`, and the SDK does the stamping, filtering, and lifecycle work.
* Events are cues, not payloads. Both ends refetch.
* The client end is `async with client.listen(...)`: **[Subscriptions](https://py.sdk.modelcontextprotocol.io/client/subscriptions/index.md)** under *Clients* is that story.
* On the low-level `Server` you assemble the same parts yourself: a bus, `ListenHandler(bus)`, the `on_subscriptions_listen` slot.
* Scaling out means implementing `SubscriptionBus`, two methods, and passing it as `MCPServer(subscriptions=...)`.

Running the server that serves all this, behind one replica or twenty, is **[Deploy & scale](https://py.sdk.modelcontextprotocol.io/run/deploy/index.md)**.
