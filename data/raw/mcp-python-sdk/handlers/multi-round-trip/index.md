# Multi-round-trip requests

Sometimes a tool can't finish in one round trip. It needs something only the user has: a choice, a confirmation, a credential.

Before 2026-07-28 the server got it by calling **back**: opening its own request to the client (an elicitation, a sampling call) in the middle of handling the original one. The 2026-07-28 spec retires that back-channel.

Instead, the server **returns**.

## Return, don't call back

The server answers `tools/call` with an **`InputRequiredResult`** instead of a `CallToolResult`. Two of its fields do the work:

* **`input_requests`**: what the server still needs, as a dict keyed by names the server chose. Each value is an `ElicitRequest`, a `CreateMessageRequest`, or a `ListRootsRequest`.
* **`request_state`**: an opaque token. The client echoes it back verbatim on the retry. Your server is the only thing that reads it.

The client fulfils each request, then calls the **same tool again**, carrying its answers in `input_responses` and the token in `request_state`. The server now has what it was missing and returns a normal `CallToolResult`.

That's the whole protocol. Every leg is an ordinary request from the client to the server. Nothing ever flows the other way.

## The server side

On `@mcp.tool()` you rarely build this by hand: declare a dependency that asks the user (`Elicit`), samples the client's LLM (`Sample`), or lists its roots (`ListRoots`) and the SDK returns the `InputRequiredResult` for you; that form is the **[Dependencies](https://py.sdk.modelcontextprotocol.io/handlers/dependencies/index.md)** page. The two forms don't mix: a call has one `input_responses`/`request_state` channel, so a tool that uses `Resolve(...)` parameters cannot also return `InputRequiredResult` from its body. A declared `InputRequiredResult` return is rejected at registration (`InvalidSignature`), and an undeclared one fails the call at runtime. The manual form is the **low-level** `Server`, whose `on_call_tool` handler is allowed to return either result type:

```python title="server.py" hl_lines="43-46"
# docs_src/mrtr/tutorial001.py
from mcp.server import Server, ServerRequestContext
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

ASK_REGION = ElicitRequest(
    params=ElicitRequestFormParams(
        message="Which region should the database live in?",
        requested_schema={
            "type": "object",
            "properties": {"region": {"type": "string"}},
            "required": ["region"],
        },
    )
)


async def list_tools(ctx: ServerRequestContext, params: PaginatedRequestParams | None) -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name="provision",
                description="Provision a database. Asks which region to put it in.",
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            )
        ]
    )


async def call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult | InputRequiredResult:
    answer = (params.input_responses or {}).get("region")
    if not isinstance(answer, ElicitResult) or answer.content is None:
        return InputRequiredResult(input_requests={"region": ASK_REGION}, request_state="provision-v1")
    name = (params.arguments or {})["name"]
    text = f"Provisioned {name!r} in {answer.content['region']}."
    return CallToolResult(content=[TextContent(type="text", text=text)])


server = Server("Provisioner", on_list_tools=list_tools, on_call_tool=call_tool)
```

* `on_call_tool` is typed `-> CallToolResult | InputRequiredResult`. Returning the second one is the entire server-side API.
* On the first call `params.input_responses` is `None`, so the guard fires and the handler asks instead of answering.
* On the retry, the `ElicitResult` the client sent is sitting under the **same key** (`"region"`) that the server used in `input_requests`.

Everything else in that file (the explicit `input_schema`, the hand-built `CallToolResult`) is the ordinary low-level `Server`, covered in **[The low-level Server](https://py.sdk.modelcontextprotocol.io/advanced/low-level-server/index.md)**. This page only adds the second return type.

## Beyond tools

`tools/call` is not special: at 2026-07-28 a server may answer `prompts/get` and `resources/read` the same way. On `MCPServer`, an `@mcp.prompt()` function — or an `@mcp.resource()` **template** function — returns the `InputRequiredResult` itself and reads the retry's answers off the context:

```python title="server.py" hl_lines="20 22 24"
# docs_src/mrtr/tutorial004.py
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.prompts.base import UserMessage
from mcp.types import ElicitRequest, ElicitRequestFormParams, ElicitResult, InputRequiredResult

mcp = MCPServer("Briefing")

ASK_AUDIENCE = ElicitRequest(
    params=ElicitRequestFormParams(
        message="Who is the briefing for?",
        requested_schema={
            "type": "object",
            "properties": {"audience": {"type": "string"}},
            "required": ["audience"],
        },
    )
)


@mcp.prompt()
async def briefing(ctx: Context) -> list[UserMessage] | InputRequiredResult:
    """Draft a briefing tuned to its audience."""
    answer = (ctx.input_responses or {}).get("audience")
    if not isinstance(answer, ElicitResult) or answer.content is None:
        return InputRequiredResult(input_requests={"audience": ASK_AUDIENCE})
    return [UserMessage(f"Write a briefing for {answer.content['audience']}.")]
```

* The first round returns the `InputRequiredResult`. On the retry, `ctx.input_responses` holds the answers under the same keys and the function returns its ordinary result — prompt messages here, resource content for a template resource.
* A `request_state` you set is sealed before it crosses the wire and verified on the echo, like everything else on the server; **[Protecting `requestState`](#protecting-requeststate)** below covers what the seal gives you and when you need to configure keys.
* An `@mcp.tool()` function can return the result directly the same way, when the dependency form doesn't fit.
* Static `@mcp.resource()` functions don't participate: they take no `Context`, so they could never read the retry. Only template resources can ask.
* The era rules below apply unchanged: returning an `InputRequiredResult` on a pre-2026 session is the same `-32603` the warning describes.

## The client side

`Client` runs the loop for you.

Register the callbacks the server might ask for (`elicitation_callback`, `sampling_callback`, `list_roots_callback`) and call the tool. When an `InputRequiredResult` arrives, `Client` dispatches each entry in `input_requests` to the matching callback, retries with the answers and the echoed `request_state`, and keeps going until a `CallToolResult` comes back:

```python title="client.py" hl_lines="11 12"
# docs_src/mrtr/tutorial003.py
from mcp import Client
from mcp.client import ClientRequestContext
from mcp.types import ElicitRequestParams, ElicitResult


async def handle_elicitation(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
    return ElicitResult(action="accept", content={"region": "eu-west-1"})


async def main() -> None:
    async with Client("http://127.0.0.1:8000/mcp", elicitation_callback=handle_elicitation) as client:
        result = await client.call_tool("provision", {"name": "orders"})
        print(result.content)
```

* That `elicitation_callback` is the same one a pre-2026 server's back-channel `elicitation/create` would have hit. The same is true of `sampling_callback` for `sampling/createMessage` and `list_roots_callback` for `roots/list`: at 2026-07-28 the standalone server->client RPCs are gone, but the identical `ElicitRequest` / `CreateMessageRequest` / `ListRootsRequest` payloads ride inside `input_requests` and dispatch to the same three callbacks. One set of callbacks serves both eras.
* `call_tool` returns a plain `CallToolResult`. The intermediate rounds are invisible to the caller.
* `get_prompt` and `read_resource` drive the same loop.

!!! check
    Leave the callback off and the loop fails on the first round: the SDK's stand-in callback
    answers every elicitation with an error, and `call_tool` raises `MCPError` with the message
    *"Elicitation not supported"*.

The loop is bounded. `Client(..., input_required_max_rounds=10)` is the default cap; a server that keeps returning `InputRequiredResult` past it makes `call_tool` raise. If a round carries only `request_state` and no `input_requests`, `Client` sleeps briefly (50ms doubling to a 250ms ceiling) before retrying, so a server that is just saying *"not done yet"* isn't busy-polled.

### Driving the loop yourself

The auto-loop is enough for a single-process client. Own the loop instead when:

* Your client is **distributed**: the process that renders the question to the user is not the process that called `call_tool`, so a different worker issues the retry. `request_state` is the persistable token you carry across that boundary, through your own storage, and `input_responses` is what the other side sends back with it.
* You want to **inspect** each round: log or audit every `input_requests` entry, refuse certain request kinds, or apply your own backoff between legs.
* You want a **wall-clock** bound rather than a round-count bound: wrap your own loop in `anyio.fail_after(...)` instead of relying on `input_required_max_rounds`.

Drop to the underlying session, where `allow_input_required=True` hands you the union directly:

```python title="client.py" hl_lines="12 13 19"
# docs_src/mrtr/tutorial002.py
from mcp import Client
from mcp.types import CallToolResult, ElicitRequest, ElicitResult, InputRequest, InputRequiredResult, InputResponse


def fulfil(request: InputRequest) -> InputResponse:
    if not isinstance(request, ElicitRequest):
        raise NotImplementedError(f"this client cannot answer a {request.method!r} request")
    return ElicitResult(action="accept", content={"region": "eu-west-1"})


async def provision(client: Client, name: str) -> CallToolResult:
    result = await client.session.call_tool("provision", {"name": name}, allow_input_required=True)
    while isinstance(result, InputRequiredResult):
        responses = {key: fulfil(request) for key, request in (result.input_requests or {}).items()}
        result = await client.session.call_tool(
            "provision",
            {"name": name},
            input_responses=responses,
            request_state=result.request_state,
            allow_input_required=True,
        )
    return result
```

* `client.session.call_tool(..., allow_input_required=True)` widens the return type to `CallToolResult | InputRequiredResult`. The `isinstance` is what narrows it back.
* `request_state` is now in your hands. Write it down between legs and the conversation can resume from a fresh process.
* For every entry in `input_requests` you put an `InputResponse` under the **same key** in `input_responses`. `fulfil` is where your UI goes; this one hard-codes the answer.
* Same tool name, same `arguments`, every leg. The retry is the original call carried out again, not a new method.

## Protecting `requestState`

Everything above treats `request_state` as an echo, and on the wire that is all it is. But the client holds it between legs (writing it down across processes is exactly what the previous section blessed), so what comes back is **client-supplied input**: it can be modified, expired, or lifted from a different call entirely. The spec requires servers to integrity-protect this state and reject the round when verification fails, whenever the state can influence authorization, resource access, or business logic.

`MCPServer` protects it by default. Every server seals outgoing `requestState` and verifies every echo — resolver state and hand-built state alike — under a key generated at process start. You configure nothing, write plaintext, and read plaintext; the wire only ever carries an opaque encrypted token.

The default key lives and dies with the process, which is the one thing you must know before deploying beyond a single process:

```python
from mcp.server.mcpserver import MCPServer, RequestStateSecurity

# Multi-instance or restart-surviving: one or more shared secret keys (>= 32 bytes each).
mcp = MCPServer("fleet", request_state_security=RequestStateSecurity(keys=[key]))
```

* **The default (no configuration)** suits a single process: stdio, or exactly one HTTP worker. A retry that lands on a different worker, a different instance behind a load balancer, or the same server after a restart is sealed under a key that process doesn't have — the client gets the frozen rejection below and must start the flow over.
* **`keys=[...]`** is required whenever a retry can reach a **different instance** (multi-worker `uvicorn`, load-balanced HTTP) or must survive restarts: every instance verifies what any sibling minted. Same machinery, your secret instead of a generated one.
* For your own crypto, such as a KMS or an existing token service, pass `RequestStateSecurity(codec=...)` instead of `keys`; **[Bring your own crypto](#bring-your-own-crypto)** below covers the contract.

### What the seal carries

Default or configured, `requestState` on the wire is an encrypted, authenticated token. Your code never sees it: handlers and resolvers write plaintext and read plaintext (`ctx.request_state`); the SDK seals on the way out and verifies on the way in. Beyond integrity, each token is bound to:

* **A time window.** Every round re-seals with a fresh expiry, so `RequestStateSecurity(ttl=...)` (default 600 seconds) bounds per-round think time, not the whole flow.
* **The authenticated principal.** When the request carries an OAuth access token the SDK validated, the state is bound to the token's client, issuer, and subject: state minted for one user fails under another, even when both users share one OAuth client. A verifier that supplies no subject degrades the binding to the client identity alone, which under URL-based client IDs is shared by every user of that client software. When auth is terminated outside the SDK (a fronting proxy), or the transport is unauthenticated, there is no principal to bind and this check is inert, unless `RequestStateSecurity(bind_principal=...)` supplies one from your own identity signal. Whichever components your token verifier supplies, it must supply them consistently: a verifier that includes the subject on some requests and omits it on others changes the principal mid-flow, and in-flight rounds are rejected.
* **The originating request.** The method, the tool or prompt name (or resource URI), and a digest of the arguments. A token replayed against a different tool, different arguments, or a different method fails.
* **The exact question asked.** Every resolver answer is pinned to the rendered question the client was shown, both on the round it first arrives and when a recorded answer is reused later. Redeploy with a reworded message or a changed schema and the server re-asks instead of consuming a stale answer. The same pinning cuts the other way: derive messages from the tool's arguments, not from per-call data. A message built from a timestamp or a live rate renders differently every round, so every recorded answer looks stale and the server re-asks until the client's round limit ends the call.

All of that is the SDK's job, not yours, and not the codec's if you bring your own.

### Rotating keys

`keys[0]` seals new state; every key in the list verifies. Zero-downtime rotation is three phases, each fully rolled out before the next:

```python
RequestStateSecurity(keys=[OLD, NEW])  # 1: every instance learns to verify NEW; OLD still mints
RequestStateSecurity(keys=[NEW, OLD])  # 2: NEW mints; in-flight OLD state keeps verifying
RequestStateSecurity(keys=[NEW])       # 3: one ttl after phase 2 is fully out, retire OLD
```

Never promote the minter first: minting under a key some instance can't yet verify drops in-flight rounds mid-rollout.

Keys are scoped to one service. The sealed envelope also carries the server's name as an audience claim, so a token minted by a different service that happens to share a secret is rejected anyway. The claim is only as distinctive as the name, so a server given an explicit policy must have a real name or set `RequestStateSecurity(audience=...)` — an unnamed one raises at construction. `audience=` also serves deliberate multi-service topologies where one service must accept state another minted. (The no-configuration default is exempt: its key never leaves the process, so the audience claim has nothing to add.)

### Bring your own crypto

`RequestStateSecurity(codec=...)` takes anything with `seal(bytes) -> str` and `unseal(str) -> bytes` that raises `InvalidRequestState` for any token it did not mint. The classic shape is envelope encryption against a KMS, where you unwrap a data key once at startup and keep the per-token crypto local:

```python title="server.py" hl_lines="12 26-27 34-35 38"
# docs_src/mrtr/tutorial005.py
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mcp.server import MCPServer
from mcp.server.mcpserver import InvalidRequestState, RequestStateSecurity

PREFIX = "kms1."  # format version; fed to GCM as associated data, so it is bound under the tag


def unwrap_data_key() -> bytes:
    """One KMS call at process start, kms.decrypt(CiphertextBlob=...); every token after that is local crypto."""
    return os.urandom(32)  # stand-in for the unwrapped 32-byte data key


class EnvelopeCodec:
    def __init__(self, data_key: bytes) -> None:
        self._aesgcm = AESGCM(data_key)

    def seal(self, payload: bytes) -> str:
        nonce = os.urandom(12)
        return PREFIX + (nonce + self._aesgcm.encrypt(nonce, payload, PREFIX.encode())).hex()

    def unseal(self, token: str) -> bytes:
        if not token.startswith(PREFIX):
            raise InvalidRequestState("unknown token format")
        body = token[len(PREFIX) :]
        try:
            raw = bytes.fromhex(body)
            if raw.hex() != body:  # only the exact string seal() produced verifies
                raise ValueError("non-canonical hex")
            return self._aesgcm.decrypt(raw[:12], raw[12:], PREFIX.encode())
        except (ValueError, InvalidTag) as exc:
            raise InvalidRequestState("token failed verification") from exc


mcp = MCPServer("Deployer", request_state_security=RequestStateSecurity(codec=EnvelopeCodec(unwrap_data_key())))
```

TTL, principal binding, and request binding are **not** the codec's job: the SDK stamps them into the payload before `seal` and re-verifies them after `unseal`, for every codec. A codec's only obligations are integrity (tampered means raise) and, ideally, confidentiality.

### When verification fails

Every inbound failure, whether tampered, expired, replayed against a different request or principal, or sealed under a key this server doesn't know, gets the same answer:

```json
{"code": -32602, "message": "Invalid or expired requestState"}
```

One frozen message for every cause, so the wire never reveals which check failed; the real reason goes to the server log. Every inbound `requestState` on `tools/call`, `prompts/get`, and `resources/read` is checked, including one arriving for a handler that never mints state. The most common rejection in practice isn't an attacker — it's the default process-local key meeting a retry from before a restart or from another instance; the client restarts the flow, and `keys=[...]` is the fix when that matters.

### Hand-built state

A `request_state` you set yourself (returning `InputRequiredResult` from a tool, prompt, or resource-template function) is sealed and verified by the same machinery as resolver state, with zero code changes: write plaintext, read plaintext, and every binding above applies.

The one thing the SDK cannot pin for you, even when configured, is question identity: it doesn't know which of *your* questions an answer in your state belongs to. If you store answers keyed by question, include your own question identifier in the state and check it on the retry.

The low-level `Server` is the no-batteries tier: unlike `MCPServer`, nothing is sealed until you append the boundary yourself, and your `request_state` crosses the wire exactly as written until you do. The one-line opt-in is shown in **[The low-level Server](https://py.sdk.modelcontextprotocol.io/advanced/low-level-server/index.md#the-other-handlers)**.

## A 2026-07-28 result

`InputRequiredResult` only exists at protocol version **2026-07-28**. The in-memory `Client(server)` negotiates it for you; over the wire, `mode="auto"` discovers it. After connecting, `client.protocol_version` tells you what you got.

!!! warning
    A pre-2026 session has nowhere to put an `InputRequiredResult`. Return one from your handler on a
    `mode="legacy"` connection and the runner cannot serialize it into the negotiated version; the
    client gets back a `-32603` *"Handler returned an invalid result"* error. A server that serves
    both eras must check `ctx.protocol_version` before reaching for it.

!!! info
    **URL-mode elicitation** rides this exact mechanism on a 2026 connection. The entry in
    `input_requests` is an `ElicitRequest` whose params are `ElicitRequestURLParams`; the user
    finishes the out-of-band flow and your client retries the call. Same loop, no new API. The
    high-level server half is in **[Elicitation](https://py.sdk.modelcontextprotocol.io/handlers/elicitation/index.md)**.

## Recap

* At 2026-07-28 a server that needs input mid-call **returns** an `InputRequiredResult`. It never opens a request to the client.
* `input_requests` is what it needs. `request_state` is an opaque resume token only the server reads.
* `Client` runs the retry loop for you: register `elicitation_callback` / `sampling_callback` / `list_roots_callback` and `call_tool` returns a plain `CallToolResult`. `input_required_max_rounds` (default 10) bounds it.
* To inspect or persist rounds, use `client.session.call_tool(..., allow_input_required=True)` and own the `while isinstance(result, InputRequiredResult)` loop yourself.
* On `@mcp.tool()`, a dependency that asks the user produces this result for you (**[Dependencies](https://py.sdk.modelcontextprotocol.io/handlers/dependencies/index.md)**); the **low-level** `Server` is the manual form.
* Prompts and resources participate too: an `@mcp.prompt()` or template `@mcp.resource()` function returns the `InputRequiredResult` itself and reads `ctx.input_responses` on the retry.
* `requestState` comes back as client-supplied input, so `MCPServer` seals it by default — resolver state and hand-built state alike — under a process-local key; multi-instance deployments pass `RequestStateSecurity(keys=[...])` (or a custom codec) so every instance can verify what a sibling minted. The seal binds every token to a time window, the originating request, and the authenticated principal when the request carries auth the SDK validated or `bind_principal=` supplies your own identity signal (**[Protecting `requestState`](#protecting-requeststate)**).

This is the mechanism that replaces server-initiated sampling and the rest of the push-style back-channel; see **[Deprecated features](https://py.sdk.modelcontextprotocol.io/deprecated/index.md)**.
