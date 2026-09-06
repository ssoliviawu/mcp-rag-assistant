# Inside your handler

A handler's arguments come from the client. Everything *else* it can read, and
everything it can do while it runs, is here.

What it can read:

* **[The Context](https://py.sdk.modelcontextprotocol.io/handlers/context/index.md)** is the one extra parameter any handler can
  ask for: the live request, its headers, its session, and the progress and
  change-notification verbs.
* **[Dependencies](https://py.sdk.modelcontextprotocol.io/handlers/dependencies/index.md)** are parameters the model never sees,
  filled in by your own functions with `Resolve`.
* **[Lifespan](https://py.sdk.modelcontextprotocol.io/handlers/lifespan/index.md)** covers state your server builds once at
  startup, and how a handler reaches it through the `Context`.

What it can do while it runs:

* Ask the user for more input with **[Elicitation](https://py.sdk.modelcontextprotocol.io/handlers/elicitation/index.md)**, and
  **[Multi-round-trip requests](https://py.sdk.modelcontextprotocol.io/handlers/multi-round-trip/index.md)**, the 2026-07-28
  pattern that carries it.
* Ask the client for an LLM completion or its workspace folders with
  **[Sampling and roots](https://py.sdk.modelcontextprotocol.io/handlers/sampling-and-roots/index.md)**, deprecated but still
  served.
* Report **[Progress](https://py.sdk.modelcontextprotocol.io/handlers/progress/index.md)** on something slow.
* Write logs (to standard error, for whoever operates the server) with
  **[Logging](https://py.sdk.modelcontextprotocol.io/handlers/logging/index.md)**.
* Tell subscribed clients that something changed with
  **[Subscriptions](https://py.sdk.modelcontextprotocol.io/handlers/subscriptions/index.md)**.

If you haven't registered a handler yet, start with
**[Tools](https://py.sdk.modelcontextprotocol.io/servers/tools/index.md)**. Every page here assumes you have one.
