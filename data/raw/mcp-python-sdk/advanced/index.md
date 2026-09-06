# Advanced

Everything an ordinary server or client needs has a topical home in the sections above.
This section is the escape hatches you reach for when `MCPServer`'s convenience
layer is in the way:

* **[The low-level Server](https://py.sdk.modelcontextprotocol.io/advanced/low-level-server/index.md)**: the class `MCPServer` is built on.
  Hand-written schemas, `on_*` handlers, nothing checked for you, and custom JSON-RPC
  methods of your own.
* **[Pagination](https://py.sdk.modelcontextprotocol.io/advanced/pagination/index.md)** and **[Middleware](https://py.sdk.modelcontextprotocol.io/advanced/middleware/index.md)**: two things you
  can *only* do on the low-level `Server`.
* **[Extensions](https://py.sdk.modelcontextprotocol.io/advanced/extensions/index.md)** and **[MCP Apps](https://py.sdk.modelcontextprotocol.io/advanced/apps/index.md)**: the protocol's
  extension surface. Compose extension packages into a server, or write your own.

A few things you might reasonably look for here live where you'd actually use them
instead:

* **Authorization** is under **[Running your server](https://py.sdk.modelcontextprotocol.io/run/index.md)** because you
  protect a server where you deploy it.
* **OAuth**, **identity assertion**, connecting to **multiple servers**, and the
  response **cache** are all under **[Clients](https://py.sdk.modelcontextprotocol.io/client/index.md)**.
* **Multi-round-trip requests** and **Subscriptions** are under
  **[Inside your handler](https://py.sdk.modelcontextprotocol.io/handlers/index.md)** because both are things a
  handler *does*.
* **URI templates** is under **[Servers](https://py.sdk.modelcontextprotocol.io/servers/index.md)**, next to Resources.
* **[Protocol versions](https://py.sdk.modelcontextprotocol.io/protocol-versions/index.md)** and
  **[Deprecated features](https://py.sdk.modelcontextprotocol.io/deprecated/index.md)** each have their own top-level page.

If you're not sure whether you need this section, you don't.
