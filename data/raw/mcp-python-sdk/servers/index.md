# Servers

An `MCPServer` exposes three primitives to a connected client. They differ by who
decides to use them:

* A **[tool](https://py.sdk.modelcontextprotocol.io/servers/tools/index.md)** is an action the *model* picks and calls. This is
  the page most people want first, and
  **[Structured Output](https://py.sdk.modelcontextprotocol.io/servers/structured-output/index.md)** is its reference companion:
  everything about the shape of what a tool returns.
* A **[resource](https://py.sdk.modelcontextprotocol.io/servers/resources/index.md)** is read-only data the *application*
  chooses to read. **[URI templates](https://py.sdk.modelcontextprotocol.io/servers/uri-templates/index.md)** is its reference
  companion: the full addressing syntax and the path-safety rules.
* A **[prompt](https://py.sdk.modelcontextprotocol.io/servers/prompts/index.md)** is a message template a *person* invokes by
  name, from a menu or a slash command.

Around the three primitives, the rest of what a server declares:

* **[Completions](https://py.sdk.modelcontextprotocol.io/servers/completions/index.md)** is server-side autocomplete for prompt
  and resource-template arguments.
* **[Images, audio & icons](https://py.sdk.modelcontextprotocol.io/servers/media/index.md)** covers everything a tool can
  return besides text, and the icons a client shows next to your server.
* **[Handling errors](https://py.sdk.modelcontextprotocol.io/servers/handling-errors/index.md)** explains the difference between an
  error the model can recover from and one it must never see.

Every page here stands on its own; jump straight to the one you need. If you haven't
built a server yet, start with **[First steps](https://py.sdk.modelcontextprotocol.io/get-started/first-steps/index.md)** instead.

What happens *inside* the functions you register (the `Context`, dependency injection,
asking the user for more input mid-call) is the next section,
**[Inside your handler](https://py.sdk.modelcontextprotocol.io/handlers/index.md)**.
