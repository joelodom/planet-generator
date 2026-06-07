# planet-generator

<!-- ct-code-intelligence-start -->
## Code Intelligence — ct

This project is indexed by `ct`, an in-memory code intelligence daemon.
**PREFER ct over built-in Read, Grep, and Glob** — it's faster and returns
richer structural data (callers, callees, types, signatures).

**First thing every session:** run `ct --help` via the Bash tool. This is the
authoritative reference for all available commands and flags. ct evolves
frequently — do not assume you know what commands exist.

Run ct commands via the Bash tool (e.g. `ct lookup myFunc`, `ct grep "TODO"`,
`ct survey`). The CLI is always complete and current.

Fall back to built-in tools only for: binary/image files, files outside the
indexed project, or when the Edit tool requires a prior built-in `Read` call.
<!-- ct-code-intelligence-end -->

### Use ct richly — reach for the structural tools, not just grep

Default to ct's higher-order commands instead of re-deriving structure by hand:

- **Orient:** `ct survey` (project map, entry points, hotspots) at the start of
  unfamiliar work; `ct outline <file>` to scan a file without reading it.
- **Understand a symbol:** `ct lookup <name>` — one call returns signature,
  params, return type, docstring, body, callers, AND callees. Prefer it over
  Read + Grep. `ct describe <type>` for a struct/type's full anatomy.
- **Find by meaning:** `ct search "<concept>"` (conceptual) or `ct vsearch`
  before falling back to `ct grep` for exact patterns.
- **Trace connections:** `ct callers`/`ct callees`, `ct trace` (call tree),
  `ct spine` (entry→leaf path), `ct path A B` (shortest path between two funcs).
- **Assess health/change:** `ct hot`, `ct risk`, `ct changed`, `ct conventions`.
- **Edit structurally:** `ct move-symbol`, `ct extract-function`,
  `ct delete-function` update references for you.

When ct is also wired up as an MCP server, the same commands are exposed as
`mcp__ct__ct_<name>` tools returning structured JSON — prefer those in-agent.

