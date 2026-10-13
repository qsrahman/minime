# AGENTS.md

## Project
MiniMe is a small interactive coding agent in three flat modules (no package):
- `config.py` — `Config` and the cached `get_config()`.
- `tools.py` — tool handlers, argument models, the `TOOLS` registry, schema
  generation and `execute_tool_call`. Imports `config`.
- `main.py` — system prompt, history trimming, the response loop and the REPL.
  Imports both; nothing imports `main`.

## Commands
- `uv sync` — install dependencies.
- `uv run main.py` — run the agent (the only entrypoint).
- Requires Python 3.14+; run through `uv` so the pinned interpreter is used.
- No test suite, linter, formatter, or CI is configured. Do not introduce or
  assume `pytest` / `ruff` / `mypy` commands; ask before adding tooling.

## Conventions and gotchas
- Adding a tool (all in `tools.py`): write the handler, a `ToolArguments`
  subclass for its parameters, and one `TOOLS` entry. The schema is generated from the model and
  the handler's docstring is the description sent to the model.
- Tool schemas use OpenAI `strict: True`, so every argument field must be
  required: declare optional ones as `X | None` with no default. Field names
  must match the handler's parameters.
- Config uses `pydantic-settings` `validation_alias`, so field names differ from
  env vars: `OPENAI_MODEL`→`model`, `MAX_TURNS_PER_RUN`→`max_turns`,
  `COMMAND_TIMEOUT_SECONDS`→`command_timeout`, `STREAMING`→`streaming`,
  `WORKSPACE_ROOT`→`workspace_root`, `BRAVE_API_KEY`→`brave_api_key`. Unknown
  env vars are ignored.
- Config is loaded lazily by the cached `get_config()`; there is no module-level
  config object.
- `workspace_root` defaults to the current working directory; `.env` is still
  read from next to `config.py`. `resolve_workspace_path` restricts the file tools
  and `run_command`'s `cwd` to it, but `run_command` itself runs with
  `shell=True` and is not sandboxed.
- `system_prompt` is a `str.format` template filled in per turn (workspace root,
  OS, date). Literal braces in it must be doubled.
- Tool results are capped at `MAX_TOOL_OUTPUT_BYTES`: `read_file` and
  `web_fetch` content, and `run_command` stdout and stderr separately.
- The web tools use `httpx2` (the client `openai` depends on; plain `httpx` is
  not installed). `build_tool_schemas` omits `web_search` when `BRAVE_API_KEY`
  is unset.
- `web_fetch` downloads fresh whenever `offset` is empty and clears the
  one-entry `fetch_page` cache; only continuation calls reuse the cached page.
- `TextExtractor` drops `script`/`style`/`nav`/`header`/`footer`/`aside` and
  containers with a navigation-type ARIA `role`, keeps `<pre>` verbatim, and
  writes links as `[text](absolute-url)`.
- The response loop calls `client.responses.stream(..., store=False)` (or
  `.create` when `STREAMING=false`) and appends outputs to in-memory
  `history`; iterations are capped by `MAX_TURNS_PER_RUN`.
- `trim_history` keeps `history` within `MAX_HISTORY_TURNS` and
  `MAX_HISTORY_BYTES`: it drops whole older turns first, then replaces the
  current turn's oldest tool outputs with `OMITTED_TOOL_OUTPUT`, and only then
  drops the request.
- `run_turn` appends a model response and all of its tool outputs to `history`
  together, so history never holds a tool call without an output. On Ctrl-C or
  an API error, completed steps stay, tool calls that did not finish get
  `INTERRUPTED_TOOL_RESULT`, and a prompt with no completed step is removed.
- `.env` is gitignored but present locally, overriding the model and base URL to a
  local OpenAI-compatible endpoint. It is read by `Config` only and is not
  exported to `os.environ`. Do not print or commit secrets.
