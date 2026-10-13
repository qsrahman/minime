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
- `uv run pytest` — run the tests in `tests/` (a few seconds; no network access,
  API key or `.env` needed).
- Requires Python 3.14+; run through `uv` so the pinned interpreter is used.
- `uv run ruff check` and `uv run ruff format` — lint and format; both must be
  clean. Rules are in `pyproject.toml`; `E501` is ignored, so long strings and
  docstrings are fine.
- No type checker or CI is configured. Do not introduce or assume `mypy`
  commands; ask before adding tooling.

## Tests
- `tests/conftest.py` has an autouse `settings` fixture: every test gets its own
  `Config` and a temporary workspace, with `get_config` patched in `tools` and
  `main`. Change behaviour per test by assigning to `settings` fields.
- Shared helpers there: `site` (a local HTTP server with per-test routes),
  `FakeClient` / `response` / `tool_call` (scripted model responses), and
  `kinds` (a one-label-per-item summary of `history`).
- `tests/test_sdk.py` drives the real `openai` client through
  `httpx2.MockTransport` with canned JSON and server-sent events, in both
  streaming modes. Keep it passing when upgrading `openai`; the other loop
  tests use stand-ins and would not notice an SDK change.
- `run_command` tests need a POSIX shell and are skipped elsewhere; the timeout
  tests take one second each because the timeout is a whole number of seconds.

## Conventions and gotchas
- Adding a tool (all in `tools.py`): write the handler, a `ToolArguments`
  subclass for its parameters, and one `TOOLS` entry. The schema is generated
  from the model and the handler's docstring is the description sent to the
  model.
- Tool schemas use OpenAI `strict: True`, so every argument field must be
  required: declare optional ones as `X | None` with no default. Field names
  must match the handler's parameters.
- Config uses `pydantic-settings` `validation_alias`, so field names differ from
  env vars: `OPENAI_MODEL`→`model`, `MAX_TURNS_PER_RUN`→`max_turns`,
  `COMMAND_TIMEOUT_SECONDS`→`command_timeout`, `STREAMING`→`streaming`,
  `MAX_HISTORY_TURNS`→`max_history_turns`,
  `MAX_HISTORY_TOKENS`→`max_history_tokens`, `WORKSPACE_ROOT`→`workspace_root`,
  `BRAVE_API_KEY`→`brave_api_key`. Unknown env vars are ignored.
- Config is loaded lazily by the cached `get_config()`; there is no module-level
  config object.
- `workspace_root` defaults to the current working directory and must be an
  existing directory (checked at startup); `.env` is still read from next to
  `config.py`. `resolve_workspace_path` restricts the file
  tools and `run_command`'s `cwd` to it, but `run_command` itself runs with
  `shell=True` and is not sandboxed.
- `system_prompt` is plain text; `main()` appends an Environment section
  (workspace root, OS, date) once at startup and passes the result to
  `run_turn` as `instructions`.
- Tool results are capped at `MAX_TOOL_OUTPUT_BYTES`, each in its own way:
  `read_file` stops at a line boundary and returns `next_start_line` (plus
  `next_offset` when a single over-long line is returned in pieces);
  `web_fetch` returns `next_offset`; `run_command` shares one limit between
  stdout and stderr (neither gets less than half) and keeps the start and end
  of a clipped stream with a note of how many bytes were omitted.
- Tool results go into `history` as plain JSON (`ensure_ascii=False`, no
  wrapper object), so non-ASCII output is not inflated by escaping.
- The web tools use `httpx2` (the client `openai` depends on; plain `httpx` is
  not installed). `build_tool_schemas` omits `web_search` when `BRAVE_API_KEY`
  is unset.
- `web_fetch` downloads fresh whenever `offset` is empty and clears the
  one-entry `fetch_page` cache; only continuation calls reuse the cached page.
  The `httpx2` timeout is per read, so `fetch_page` also enforces an overall
  `WEB_TIMEOUT_SECONDS` deadline on the download.
- `TextExtractor` drops `script`/`style`/`nav`/`header`/`footer`/`aside` and
  containers with a navigation-type ARIA `role`, keeps `<pre>` verbatim, and
  writes links as `[text](absolute-url)`.
- The response loop calls `client.responses.stream(..., store=False)` (or
  `.create` when `STREAMING=false`) and appends outputs to in-memory
  `history`; iterations are capped by `MAX_TURNS_PER_RUN`.
- `Conversation` (in `main.py`) holds `history` and `tokens_per_byte`. The
  token limit is enforced in bytes: `run_turn` converts `max_history_tokens`
  to a byte budget with that ratio (minus the instructions and tool schemas,
  which are sent every time) and re-learns the ratio from each response's
  `usage.input_tokens`, bounded to 0.1-1.0. `/clear` empties `history` and
  keeps the ratio.
- `store=False` means the API keeps nothing, so reasoning items in
  `response.output` must be replayed with their `encrypted_content`. The API
  returns that automatically when `store=False` (checked 2026-10-13 on
  gpt-5.4-mini, gpt-5-mini and o4-mini); no `include` parameter is needed. Do
  not strip or rebuild output items before appending them to `history`.
- `trim_history(history, max_bytes)` keeps `history` within that budget and
  `max_history_turns`: it drops whole older turns first, then replaces the
  current turn's oldest tool outputs with `OMITTED_TOOL_OUTPUT`, and only then
  drops the request.
- `run_turn` appends a model response and all of its tool outputs to `history`
  together, so history never holds a tool call without an output. On Ctrl-C or
  an API error, completed steps stay, tool calls that did not finish get
  `INTERRUPTED_TOOL_RESULT`, and a prompt with no completed step is removed.
- `.env` is gitignored but present locally, overriding the model and base URL to a
  local OpenAI-compatible endpoint. It is read by `Config` only and is not
  exported to `os.environ`. Do not print or commit secrets.
