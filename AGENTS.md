# AGENTS.md

## Project
MiniMe is a small interactive coding agent in three flat modules (no package),
written as a handful of classes that get their dependencies through their
constructors; there is no global state:
- `config.py` — `Config` and `load_config()`, which reports invalid settings
  and exits. `main()` calls it once and passes the result down.
- `tools.py` — `Workspace` (path checking), `Tool` (abstract base) with one
  subclass per tool, and `Toolbox` (`for_config()` builds the standard set,
  `schemas()`, `execute()`). The byte-bounding and HTML helpers stay plain
  functions. Imports `config` for the `Config` type only.
- `main.py` — `Conversation` (history, token ratio, trimming), `Agent` (the
  tool loop; prints nothing, reports through `on_text` / `on_notice` callbacks
  and raises `TurnError`), `Repl` (terminal input and output) and `main()`.
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
  `Config` and a temporary workspace. Objects read their settings when they are
  built, so change `settings` before building a `Toolbox` or `Agent`.
- Shared helpers there: `tool` (the standard tools for the test, callable like
  functions: `tool.read_file("f.txt", 3)`; arguments left out are None or
  False), `site` (a local HTTP server with per-test routes), `api` (see below),
  `tool_call` (a tool call for `Toolbox.execute`), and `kinds` (a
  one-label-per-item summary of `history` or of a request's input).
- Build agents in tests with `api.agent()`, never `main.Agent(settings)`: an
  agent that makes its own OpenAI client sets up TLS, which on macOS starts a
  system thread that outlives the test and breaks the pty tests' fork.
- The loop tests (`tests/test_loop.py`) run the real `openai` client: the
  `api` fixture answers its HTTP requests through `httpx2.MockTransport` from
  a script of `response_body(...)` replies, HTTP error responses or exceptions,
  and records each request body. In `test_loop.py`, `run()` goes through
  `Repl.ask` (output printed, problems reported) and `turn()` calls
  `Agent.run_turn` directly (problems raised). There are no stand-in response objects, so
  these tests also catch SDK changes; keep them passing when upgrading `openai`.
  Use the `streaming` fixture to run a test in both modes.
- `tests/test_terminal.py` runs the real `main.main()` as a child process on a
  pseudo-terminal and types Ctrl-C / Ctrl-D at it, with a local HTTP server
  streaming scripted replies. It is the only test of real signal delivery; the
  loop tests raise `KeyboardInterrupt` by hand. The child must be forked before
  the server thread starts (see the `terminal` fixture). POSIX only.
- `run_command` tests need a POSIX shell and are skipped elsewhere; the timeout
  tests take one second each because the timeout is a whole number of seconds.

## Conventions and gotchas
- Adding a tool (all in `tools.py`): write a `ToolArguments` subclass for its
  parameters and a `Tool` subclass with `name`, `arguments_model` and
  `run(args)`, then add it in `Toolbox.for_config`. The schema is generated
  from the arguments model and the subclass docstring is the description sent
  to the model. `run` raises on bad input; `Toolbox.execute` turns any
  exception into `{"ok": False, "message": ...}`.
- Tool schemas use OpenAI `strict: True`, so every argument field must be
  required: declare optional ones as `X | None` with no default.
- Config uses `pydantic-settings` `validation_alias`, so field names differ from
  env vars: `OPENAI_MODEL`→`model`, `MAX_TURNS_PER_RUN`→`max_turns`,
  `COMMAND_TIMEOUT_SECONDS`→`command_timeout`, `STREAMING`→`streaming`,
  `REASONING_EFFORT`→`reasoning_effort`, `SUMMARIZE`→`summarize`,
  `MAX_HISTORY_TOKENS`→`max_history_tokens`, `WORKSPACE_ROOT`→`workspace_root`,
  `BRAVE_API_KEY`→`brave_api_key`. Unknown env vars are ignored.
- `workspace_root` defaults to the current working directory and must be an
  existing directory (checked at startup); `.env` is still read from next to
  `config.py`. `Workspace.resolve` restricts the file tools and
  `run_command`'s `cwd` to it, but `run_command` itself runs with
  `shell=True` and is not sandboxed.
- `system_prompt` is plain text; `Agent.instructions()` appends an Environment
  section (workspace root, OS, date), and `run_turn` rebuilds it every turn.
- Tool results are capped at `MAX_TOOL_OUTPUT_BYTES`, each in its own way:
  `read_file` stops at a line boundary and returns `next_start_line` (plus
  `next_offset` when a single over-long line is returned in pieces);
  `web_fetch` returns `next_offset`; `run_command` shares one limit between
  stdout and stderr (neither gets less than half) and keeps the start and end
  of a clipped stream with a note of how many bytes were omitted.
- Tool results go into `history` as plain JSON (`ensure_ascii=False`, no
  wrapper object), so non-ASCII output is not inflated by escaping.
- The web tools use `httpx2` (the client `openai` depends on; plain `httpx` is
  not installed). `Toolbox.for_config` leaves `web_search` out when
  `BRAVE_API_KEY` is unset.
- `WebFetchTool` keeps the last page it downloaded; only continuation calls
  (a non-empty `offset` for the same URL) reuse it, so a first part is always
  fresh. The `httpx2` timeout is per read, so `fetch_page` also enforces an overall
  `WEB_TIMEOUT_SECONDS` deadline on the download.
- `TextExtractor` drops `script`/`style`/`nav`/`header`/`footer`/`aside` and
  containers with a navigation-type ARIA `role`, keeps `<pre>` verbatim, and
  writes links as `[text](absolute-url)`.
- The response loop calls `client.responses.stream(..., store=False)` (or
  `.create` when `STREAMING=false`) and appends outputs to in-memory
  `history`; iterations are capped by `MAX_TURNS_PER_RUN`.
- `Conversation` (in `main.py`) holds `history` and `tokens_per_byte`, and all
  history changes go through it (`add_prompt`, `add_step`,
  `discard_unanswered`, `trim`, `clear`). The token limit is enforced in bytes:
  `Agent.run_turn` converts `max_history_tokens`
  to a byte budget with that ratio (minus the instructions and tool schemas,
  which are sent every time) and re-learns the ratio from each response's
  `usage.input_tokens`, bounded to 0.1-1.0. `/clear` empties `history` and
  keeps the ratio.
- `store=False` means the API keeps nothing, so reasoning items in
  `response.output` must be replayed with their `encrypted_content`. The API
  returns that automatically when `store=False` (checked 2026-10-13 on
  gpt-5.4-mini, gpt-5-mini and o4-mini); no `include` parameter is needed. Do
  not strip or rebuild output items before appending them to `history`.
- `Conversation.trim(max_bytes)` does nothing until `history` exceeds the
  budget, then cuts it back to `TRIM_TARGET` (75%) of it: whole older turns
  first, then the current turn's oldest tool outputs (replaced with
  a placeholder). The request itself is dropped only if it alone
  exceeds the budget.
- Prompt caching (OpenAI, measured 2026-10-13 on gpt-5.4-mini; works with
  `store=False`): a request's prefix is cached from 1,024 tokens, and only an
  exact prefix match hits. Consequences that the code depends on:
  - Cached tokens are a subset of `usage.input_tokens` and still fill the
    context window, so the token limit counts them; they are only reported.
  - Any change near the start of the prompt (a trim, an edited earlier item)
    is a full miss, which is why trimming cuts to 75% instead of to the limit
    (cache hits over a 14-turn run at the limit went from 24% to 53%).
  - `system_prompt` plus the tool schemas come to about 830 tokens, under the
    minimum, so they are not cached on their own; caching starts once the
    conversation takes the request past 1,024 tokens. Keep that part static
    and keep per-session values in the Environment section, which
    `Agent.instructions()` appends last.
- A streamed response that does not complete ends with a `response.incomplete`
  or `response.failed` event and the SDK's `get_final_response()` then raises,
  so `Agent._create_response` takes the response from that event; both
  statuses end the turn with a `TurnError`, which `Repl` reports. The incomplete case was confirmed against the real API; the
  failed case could not be produced there and is covered only by the fixture.
- `[thinking…]` is printed when a stream adds a reasoning item, so it only
  appears in streaming mode. `reasoning` is sent only when `REASONING_EFFORT`
  is set.
- When `Conversation.trim` drops whole turns, `run_turn` calls
  `Agent._summarize` (unless `SUMMARIZE=false`), which inserts the result as a
  leading user message starting with "[Summary of the earlier conversation…]"
  (so a later trim folds it into the next summary). The summary request
  reuses the normal instructions, tools and reasoning setting with
  `tool_choice="none"` to stay on the cached prefix. A failed request falls
  back to dropping; Ctrl-C during it puts the dropped turns back; a summary
  that is empty or would not fit is not inserted.
- `Conversation.add_step` records a model response and all of its tool outputs
  together, so history never holds a tool call without an output. On Ctrl-C or
  an API error, completed steps stay, tool calls that did not finish get
  an "Interrupted by the user." result, and a prompt with no completed step is removed;
  the exception then propagates to `Repl`, which reports it.
- `Agent.usage` holds the token counts of the last turn; `Repl.ask` prints the
  `[Context: …]` line from it after reporting any error.
- `.env` is gitignored but present locally, overriding the model and base URL to a
  local OpenAI-compatible endpoint. It is read by `Config` only and is not
  exported to `os.environ`. Do not print or commit secrets.
