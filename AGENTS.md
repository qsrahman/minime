# AGENTS.md

## Project
MiniMe is a small interactive coding agent: the package `minime` in
`src/minime/` (eight modules), written as a handful of classes that get their dependencies through their
constructors; there is no global state:
- `config.py` — `Config`. `main()` builds it once, reporting invalid settings
  and exiting, and passes it down.
- `tools.py` — what every tool shares: `MAX_TOOL_OUTPUT_BYTES`,
  `ToolArguments`, the abstract `Tool` base class and
  `Toolbox` (`schemas()`, `execute()`). Imports nothing from the project.
- `fs.py` — `read_file`, `write_file` and `edit_file` (a `Tool` subclass and
  an arguments model each).
- `shell.py` — `run_command`, with the plain helper functions that capture,
  clip and kill command output and processes.
- `web.py` — `web_fetch` and `web_search`, with the HTML-to-text and download
  helpers and their limits.
- `context.py` — `Context`: the history and the learned tokens per byte, with
  every change to the history (adding prompts, steps and summaries, trimming,
  reset), plus `json_bytes`.
- `agent.py` — the system prompt, `TurnError` and `Agent` (the tool
  loop; prints nothing, reports through `on_text` / `on_notice` callbacks and
  raises `TurnError`). `Agent(config, client, toolbox)` is given its OpenAI
  client and tools; it imports only `config`, `context` and `tools`, never
  the concrete tool modules.
- `main.py` — the command line and the composition root: `Repl` (all
  terminal input and output, and the slash commands in its `commands` table,
  which `/help` lists), `build_toolbox(config)` (the standard tools),
  `CLI_OPTIONS` and `main()`, which loads `Config` from flags, environment and
  `.env` (reporting invalid settings), builds the client, toolbox and agent,
  and runs `Repl(agent, config.model)`. `Repl` uses only the agent's
  `run_turn()` and `reset()` and `TurnError`. Nothing imports `main` except
  the tests.

## Commands
- `uv sync` — install dependencies.
- `uv run minime` — run the agent. The `minime` command comes from
  `[project.scripts]` (`minime.main:main`); `uv sync` installs the package in
  editable mode, so source changes apply without reinstalling.
  `python -m minime.main` also works.
- `uv run pytest` — run the offline tests in `tests/` (a few seconds; no network
  access, API key or `.env` needed).
- `uv run pytest -m live` — run `tests/test_live.py` against the real model in
  `.env` (about a minute, real API calls). Settings exported in the shell
  override `.env`, e.g. `OPENAI_BASE_URL=https://api.openai.com/v1
  OPENAI_MODEL=gpt-5.4-mini uv run pytest -m live`.
- Requires Python 3.14+; run through `uv` so the pinned interpreter is used.
- `uv run ruff check` and `uv run ruff format` — lint and format; both must be
  clean. Rules are in `pyproject.toml`; `E501` is ignored, so long strings and
  docstrings are fine.
- No type checker or CI is configured. Do not introduce or assume `mypy`
  commands; ask before adding tooling.

## Tests
- Tests import the installed package (`from minime import tools`,
  `from minime.agent import Agent`); ruff's `src = ["src", "tests"]` makes them
  sort as first-party. Patch names by their full module path, e.g.
  `"minime.agent.SYSTEM_PROMPT"`. Code that starts the agent as a process runs
  `python -m minime.main` with the venv's interpreter.
- `tests/conftest.py` has an autouse `settings` fixture: every test gets its own
  `Config` and runs in a temporary working directory (the `workspace`
  fixture), since the tools take relative paths from the current directory.
  Objects read their settings when they are
  built, so change `settings` before building a `Toolbox` or `Agent`.
- Shared helpers there: `tool` (the standard tools for the test, callable like
  functions: `tool.read_file("f.txt", 3)`; arguments left out are None or
  False), `site` (a local HTTP server with per-test routes), `api` (see below),
  `tool_call` (a tool call for `Toolbox.execute`), and `kinds` (a
  one-label-per-item summary of `history` or of a request's input).
- Patch a name in the module that looks it up: `"minime.agent.SYSTEM_PROMPT"`,
  `"minime.agent.date"` for the agent, `main.Config` and `main.OpenAI` (with
  `from minime import main`) for the entry point; a `Config` stand-in must
  accept keyword arguments, since `main()` passes `CLI_OPTIONS`.
- Build agents in offline tests with `api.agent()`, which passes the fake
  client: a real OpenAI client sets up TLS, which on macOS starts a system
  thread that outlives the test and breaks the pty tests' fork.
- The loop tests (`tests/test_loop.py`) run the real `openai` client: the
  `api` fixture answers its HTTP requests through `httpx2.MockTransport` from
  a script of `response_body(...)` replies, HTTP error responses or exceptions,
  and records each request body. In `test_loop.py`, `run()` goes through
  `Repl.ask` (output printed, problems reported) and `turn()` calls
  `Agent.run_turn` directly (problems raised). There are no stand-in response objects, so
  these tests also catch SDK changes; keep them passing when upgrading `openai`.
  Use the `streaming` fixture to run a test in both modes. `TestRepl` drives
  `Repl` over a `FakeAgent` (just `run_turn()` and `reset()`) via
  `repl_session(monkeypatch, *inputs)`; only `test_main_wires_up_the_agent`
  goes through `main()`.
- `tests/test_terminal.py` runs the real `main.main()` as a child process on a
  pseudo-terminal and types Ctrl-C / Ctrl-D at it, with a local HTTP server
  streaming scripted replies. It is the only test of real signal delivery; the
  loop tests raise `KeyboardInterrupt` by hand. The child must be forked before
  the server thread starts (see the `terminal` fixture). POSIX only.
- Live tests (`-m live`, deselected by default) use the `.env` settings with a
  temporary workspace and assert on effects (files, tool calls, recalled
  facts), not wording. `SHELL_SETTINGS` captures the shell's settings before
  the autouse fixture clears them. `conftest.py` sorts live tests last because
  their real clients start threads that would break the pty tests' fork.
- `tests/test_live_repl.py` (live) types at the real `minime` on a
  pseudo-terminal against the `.env` model: commands, memory and `/reset`,
  Ctrl-C while an answer streams, while a command runs and before the answer
  starts, a double Ctrl-C, and quitting. The shared `Terminal` helper is in
  `conftest.py`; pseudo-terminal tests carry the `pty` marker and are sorted
  first within the offline and the live group, before real clients start
  threads.
- `run_command` tests need a POSIX shell and are skipped elsewhere; the timeout
  tests take one second each because the timeout is a whole number of seconds.

## Conventions and gotchas
- Adding a tool: in `fs.py`, `shell.py`, `web.py` or a new module beside them,
  write a `ToolArguments` subclass for its parameters and a `Tool` subclass
  with `name`, `arguments_model` and `run(args)`, then add it in
  `main.build_toolbox`. Tool modules import only from `tools`, never from
  each other or from `agent`, which keeps imports one-way. The schema is generated
  from the arguments model by the SDK's `openai.pydantic_responses_function_tool`
  (copied into a plain dict: with the model attached, the SDK would parse
  arguments itself and abort the turn on invalid ones), and the subclass
  docstring is the description sent to the model. `run` raises on bad input; `Toolbox.execute` turns any
  exception into `{"ok": False, "message": ...}`.
- Adding a REPL command: add a method to `Repl` and an entry in
  `Repl.commands` (the name, lower-case with a leading `/`, maps to the
  handler and its `/help` description). Commands take no arguments; a single
  unknown `/word` is reported, not sent to the model.
- Tool schemas use OpenAI `strict: True`, so every argument field must be
  required: declare optional ones as `X | None` with no default.
- Config fields read the env var of the same name, case-insensitively
  (`STREAMING`→`streaming`); `Config(STREAMING=False)` works too. Where the
  names differ, `validation_alias` maps them: `OPENAI_API_KEY`→`api_key`,
  `OPENAI_MODEL`→`model`, `OPENAI_BASE_URL`→`base_url`,
  `MAX_STEPS_PER_TURN`→`max_steps`,
  `COMMAND_TIMEOUT_SECONDS`→`command_timeout`. Unknown env vars are ignored.
- Command-line flags come from pydantic-settings' CLI support, enabled only
  by `main()` (`Config(**CLI_OPTIONS)`), so other `Config()` calls never
  read `sys.argv`; the autouse fixture also sets `sys.argv` to `["minime"]`.
  Flags are the env var names in kebab case (`--max-steps-per-turn`), booleans
  are `--x` / `--no-x`, and flags override the environment and `.env`.
- Bounds on tool arguments go in the arguments model (`Field(ge=1)`), not in
  `run`: strict mode sends them to the model as `minimum` / `maximum`, which
  it honours (checked on gpt-5.4-mini), and pydantic rejects out-of-range
  values from endpoints that ignore the schema.
- The agent works in the directory it is started from: the tools use paths
  as given, so relative ones (and `run_command`'s `cwd`) start from the
  current directory, and a leading `~` is expanded to the home directory. Nothing restricts them; they reach anywhere the user
  can, and `run_command` runs with `shell=True`, unsandboxed.
- `.env` is read from the project root (`parents[2]` of
  `src/minime/config.py`), wherever the agent is started from. That only
  works for the editable install; a copied install (a plain
  `uv tool install .`) has no `.env` and reads settings from the environment.
- `SYSTEM_PROMPT` is plain text; `Agent._instructions()` appends an Environment
  section (working directory, OS, date), and `run_turn` rebuilds it every turn.
- `read_file` opens files in text mode, so a CRLF file is shown with plain
  `\n`. `edit_file` matches exactly first and, only if that fails on a CRLF
  file, converts `old_content` / `new_content` to CRLF; keep the two in step.
- Tool results are capped at `MAX_TOOL_OUTPUT_BYTES`, each in its own way:
  `read_file` stops at a line boundary and returns `next_start_line` (plus
  `next_offset` when a single over-long line is returned in pieces). It sets
  them whenever the file continues past the returned content, including when
  it stopped at `end_line`, and then adds a `message` naming the exact
  arguments to continue with: models habitually pass an `end_line`, and with a
  null `next_start_line` they took it for the end of the file (seen live);
  `web_fetch` returns `next_offset`; `run_command` shares one limit between
  stdout and stderr (neither gets less than half) and keeps the start and end
  of a clipped stream with a note of how many bytes were omitted.
- Tool results go into `history` as plain JSON (`ensure_ascii=False`, no
  wrapper object), so non-ASCII output is not inflated by escaping.
- The web tools use `httpx2` (the client `openai` depends on; plain `httpx` is
  not installed). `build_toolbox` leaves `web_search` out when
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
  `history`; iterations are capped by `MAX_STEPS_PER_TURN`.
- `Context` (in `context.py`) holds `history` and `tokens_per_byte`, and all
  history changes go through it (`add_prompt`, `add_step`,
  `discard_unanswered`, `trim`, `reset`). It also owns the ratio: the token
  limit is enforced in bytes, `byte_budget(max_tokens, overhead_bytes)`
  converts `max_context_tokens` with it (minus the instructions and tool
  schemas, which are sent every time), and `learn_tokens_per_byte` re-learns
  it from each response's `usage.input_tokens`, bounded to 0.1-1.0.
  `/reset` (`Agent.reset()`) empties `history` and keeps the ratio.
- `store=False` means the API keeps nothing, so reasoning items in
  `response.output` must be replayed with their `encrypted_content`. The API
  returns that automatically when `store=False` (checked 2026-10-13 on
  gpt-5.4-mini, gpt-5-mini and o4-mini); no `include` parameter is needed. Do
  not strip or rebuild output items before appending them to `history`.
- `Context.trim(max_bytes)` does nothing until `history` exceeds the
  budget, then cuts it back to `TRIM_TARGET` (75%) of it: whole older turns
  first, then the current turn's oldest tool outputs (replaced with
  a placeholder). Only if the request alone exceeds the budget is it
  dropped, emptying the history. It returns the whole turns it dropped, which
  `run_turn` hands to `_summarize`.
- Prompt caching (OpenAI, measured 2026-10-13 on gpt-5.4-mini; works with
  `store=False`): a request's prefix is cached from 1,024 tokens, and only an
  exact prefix match hits. Consequences that the code depends on:
  - Cached tokens are a subset of `usage.input_tokens` and still fill the
    context window, so the token limit counts them; they are only reported.
  - Any change near the start of the prompt (a trim, an edited earlier item)
    is a full miss, which is why trimming cuts to 75% instead of to the limit
    (cache hits over a 14-turn run at the limit went from 24% to 53%).
  - `SYSTEM_PROMPT` plus the tool schemas come to about 830 tokens, under the
    minimum, so they are not cached on their own; caching starts once the
    conversation takes the request past 1,024 tokens. Keep that part static
    and keep per-session values in the Environment section, which
    `Agent._instructions()` appends last.
- A streamed response that does not complete ends with a `response.incomplete`
  or `response.failed` event and the SDK's `get_final_response()` then raises,
  so `Agent._create_response` takes the response from that event; both
  statuses end the turn with a `TurnError`, which `Repl` reports. The incomplete case was confirmed against the real API; the
  failed case could not be produced there and is covered only by the fixture.
- Reasoning is not shown: nothing is printed while the model reasons.
  `reasoning` is sent only when `REASONING_EFFORT` is set.
- When `Context.trim` drops whole turns, `run_turn` calls
  `Agent._summarize` (unless `SUMMARIZE=false`). It only makes the request,
  built by the same `Agent._request` as every other one plus
  `tool_choice="none"`, to stay on the cached prefix. The history side is
  `Context`'s: `add_summary` inserts the result as a leading user message
  starting with "[Summary of the earlier conversation…]" (so a later trim
  folds it into the next summary) unless it is empty or would not fit. A
  failed or interrupted request falls back to dropping. Nothing outside `Context` changes `history` directly.
- `Context.add_step` records a model response and all of its tool outputs
  together, so history never holds a tool call without an output. On Ctrl-C or
  an API error, completed steps stay, tool calls that did not finish get
  an "Interrupted by the user." result, and a prompt with no completed step is removed;
  the exception then propagates to `Repl`, which reports it.
- Ctrl-C during a turn is a plain `KeyboardInterrupt`, caught by `Repl.ask`.
  Known limitation: if it lands while the SDK validates a stream event,
  `TypeAdapter.validate_python` (pydantic-core) loses it, Python prints
  "Exception ignored … KeyboardInterrupt" and the answer carries on. It is
  rare on slow streams but frequent on fast ones (5 of 8 live runs on
  gpt-5.4-mini failed `test_ctrl_c_while_an_answer_streams`); each press is
  an independent chance, so pressing it again normally stops the answer.
- `.env` is gitignored but present locally, overriding the model and base URL to a
  local OpenAI-compatible endpoint. It is read by `Config` only and is not
  exported to `os.environ`. Do not print or commit secrets.
