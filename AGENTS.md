# AGENTS.md

## Project
MiniMe is a single-file interactive coding agent. Everything (config, tools,
tool schemas, and the REPL / response loop) lives in `main.py`. There is no
package layout or separate modules.

## Commands
- `uv sync` — install dependencies.
- `uv run main.py` — run the agent (the only entrypoint).
- Requires Python 3.14+; run through `uv` so the pinned interpreter is used.
- No test suite, linter, formatter, or CI is configured. Do not introduce or
  assume `pytest` / `ruff` / `mypy` commands; ask before adding tooling.

## Conventions and gotchas
- Adding a tool: write the handler, a `ToolArguments` subclass for its
  parameters, and one `TOOLS` entry. The schema is generated from the model and
  the handler's docstring is the description sent to the model.
- Tool schemas use OpenAI `strict: True`, so every argument field must be
  required: declare optional ones as `X | None` with no default. Field names
  must match the handler's parameters.
- Config uses `pydantic-settings` `validation_alias`, so field names differ from
  env vars: `OPENAI_MODEL`→`model`, `MAX_TURNS_PER_RUN`→`max_turns`,
  `COMMAND_TIMEOUT_SECONDS`→`command_timeout`. Unknown env vars are ignored.
- `workspace_root` defaults to the current working directory (override with
  `WORKSPACE_ROOT`); `.env` is still read from next to `main.py`. `resolve_workspace_path`
  restricts only the file tools to it; `run_command` runs with `shell=True` and is
  not sandboxed.
- The response loop calls `client.responses.stream(..., store=False)` (or
  `.create` when `STREAMING=false`) and appends outputs to in-memory
  `history`; iterations are capped by `MAX_TURNS_PER_RUN`.
- `.env` is gitignored but present locally, overriding the model and base URL to a
  local OpenAI-compatible endpoint. It is read by `Config` only and is not
  exported to `os.environ`. Do not print or commit secrets.
