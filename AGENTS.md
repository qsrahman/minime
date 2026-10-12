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
- Adding a tool requires two edits kept in sync: the handler in `TOOLS` and the
  matching entry in `tool_schemas`.
- Tool schemas use OpenAI `strict: True`, so every property must appear in
  `required` (mark optional ones nullable, e.g. `["type", "null"]`) and
  `additionalProperties` must be `False`. Handler signatures must match.
- Config uses `pydantic-settings` `validation_alias`, so field names differ from
  env vars: `OPENAI_MODEL`→`model`, `MAX_TURNS_PER_RUN`→`max_turns`,
  `COMMAND_TIMEOUT_SECONDS`→`command_timeout`. Unknown env vars are ignored.
- `workspace_root` is the directory containing `main.py`. `resolve_workspace_path`
  restricts only the file tools to it; `run_command` runs with `shell=True` and is
  not sandboxed.
- The response loop streams via `client.responses.stream(..., store=False)` and
  appends outputs to in-memory `history`; iterations are capped by
  `MAX_TURNS_PER_RUN`.
- `.env` is gitignored but present locally, overriding the model and base URL to a
  local OpenAI-compatible endpoint. Do not print or commit secrets.
