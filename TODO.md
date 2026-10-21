# TODO

Ideas for later, roughly in order of value for effort.

1. **Ask before side-effecting tools.** A `needs_approval` flag on
   `run_command`, `write_file` and `edit_file`; an `approve` callback the agent
   calls before them; `Repl` asks `Run? [y/N]`. A declined call goes back to
   the model as a tool result. `/approve on|off` turns it off for trusted
   sessions. Matters more now that the tools are not confined to a workspace.
   Each call is already announced before it runs (`Toolbox.describe`).
2. **Mark `test_ctrl_c_while_an_answer_streams` as a non-strict xfail.** It
   fails on most gpt-5.4-mini runs because pydantic-core loses the interrupt
   (see the known limitation in `AGENTS.md`).
3. **`/save <file>` and `/load <file>`.** History mixes plain dicts and SDK
   objects, so saving needs `model_dump()`.
4. **Load project instructions.** Read `AGENTS.md` (or `CLAUDE.md`) from the
   working directory and append it to the instructions, after the static
   prompt and before or with the Environment section, so the cached prefix
   stays the same. Every major coding agent does this to pick up a project's
   build commands and conventions. Decide whether to look in parent
   directories too, and cap its size so a large file cannot crowd out the
   history.

## Done

Finished on the `tui` branch:

- [x] **Show tool calls as they happen.** Each call is announced before it
  runs, e.g. `[run_command: pytest -q]` (`Toolbox.describe`).
- [x] **Better input with prompt_toolkit.** History kept across sessions, Tab
  completion of `/commands` and of directories after `/cd`, Esc-Enter for a
  new line, and a status bar; replaces the `readline` import.
- [x] **Markdown answers with Rich.** Answers stream raw and are redrawn as
  Markdown once complete.
- [x] **Input history file** (`~/.minime_history`), through prompt_toolkit.
- [x] **A pasted multi-line block becomes several prompts** (command line):
  bracketed paste keeps it one prompt.
- [x] **Show how each tool call ended:** `✓` or `✗`, the exit code, timeout
  or message, and how long it took (`on_tool_start` / `on_tool_end`).
- [x] **A spinner while waiting** for the model or a tool, with the time
  elapsed.
- [x] **Notices coloured by kind:** dim status, yellow warnings, red errors.
- [x] **Code in the terminal's own colours,** with no fixed background.
- [x] **Banner and prompt:** `minime · model · ~/proj` and a cyan `›` prompt.
- [x] **Markdown answers capped at 100 columns** on wide terminals.
- [x] **Command descriptions in the completion menu** and **suggestions from
  history** (→ accepts).
- [x] **A quiet status bar:** dim text, no reverse band, key hints at the
  right edge when they fit.
- [x] **Tool lines and the spinner kept to one row,** the description
  shortened; the spinner says "ctrl-c to stop".

## Look

Smaller improvements to the terminal's appearance, not yet done:

1. **Separate turns** with a dim rule, or the prompt echoed in colour.
2. **`/help` and `/context` as Rich tables,** with a bar for context use.
3. **Diffs for `edit_file` and `write_file`** under their lines; best with
   approvals (item 1).
4. **Clickable paths** in tool lines (OSC-8 hyperlinks).
5. **Terminal title and a finished signal:** `minime · working…` as the tab
   title during a turn; ring the bell after a turn longer than about 30 s.
6. **Left-aligned headings:** Rich centres `# H1` in a box; a small
   `Markdown` subclass could draw headings left-aligned and bold.
7. **Friendlier API errors:** a dim hint under the red line for the common
   cases (connection: check `OPENAI_BASE_URL`; 401: the API key; unknown
   model: `/model`).
8. **Confirm `/reset`** when it would discard more than a couple of turns.
9. **A shorter status bar on narrow terminals:** at 60 columns the token
   count and cache share are cut off.

## Command line

Problems first, since each silently does the wrong thing today:

1. **Ctrl-C at the prompt ends the session and loses the conversation,** and
   because of the lost-interrupt limitation a second press often lands there.
   Make Ctrl-C at the prompt clear the line, as `python` and `bash` do (a
   `prompt_toolkit` key binding), and leave quitting to Ctrl-D, `/exit` and
   `/quit`. Also catch `KeyboardInterrupt` around the whole loop body: one
   that arrives while `Repl.ask` prints its "Interrupted" notice now ends the
   program with a traceback. A few lines.
2. **Installed copies cannot be configured from a file.** `.env` is read only
   from the source tree, so after `uv tool install .` only environment
   variables and flags remain. Add a user config file such as
   `~/.config/minime/config.toml` (pydantic-settings'
   `TomlConfigSettingsSource`), below `.env` in precedence, and say where it
   goes.

Additions:

3. **One-shot mode:** `minime "explain this error"` or
   `git diff | minime "review this"` runs one turn, prints the answer and
   exits with status 0, or 1 on a `TurnError` or API error. When input is
   piped, show no banner or prompts and send notices to stderr, so stdout
   carries only the answer. A positional argument via pydantic-settings'
   `CliPositionalArg` plus `isatty()` checks; about 40 lines and tests.
4. **`--version`** from the package metadata (`importlib.metadata`).
5. **`!command` shell escape:** run `!git status` without asking the model;
   show the output and optionally add it to the conversation.
6. **`--continue`** to reopen the last session, on top of `/save` and
   `/load` (item 3 above).

Suggested order: 1, then 3 and 2; 4 and 5 are cheap extras.

## New tools

Each is a `ToolArguments` model, a `Tool` subclass and a line in
`build_toolbox`. Read-only tools will not need approval (item 1), which is
much of their value. In order:

1. **`search_files(pattern, path, glob)` and `list_files(path, glob)`.**
   Read-only search across file contents and directory listing, skipping
   `.git`, `node_modules` and similar, capped like the other tools. Models
   now do this through `run_command` (`grep -rn`, `find`, `ls -R`), with
   noisy output and platform-specific flags. Pure Python (`os.walk` and
   `re`), or `rg` when it is installed.
2. **`remember(note)` and `recall(query)`.** Memory across sessions in a
   notes file such as `~/.minime/memory.md`, for preferences, project facts
   and decisions; small notes could go straight into the instructions. Today
   everything is lost when MiniMe exits.
3. **`view_image(path)`.** Let the model see screenshots, diagrams and
   photos, sent as a base64 data URL. First check whether a function result
   can carry an image in the Responses API or it needs a separate input
   item; local models may not support images.
4. **MCP client.** Connect to MCP servers named in the config and offer
   their tools through the `Toolbox` (GitHub, databases, calendars). Needs
   the `mcp` SDK, a bridge to its async API, and care with strict mode,
   since server schemas often are not strict.
5. **`update_plan(steps)`.** A to-do list the model keeps for multi-step
   tasks, shown to the user as a notice when it changes (like Codex's
   `update_plan`).
6. **`read_document(path)`.** Text from PDF and DOCX files, paged like
   `read_file`; needs `pypdf` and `python-docx`.
7. **`apply_patch(patch)`.** Several edits in one call, in the patch format
   OpenAI's models are trained on; fewer round trips for edits in many
   places.
8. **`http_request(method, url, headers, body)`.** POST, PUT and JSON calls
   to APIs, which `web_fetch` (GET only, converted to text) cannot make; a
   side-effecting tool for approvals.

Not worth a tool: a calculator (`python -c` through `run_command`), the date
(in the Environment section), asking the user (the model can just ask),
deleting or moving files (`rm`, `mv`), and sub-agents.

## Concurrency

1. **Run a step's read-only tool calls in parallel.** The model often asks
   for several calls in one response (three `web_fetch`, four `read_file`);
   `run_turn` runs them one after another, so a step takes the sum of their
   times instead of the slowest. Run them in a `ThreadPoolExecutor` only when
   every call in the step is read-only (a `read_only` flag on `Tool`, which
   approvals need too), keeping results in call order; anything else stays
   sequential. Reason: Ctrl-C only reaches the main thread, so a
   `run_command` in a worker would not be killed, and side-effecting calls
   could race. On Ctrl-C the unfinished calls are recorded as interrupted;
   orphaned read-only calls end within their timeouts. About 20-30 lines.
2. **Rewrite on asyncio** (`AsyncOpenAI`, async `httpx2`, asyncio
   subprocesses), together with the TUI or MCP, which are both async-first.
   Concurrency becomes `asyncio.gather` and cancelling becomes
   `task.cancel()`, which reaches running commands too. Check whether it
   fixes the lost Ctrl-C: `asyncio.run` cancels the main task on SIGINT
   instead of raising `KeyboardInterrupt` wherever the code is, so the
   cancellation should land at the next await, not inside pydantic-core's
   validation (untested). Touches most of `agent.py`, `shell.py`, `web.py`
   and the loop and terminal tests.
3. **Not now: concurrent turns or sub-agents.** Typing while the agent works,
   or agents splitting a task, needs a TUI, rules for shared history and cost
   control.

## Later: a full-screen interface

- **TUI:** a Textual app (`minime-tui`) beside the REPL, with streamed
  Markdown (`Markdown.get_stream()`), tool calls in collapsible panels, a
  status bar and modal approvals.
- **Prerequisite for any non-REPL interface:** a cooperative stop flag in
  `Agent`. Ctrl-C cannot reach the worker thread a TUI or GUI runs the agent
  in, so cancelling a turn needs the flag plus closing the HTTP stream, and
  `run_command` must check it while it waits.
- **Web GUI**, only if needed: NiceGUI, bound to localhost and behind a token,
  since the page would control an unsandboxed shell.
