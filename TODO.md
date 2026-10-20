# TODO

Ideas for later, roughly in order of value for effort. 1 and 2 belong
together: one shows what the agent does, the other lets you stop it first.

1. **Show tool calls as they happen.** Before each tool runs, `Agent` calls
   `on_notice` with its name and key argument: `[run_command: pytest -q]`,
   `[write_file: src/x.py]`, `[web_fetch: https://…]`. Today nothing is shown
   while tools run, so a long step looks like a hang.
2. **Ask before side-effecting tools.** A `needs_approval` flag on
   `run_command`, `write_file` and `edit_file`; an `approve` callback the agent
   calls before them; `Repl` asks `Run? [y/N]`. A declined call goes back to
   the model as a tool result. `/approve on|off` turns it off for trusted
   sessions. Matters more now that the tools are not confined to a workspace.
3. **Better input with prompt_toolkit.** Multi-line prompts, history kept
   across sessions, Tab completion of `/commands` (from `Repl.commands`) and of
   paths after `/cd`. Replaces the `readline` import; a new dependency.
4. **Markdown answers with Rich.** Stream raw text as now and re-render each
   answer once it is complete (Rich `Live` redraws badly when output
   overflows the screen). A new dependency.
5. **Input history file** (`~/.minime_history`) through `readline`, if 3 is
   not done.
6. **Mark `test_ctrl_c_while_an_answer_streams` as a non-strict xfail.** It
   fails on most gpt-5.4-mini runs because pydantic-core loses the interrupt
   (see the known limitation in `AGENTS.md`).
7. **`/save <file>` and `/load <file>`.** History mixes plain dicts and SDK
   objects, so saving needs `model_dump()`.
8. **Load project instructions.** Read `AGENTS.md` (or `CLAUDE.md`) from the
   working directory and append it to the instructions, after the static
   prompt and before or with the Environment section, so the cached prefix
   stays the same. Every major coding agent does this to pick up a project's
   build commands and conventions. Decide whether to look in parent
   directories too, and cap its size so a large file cannot crowd out the
   history.

## Later: a full-screen interface

- **TUI:** a Textual app (`minime-tui`) beside the REPL, with streamed
  Markdown (`Markdown.get_stream()`), tool calls in collapsible panels, a
  status bar and modal approvals. Best after 3 and 4.
- **Prerequisite for any non-REPL interface:** a cooperative stop flag in
  `Agent`. Ctrl-C cannot reach the worker thread a TUI or GUI runs the agent
  in, so cancelling a turn needs the flag plus closing the HTTP stream, and
  `run_command` must check it while it waits.
- **Web GUI**, only if needed: NiceGUI, bound to localhost and behind a token,
  since the page would control an unsandboxed shell.
