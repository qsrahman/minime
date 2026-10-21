import math
import sys
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from textwrap import shorten
from typing import Any

from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    PathCompleter,
    WordCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from pydantic import ValidationError
from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.theme import Theme

from minime.agent import Agent, TurnError
from minime.commands import Command, standard_commands
from minime.config import Config
from minime.fs import EditFileTool, ReadFileTool, WriteFileTool
from minime.shell import RunCommandTool
from minime.tools import Tool, Toolbox
from minime.web import WebFetchTool, WebSearchTool

HISTORY_FILE = Path("~/.minime_history")
PROMPT = "› "
# Answers wider than this are hard to read on a wide terminal.
MAX_ANSWER_WIDTH = 100

# Plain ANSI colours, which the terminal's own palette adapts to a light or dark
# background; Rich's defaults put code on a fixed black or dark grey one.
THEME = Theme(
    {
        "markdown.code": "bold cyan",
        "markdown.code_block": "cyan",
        "progress.elapsed": "dim",
    }
)
CODE_THEME = "ansi_light"  # plain colours; "ansi_dark" uses the bright ones


def home_relative(path: Path) -> str:
    """The path with the home directory shortened to "~"."""
    if not path.is_relative_to(Path.home()):
        return str(path)
    return str(Path("~") / path.relative_to(Path.home()))


class CommandCompleter(Completer):
    """Completes a command name at the start of the line, and a directory
    after /cd. (NestedCompleter does neither: its words cannot start with "/".)"""

    def __init__(self, names: list[str]) -> None:
        self.names = WordCompleter(names, ignore_case=True, WORD=True)
        self.directories = PathCompleter(only_directories=True, expanduser=True)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        text = document.text_before_cursor
        if " " not in text:
            yield from self.names.get_completions(document, complete_event)
        elif text.lower().startswith("/cd "):
            path = Document(text[4:].lstrip())
            yield from self.directories.get_completions(path, complete_event)


class Repl:
    """The interactive terminal: reads prompts and slash commands and prints
    what the agent says.

    It uses the agent's run_turn() and TurnError; the commands use a few
    more of its methods and attributes.
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.running = False
        self.at_line_start = True
        self.last_prompt: str | None = None  # for /retry
        self.commands: dict[str, Command] = {
            name: command for command in standard_commands() for name in command.names
        }
        self.console = Console(theme=THEME)
        # Model text since the last notice, re-rendered as Markdown once complete.
        self.answer = ""
        # Shown while nothing is printed: waiting for the model or a tool.
        self.spinner: Progress | None = None
        # The tool call now running and when it started.
        self.tool: tuple[str, float] | None = None

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        self.console.print(
            Text.assemble(
                ("minime", "bold"),
                (f" · {self.agent.model} · {home_relative(Path.cwd())}", "dim"),
            ),
            highlight=False,
            soft_wrap=True,
        )
        self.console.print("Type /help to list commands.\n", style="dim")

        if sys.stdin.isatty():
            read_line = self.prompt_session().prompt
        else:
            read_line = partial(input, PROMPT)
        self.running = True
        while self.running:
            try:
                line = read_line().strip()
            except KeyboardInterrupt, EOFError:
                print()  # Ctrl-C / Ctrl-D leave the cursor on the prompt line
                break
            if not line:
                continue

            name, _, arg = line.partition(" ")
            command = self.commands.get(name.lower())
            if command:
                self.run_command(name.lower(), command, arg.strip())
            # A single word starting with "/" is a mistyped command, not a
            # prompt; longer input such as "/usr/bin is missing" still goes
            # to the model.
            elif line.startswith("/") and " " not in line:
                self.report(
                    f"Unknown command: {line}. Type /help to list commands.", "yellow"
                )
            else:
                self.ask(line)
        print("Bye.")

    def prompt_session(self) -> PromptSession[str]:
        """Line editing for a terminal: history kept across sessions, Tab
        completion of commands (and of directories after /cd), Esc then Enter
        for a new line, and a status bar."""
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        return PromptSession(
            message=[("class:prompt", PROMPT)],
            style=Style.from_dict({"prompt": "ansicyan bold"}),
            history=FileHistory(HISTORY_FILE.expanduser()),
            completer=CommandCompleter(list(self.commands)),
            key_bindings=bindings,
            bottom_toolbar=self.status,
        )

    def status(self) -> str:
        """The status bar: model, reasoning effort and context use."""
        agent = self.agent
        parts = [agent.model, f"effort {agent.reasoning_effort or 'default'}"]
        usage = agent.last_usage
        if usage and usage.input_tokens:
            limit = agent.config.max_context_tokens
            parts.append(f"{usage.input_tokens:,}/{limit:,} tokens")
            details = usage.input_tokens_details
            if details and details.cached_tokens:
                parts.append(f"{details.cached_tokens / usage.input_tokens:.0%} cached")
        return " · ".join(parts)

    def run_command(self, name: str, command: Command, arg: str) -> None:
        """Run a slash command, checking its argument and reporting failures."""
        if arg and not command.argument:
            self.report(f"{name} takes no argument", "yellow")
        elif command.argument.startswith("<") and not arg:
            self.report(f"Usage: {name} {command.argument}", "yellow")
        else:
            try:
                command.run(self, arg)
            except KeyboardInterrupt:
                self.report("Interrupted; returning to prompt", "yellow")
            except Exception as exc:  # noqa: BLE001
                self.report(f"{name} failed: {type(exc).__name__}: {exc}", "red")

    def ask(self, prompt: str) -> None:
        """Run one turn, printing its text as it arrives and reporting problems
        without leaving the loop."""
        self.last_prompt = prompt
        print()
        self.start_spinner("thinking")
        try:
            self.agent.run_turn(
                prompt,
                on_text=self.write,
                on_notice=self.turn_notice,
                on_tool_start=self.tool_start,
                on_tool_end=self.tool_end,
            )
        except TurnError as exc:
            self.notice(str(exc), "yellow")
        except KeyboardInterrupt:
            if self.tool:
                self.show_tool({"ok": False, "message": "interrupted"})
            self.notice("Interrupted; returning to prompt", "yellow")
        except Exception as exc:  # noqa: BLE001
            self.notice(f"Agent error: {type(exc).__name__}: {exc}", "red")
        self.tool = None
        self.end_answer()
        print()

    def start_spinner(self, label: str) -> None:
        """Show a spinner with the label and the time elapsed until something
        is printed; only in an interactive terminal."""
        self.stop_spinner()
        if not self.console.is_interactive:
            return
        self.spinner = Progress(
            SpinnerColumn(style="dim"),
            TextColumn("{task.description}", style="dim", markup=False),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.spinner.add_task(label, total=None)
        self.spinner.start()

    def stop_spinner(self) -> None:
        """Erase the spinner, leaving the cursor where it was drawn."""
        if self.spinner:
            self.spinner.stop()
            self.spinner = None

    def turn_notice(self, text: str) -> None:
        """A notice from the agent during a turn, which then carries on."""
        self.notice(text)
        self.start_spinner("thinking")

    def tool_start(self, description: str) -> None:
        self.end_answer()
        self.tool = (description, time.monotonic())
        self.start_spinner(description)

    def tool_end(self, description: str, result: dict[str, Any]) -> None:
        self.show_tool(result)
        self.start_spinner("thinking")

    def show_tool(self, result: dict[str, Any]) -> None:
        """Print the running tool call with how it ended and how long it took:
        "✓ read_file: x.py  0.1s" or "✗ run_command: pytest  exit 1  2.3s"."""
        if self.tool is None:
            return
        (description, started), self.tool = self.tool, None
        self.end_answer()
        line = Text()
        if result.get("ok"):
            line.append("✓ ", "green").append(description, "dim")
        else:
            if result.get("timed_out"):
                failure = "timed out"
            elif result.get("returncode") is not None:
                failure = f"exit {result['returncode']}"
            else:
                message = str(result.get("message", "failed"))
                failure = shorten(message, 60, placeholder=" …")
            line.append("✗ ", "red").append(description, "dim")
            line.append(f"  {failure}", "red")
        line.append(f"  {time.monotonic() - started:.1f}s", "dim")
        self.console.print(line, highlight=False, soft_wrap=True)

    def write(self, text: str) -> None:
        """Print model text, remembering whether the cursor is at a line start."""
        self.stop_spinner()
        print(text, end="", flush=True)
        if text:
            self.answer += text
            self.at_line_start = text.endswith("\n")

    def end_answer(self) -> None:
        """Finish the model text printed since the last notice, replacing it
        with its Markdown rendering when the terminal allows."""
        self.stop_spinner()
        if not self.at_line_start:
            self.write("\n")
        text, self.answer = self.answer, ""
        console = self.console
        if not text.strip() or not console.is_terminal or console.is_dumb_terminal:
            return
        rows = sum(
            max(1, math.ceil(cell_len(line.expandtabs()) / console.width))
            for line in text.removesuffix("\n").split("\n")
        )
        # Text that has scrolled off the screen cannot be erased; leave it raw.
        if rows >= console.height:
            return
        # Move the cursor up to where the text started and erase to the end.
        print(f"\033[{rows}F\033[J", end="")
        console.print(
            Markdown(text, code_theme=CODE_THEME),
            width=min(console.width, MAX_ANSWER_WIDTH),
        )

    def notice(self, text: str, style: str = "dim") -> None:
        """Print a bracketed status line on a line of its own: dim for
        status, yellow for warnings and red for errors."""
        self.end_answer()
        self.console.print(
            f"[{text}]", style=style, markup=False, highlight=False, soft_wrap=True
        )

    def report(self, text: str, style: str = "dim") -> None:
        """Print a command's result as a notice followed by a blank line."""
        self.notice(text, style)
        print()


# pydantic-settings options that turn Config into command-line flags
# (--max-steps-per-turn 8, --no-streaming); they override the environment
# and .env. Only main() passes them, so no other Config() reads sys.argv.
CLI_OPTIONS: dict[str, Any] = {
    "_cli_parse_args": True,
    "_cli_prog_name": "minime",
    "_cli_kebab_case": True,
    "_cli_implicit_flags": True,
}


def build_toolbox(config: Config) -> Toolbox:
    """The standard tools; web_search only when there is a key for it."""
    tools: list[Tool[Any]] = [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        RunCommandTool(config.command_timeout),
        WebFetchTool(),
    ]
    if config.brave_api_key:
        tools.append(WebSearchTool(config.brave_api_key))
    return Toolbox(tools)


def main() -> None:
    """Load the configuration, exiting with a readable error if it is invalid,
    then build the agent and run the terminal loop."""
    try:
        config = Config(**CLI_OPTIONS)
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    agent = Agent(config, client, build_toolbox(config))
    Repl(agent).run()


if __name__ == "__main__":
    main()
