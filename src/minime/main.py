import math
import sys
from collections.abc import Iterator
from pathlib import Path
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
from pydantic import ValidationError
from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown

from minime.agent import Agent, TurnError
from minime.commands import Command, standard_commands
from minime.config import Config
from minime.fs import EditFileTool, ReadFileTool, WriteFileTool
from minime.shell import RunCommandTool
from minime.tools import Tool, Toolbox
from minime.web import WebFetchTool, WebSearchTool

HISTORY_FILE = Path("~/.minime_history")


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
        self.console = Console()
        # Model text since the last notice, re-rendered as Markdown once complete.
        self.answer = ""

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        print(
            f"AI coding agent | model: {self.agent.model}\n"
            "Type /help to list commands.\n"
        )

        read_line = self.prompt_session().prompt if sys.stdin.isatty() else input
        self.running = True
        while self.running:
            try:
                line = read_line("> ").strip()
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
                self.report(f"Unknown command: {line}. Type /help to list commands.")
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
            self.report(f"{name} takes no argument")
        elif command.argument.startswith("<") and not arg:
            self.report(f"Usage: {name} {command.argument}")
        else:
            try:
                command.run(self, arg)
            except KeyboardInterrupt:
                self.report("Interrupted; returning to prompt")
            except Exception as exc:  # noqa: BLE001
                self.report(f"{name} failed: {type(exc).__name__}: {exc}")

    def ask(self, prompt: str) -> None:
        """Run one turn, printing its text as it arrives and reporting problems
        without leaving the loop."""
        self.last_prompt = prompt
        print()
        try:
            self.agent.run_turn(prompt, on_text=self.write, on_notice=self.notice)
        except TurnError as exc:
            self.notice(str(exc))
        except KeyboardInterrupt:
            self.notice("Interrupted; returning to prompt")
        except Exception as exc:  # noqa: BLE001
            self.notice(f"Agent error: {type(exc).__name__}: {exc}")
        self.end_answer()
        print()

    def write(self, text: str) -> None:
        """Print model text, remembering whether the cursor is at a line start."""
        print(text, end="", flush=True)
        if text:
            self.answer += text
            self.at_line_start = text.endswith("\n")

    def end_answer(self) -> None:
        """Finish the model text printed since the last notice, replacing it
        with its Markdown rendering when the terminal allows."""
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
        console.print(Markdown(text))

    def notice(self, text: str) -> None:
        """Print a bracketed status line, dimmed, on a line of its own."""
        self.end_answer()
        self.console.print(
            f"[{text}]", style="dim", markup=False, highlight=False, soft_wrap=True
        )

    def report(self, text: str) -> None:
        """Print a command's result as a notice followed by a blank line."""
        self.notice(text)
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
