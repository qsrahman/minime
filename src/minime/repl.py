"""The terminal: the REPL that reads prompts and slash commands and shows
what the agent does."""

import inspect
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from textwrap import shorten
from typing import Any, ClassVar

from openai import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
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
from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Heading, Markdown, MarkdownElement
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.theme import Theme

from minime.agent import Agent, TurnError
from minime.commands import Command, standard_commands

HISTORY_FILE = Path("~/.minime_history")
PROMPT = "› "
# Shown at the right of the status bar when there is room.
KEY_HINTS = "esc ⏎ newline · ^X^E editor · → accept suggestion · ^D quit"
# A second Ctrl-C at the prompt within this many seconds quits.
QUIT_SECONDS = 2
# Answers wider than this are hard to read on a wide terminal.
MAX_ANSWER_WIDTH = 100
# A turn that took longer rings the bell when it ends, for whoever switched away.
LONG_TURN_SECONDS = 30
TITLE = "minime"

# What to check for the API errors people commonly hit; the first match wins,
# so subclasses (APITimeoutError is an APIConnectionError) come first if listed.
API_ERROR_HINTS: list[tuple[type[OpenAIError], str]] = [
    (APIConnectionError, "Check the network, and OPENAI_BASE_URL if you set one."),
    (AuthenticationError, "Check OPENAI_API_KEY."),
    (PermissionDeniedError, "This API key may not have access to the model."),
    (NotFoundError, "Check the model name (/model) and OPENAI_BASE_URL."),
    (
        RateLimitError,
        "Rate limit or quota reached: wait, or check the account's usage.",
    ),
    (InternalServerError, "The provider failed; /retry in a moment."),
]

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


class LeftHeading(Heading):
    """A heading drawn left-aligned, like the rest of the answer (Rich
    centres level 1)."""

    LEVEL_ALIGN: ClassVar = {}  # every level falls back to "left"


class AnswerMarkdown(Markdown):
    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "heading_open": LeftHeading,
    }


def home_relative(path: Path) -> str:
    """The path with the home directory shortened to "~"."""
    if not path.is_relative_to(Path.home()):
        return str(path)
    return str(Path("~") / path.relative_to(Path.home()))


class CommandCompleter(Completer):
    """Completes a command name at the start of the line, shown with its
    /help text, and a directory after /cd. (NestedCompleter does neither: its
    words cannot start with "/".)"""

    def __init__(self, commands: dict[str, Command]) -> None:
        self.names = WordCompleter(
            list(commands),
            ignore_case=True,
            WORD=True,
            meta_dict={
                name: inspect.getdoc(command) for name, command in commands.items()
            },
        )
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
        self.last_prompt: str | None = None  # for /retry
        self.draft = ""  # text the next prompt starts with (/edit)
        self.commands: dict[str, Command] = {
            name: command for command in standard_commands() for name in command.names
        }
        self.console = Console(theme=THEME)
        # Model text since the last notice, re-rendered as Markdown once complete;
        # it also tells whether the cursor is at the start of a line.
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
        self.terminal_control("\033[22;0t")  # save the title, restored on exit
        self.set_title(TITLE)

        session = self.prompt_session() if sys.stdin.isatty() else None
        last_ctrl_c = -math.inf
        self.running = True
        while self.running:
            try:
                if session:
                    line = session.prompt(default=self.draft).strip()
                else:
                    line = input(PROMPT).strip()
            except EOFError:
                print()  # Ctrl-D leaves the cursor on the prompt line
                break
            except KeyboardInterrupt:
                # The first Ctrl-C only clears the line, so a stray one (often
                # meant for the turn that just ended) does not lose the session.
                if time.monotonic() - last_ctrl_c < QUIT_SECONDS:
                    print()
                    break
                last_ctrl_c = time.monotonic()
                self.console.print(
                    "Press Ctrl-C again to quit, or Ctrl-D.", style="dim"
                )
                continue
            finally:
                self.draft = ""
            if not line:
                continue
            try:
                self.handle(line)
            except KeyboardInterrupt:
                # One that lands outside a turn or command, e.g. while an
                # interruption is being reported.
                self.notice("Interrupted", "yellow")
        print("Bye.")
        self.terminal_control("\033[23;0t")

    def handle(self, line: str) -> None:
        """Run a command line, or send the line to the model."""
        name, _, arg = line.partition(" ")
        name = name.lower()
        command = self.commands.get(name)
        if command:
            self.run_command(name, command, arg.strip())
        # A single word starting with "/" is a mistyped command, not a prompt;
        # longer input such as "/usr/bin is missing" still goes to the model.
        elif line.startswith("/") and " " not in line:
            self.report(
                f"Unknown command: {line}. Type /help to list commands.", "yellow"
            )
        else:
            self.ask(line)

    def prompt_session(self) -> PromptSession[str]:
        """Line editing for a terminal: history kept across sessions and
        suggested as you type, completion of commands (and of directories
        after /cd), Esc then Enter for a new line, Ctrl-X Ctrl-E to write the
        prompt in $EDITOR, and a status bar."""
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        return PromptSession(
            message=[("class:prompt", PROMPT)],
            # The toolbar is dim text rather than the default reverse band.
            style=Style.from_dict(
                {"prompt": "ansicyan bold", "bottom-toolbar": "noreverse dim"}
            ),
            history=FileHistory(HISTORY_FILE.expanduser()),
            enable_open_in_editor=True,  # Ctrl-X Ctrl-E opens $EDITOR
            auto_suggest=AutoSuggestFromHistory(),
            completer=CommandCompleter(self.commands),
            key_bindings=bindings,
            bottom_toolbar=lambda: self.toolbar(get_app().output.get_size().columns),
        )

    def toolbar(self, width: int) -> str:
        """The status, with the key hints at the right edge if they fit."""
        status = self.status()
        gap = width - len(status) - len(KEY_HINTS) - 1
        return f"{status}{' ' * gap}{KEY_HINTS}" if gap >= 2 else status

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
        self.set_title(f"{TITLE} · working…")
        started = time.monotonic()
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
            hint = next(
                (hint for kind, hint in API_ERROR_HINTS if isinstance(exc, kind)), None
            )
            if hint:
                self.console.print(f"  {hint}", style="dim", highlight=False)
        self.end_answer()
        print()
        self.set_title(TITLE)
        if (
            time.monotonic() - started > LONG_TURN_SECONDS
            and self.console.is_interactive
        ):
            self.console.bell()

    def set_title(self, title: str) -> None:
        """Set the terminal window or tab title."""
        self.terminal_control(f"\033]0;{title}\007")

    def terminal_control(self, code: str) -> None:
        """Write an escape code, only to a terminal that understands them."""
        if self.console.is_interactive:
            print(code, end="", flush=True)

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
            TextColumn("· ctrl-c to stop", style="dim", markup=False),
            console=self.console,
            transient=True,
        )
        # Room for the spinner, elapsed time and hint on the same row.
        self.spinner.add_task(self.fit(label, 30), total=None)
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
        if result.get("ok"):
            mark, failure = Text("✓ ", "green"), ""
        else:
            if result.get("timed_out"):
                failure = "timed out"
            elif result.get("returncode") is not None:
                failure = f"exit {result['returncode']}"
            else:
                message = str(result.get("message", "failed"))
                failure = shorten(message, 60, placeholder=" …")
            mark, failure = Text("✗ ", "red"), f"  {failure}"
        duration = f"  {time.monotonic() - started:.1f}s"
        # One row each in a terminal: shorten the description, not the rest.
        if self.console.is_interactive:
            description = self.fit(description, 2 + len(failure) + len(duration))
        line = Text.assemble(
            mark, (description, "dim"), (failure, "red"), (duration, "dim")
        )
        self.console.print(line, highlight=False, soft_wrap=True)

    def fit(self, text: str, reserved: int) -> str:
        """The text, cut with "…" to fit the console width less reserved columns."""
        fitted = Text(text)
        fitted.truncate(max(10, self.console.width - reserved), overflow="ellipsis")
        return fitted.plain

    def write(self, text: str) -> None:
        """Print model text as it arrives, keeping it for end_answer()."""
        self.stop_spinner()
        print(text, end="", flush=True)
        self.answer += text

    def end_answer(self) -> None:
        """Finish the model text printed since the last notice, replacing it
        with its Markdown rendering when the terminal allows."""
        self.stop_spinner()
        if self.answer and not self.answer.endswith("\n"):
            self.write("\n")
        text, self.answer = self.answer, ""
        console = self.console
        if not text.strip() or not console.is_interactive:
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
            AnswerMarkdown(text, code_theme=CODE_THEME),
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
