"""The slash commands typed at the REPL prompt, such as /help or /model."""

import inspect
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast, get_args

from openai.types.shared import ReasoningEffort

if TYPE_CHECKING:
    from minime.main import Repl

# The levels /effort accepts: the API's own, plus "default" to send none.
EFFORT_LEVELS = (*get_args(get_args(ReasoningEffort)[0]), "default")


class Command(ABC):
    """A command the user types at the prompt.

    A subclass's docstring is its description in /help. Results are shown
    with repl.report(); Repl reports any exception a command raises.
    """

    names: ClassVar[tuple[str, ...]]  # lower case, with the leading "/"
    # Shown in /help: "" takes no argument, "[x]" an optional one, "<x>" a
    # required one.
    argument: ClassVar[str] = ""

    @abstractmethod
    def run(self, repl: Repl, arg: str) -> None:
        """Carry out the command; arg is the rest of the line, stripped."""


class HelpCommand(Command):
    """list commands"""

    names = ("/help",)

    def run(self, repl: Repl, arg: str) -> None:
        rows = [
            (f"{name} {command.argument}".strip(), inspect.getdoc(command))
            for name, command in repl.commands.items()
        ]
        width = max(len(usage) for usage, _ in rows)
        for usage, text in rows:
            print(f"  {usage:<{width}}  {text}")
        print()


class ExitCommand(Command):
    """end the session"""

    names = ("/exit", "/quit")

    def run(self, repl: Repl, arg: str) -> None:
        repl.running = False


class ClearCommand(Command):
    """clear the screen; the conversation is kept"""

    names = ("/clear",)

    def run(self, repl: Repl, arg: str) -> None:
        # Clear the screen and scrollback, then move the cursor home, like
        # `clear`; not when output is redirected, where the codes are garbage.
        if sys.stdout.isatty():
            print("\033[2J\033[3J\033[H", end="", flush=True)


class ResetCommand(Command):
    """clear the conversation history"""

    names = ("/reset",)

    def run(self, repl: Repl, arg: str) -> None:
        repl.agent.reset()
        repl.report("History reset")


class UndoCommand(Command):
    """remove the last turn from the history"""

    names = ("/undo",)

    def run(self, repl: Repl, arg: str) -> None:
        repl.report("Last turn removed" if repl.agent.undo() else "Nothing to undo")


class RetryCommand(Command):
    """send the last prompt again"""

    names = ("/retry",)

    def run(self, repl: Repl, arg: str) -> None:
        if repl.last_prompt is None:
            repl.report("Nothing to retry")
        else:
            repl.ask(repl.last_prompt)


class CompactCommand(Command):
    """replace the history with a summary of it"""

    names = ("/compact",)

    def run(self, repl: Repl, arg: str) -> None:
        if repl.agent.compact(on_notice=repl.notice):
            repl.report("History compacted")
        else:
            print()


class ContextCommand(Command):
    """show how much of the context limit is in use"""

    names = ("/context",)

    def run(self, repl: Repl, arg: str) -> None:
        agent = repl.agent
        limit = agent.config.max_context_tokens
        usage = agent.last_usage
        if usage and usage.input_tokens:
            used = f"{usage.input_tokens:,} of {limit:,} tokens"
            used += f" ({usage.input_tokens / limit:.0%})"
            details = usage.input_tokens_details
            if details and details.cached_tokens:
                used += f", {details.cached_tokens:,} cached"
        else:
            used = f"limit {limit:,} tokens, no request since the last change"
        bytes_per_token = 1 / agent.context.tokens_per_byte
        repl.report(
            f"Context: {used}; {agent.context.turns()} turns; "
            f"{bytes_per_token:.1f} bytes per token"
        )


class CdCommand(Command):
    """change the working directory"""

    names = ("/cd",)
    argument = "<dir>"

    def run(self, repl: Repl, arg: str) -> None:
        os.chdir(Path(arg).expanduser())
        repl.report(f"Working directory: {Path.cwd()}")


class ModelCommand(Command):
    """show or switch the model"""

    names = ("/model",)
    argument = "[name]"

    def run(self, repl: Repl, arg: str) -> None:
        if arg:
            repl.agent.model = arg
            repl.report(f"Model set to {arg}")
        else:
            repl.report(f"Model: {repl.agent.model}")


class EffortCommand(Command):
    """show or set the reasoning effort"""

    names = ("/effort",)
    argument = "[level]"

    def run(self, repl: Repl, arg: str) -> None:
        if not arg:
            effort = repl.agent.reasoning_effort or "model default"
            repl.report(f"Reasoning effort: {effort}")
            return
        level = arg.lower()
        if level not in EFFORT_LEVELS:
            repl.report(
                f"Unknown effort {arg!r}; use one of: {', '.join(EFFORT_LEVELS)}"
            )
            return
        repl.agent.reasoning_effort = (
            None if level == "default" else cast(ReasoningEffort, level)
        )
        repl.report(f"Reasoning effort set to {level}")


def standard_commands() -> list[Command]:
    """Every command, in the order /help lists them."""
    return [
        HelpCommand(),
        ContextCommand(),
        ModelCommand(),
        EffortCommand(),
        CdCommand(),
        RetryCommand(),
        UndoCommand(),
        CompactCommand(),
        ResetCommand(),
        ClearCommand(),
        ExitCommand(),
    ]
