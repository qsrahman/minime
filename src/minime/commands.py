"""The slash commands typed at the REPL prompt, such as /help or /model."""

import inspect
import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast, get_args

from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
from openai.types.shared import ReasoningEffort

if TYPE_CHECKING:
    from minime.agent import Agent
    from minime.repl import Repl

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
        undone = repl.agent.undo() is not None
        repl.report("Last turn removed" if undone else "Nothing to undo")


class EditCommand(Command):
    """remove the last turn and put its prompt back to edit and resend"""

    names = ("/edit",)

    def run(self, repl: Repl, arg: str) -> None:
        prompt = repl.agent.undo()
        if prompt is None:
            repl.report("Nothing to edit")
        else:
            repl.draft = prompt


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


class ExportCommand(Command):
    """save the conversation as Markdown"""

    names = ("/export",)
    argument = "[file]"

    def run(self, repl: Repl, arg: str) -> None:
        if not repl.agent.context.history:
            repl.report("Nothing to export")
            return
        path = Path(arg or f"minime-{datetime.now():%Y%m%d-%H%M%S}.md").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(transcript(repl.agent), encoding="utf-8")
        repl.report(f"Conversation exported to {path}")


def transcript(agent: Agent) -> str:
    """The conversation as Markdown: each prompt (or summary) quoted, the
    model's text, and one line per tool call with whether it succeeded."""
    history = cast(list[Any], agent.context.history)
    succeeded = {
        item["call_id"]: json.loads(item["output"]).get("ok", False)
        for item in history
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }
    lines = [f"# MiniMe conversation, {datetime.now():%Y-%m-%d %H:%M}", ""]
    lines += [f"Model: {agent.model}. Working directory: {Path.cwd()}.", ""]
    for item in history:
        if isinstance(item, dict) and item.get("role") == "user":
            quoted = str(item["content"]).replace("\n", "\n> ")
            lines += ["---", "", f"> {quoted}", ""]
        elif isinstance(item, ResponseOutputMessage):
            text = "".join(getattr(part, "text", "") for part in item.content)
            lines += [text.strip(), ""]
        elif isinstance(item, ResponseFunctionToolCall):
            mark = "✓" if succeeded.get(item.call_id) else "✗"
            lines += [f"- {mark} `{agent.toolbox.describe(item)}`", ""]
    return "\n".join(lines)


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
    """Every command; /help lists them by name."""
    return [
        HelpCommand(),
        ContextCommand(),
        ModelCommand(),
        EffortCommand(),
        CdCommand(),
        RetryCommand(),
        EditCommand(),
        UndoCommand(),
        CompactCommand(),
        ExportCommand(),
        ResetCommand(),
        ClearCommand(),
        ExitCommand(),
    ]
