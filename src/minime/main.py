import sys
from collections.abc import Callable

from pydantic import ValidationError

from minime.agent import Agent, TurnError
from minime.config import Config

try:
    import readline  # noqa: F401  # line editing and history for input()
except ImportError:  # not available on Windows
    pass


class Repl:
    """The interactive terminal: reads prompts and prints what the agent says."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.running = False
        self.at_line_start = True
        # Each command maps to its handler and the description shown by /help.
        self.commands: dict[str, tuple[Callable[[], None], str]] = {
            "/clear": (self.clear, "clear the screen"),
            "/exit": (self.stop, "end the session"),
            "/help": (self.help, "list commands"),
            "/quit": (self.stop, "end the session"),
            "/reset": (self.reset, "clear the conversation history"),
        }

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        config = self.agent.config
        print(
            f"AI coding agent started | model: {config.model} | "
            f"workspace: {config.workspace_root}\n"
            "Type /help to list commands.\n"
        )

        self.running = True
        while self.running:
            try:
                prompt = input("> ").strip()
            except KeyboardInterrupt, EOFError:
                break
            if not prompt:
                continue

            command = self.commands.get(prompt.lower())
            if command:
                handler, _ = command
                handler()
            # A single word starting with "/" is a mistyped command, not a
            # prompt; longer input such as "/usr/bin is missing" still goes
            # to the model.
            elif prompt.startswith("/") and " " not in prompt:
                print(f"[Unknown command: {prompt}. Type /help to list commands.]\n")
            else:
                self.ask(prompt)

        print("\nGoodbye!")

    def ask(self, prompt: str) -> None:
        """Run one turn, printing its text as it arrives and reporting problems
        without leaving the loop."""
        print()
        try:
            self.agent.run_turn(prompt, on_text=self.write, on_notice=self.notice)
        except TurnError as exc:
            self.notice(str(exc))
        except KeyboardInterrupt:
            self.notice("Interrupted; returning to prompt")
        except Exception as exc:  # noqa: BLE001
            self.notice(f"Agent error: {type(exc).__name__}: {exc}")
        print()

    def write(self, text: str) -> None:
        """Print model text, remembering whether the cursor is at a line start."""
        print(text, end="", flush=True)
        if text:
            self.at_line_start = text.endswith("\n")

    def notice(self, text: str) -> None:
        """Print a bracketed status line, starting a new line if needed."""
        if not self.at_line_start:
            self.write("\n")
        print(f"[{text}]")

    def help(self) -> None:
        """List the available commands."""
        width = max(len(name) for name in self.commands)
        for name, (_, description) in self.commands.items():
            print(f"  {name:<{width}}  {description}")
        print()

    def reset(self) -> None:
        """Clear the conversation history."""
        self.agent.context.reset()
        print("[History reset]\n")

    def clear(self) -> None:
        """Clear the terminal screen; the history is kept."""
        # Clear the screen and scrollback, then move the cursor home, like
        # `clear`; not when output is redirected, where the codes are garbage.
        if sys.stdout.isatty():
            print("\033[2J\033[3J\033[H", end="", flush=True)

    def stop(self) -> None:
        """Leave the loop after the current command."""
        self.running = False


def main() -> None:
    """Load the configuration, exiting with a readable error if it is invalid,
    and run the terminal loop."""
    try:
        config = Config()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None

    Repl(Agent(config)).run()


if __name__ == "__main__":
    main()
