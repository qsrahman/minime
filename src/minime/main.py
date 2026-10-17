import signal
import sys
from types import FrameType

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
        self.at_line_start = True

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        config = self.agent.config
        print(
            f"AI coding agent started | model: {config.model} | "
            f"workspace: {config.workspace_root}\n"
            "Type '/exit' or '/quit' to stop, '/reset' to start a new conversation.\n"
        )

        while True:
            try:
                prompt = input("> ").strip()
            except KeyboardInterrupt, EOFError:
                break

            if not prompt:
                continue

            command = prompt.lower()
            if command in {"/exit", "/quit"}:
                break
            if command == "/reset":
                self.agent.context.reset()
                print("[Conversation cleared]\n")
                continue

            self.ask(prompt)

        print("\nGoodbye!")

    def ask(self, prompt: str) -> None:
        """Run one turn, printing its text as it arrives and reporting problems
        without leaving the loop."""
        previous_handler = signal.signal(signal.SIGINT, self._interrupt)
        previous_hook = sys.unraisablehook

        def quiet_lost_interrupt(unraisable: sys.UnraisableHookArgs) -> None:
            # A Ctrl-C swallowed by library code; stop_requested handles it,
            # so its "Exception ignored" traceback is only noise.
            if not isinstance(unraisable.exc_value, KeyboardInterrupt):
                previous_hook(unraisable)

        sys.unraisablehook = quiet_lost_interrupt
        try:
            self.agent.run_turn(prompt, on_text=self.write, on_notice=self.notice)
        except TurnError as exc:
            self.notice(str(exc))
        except KeyboardInterrupt:
            self.notice("Interrupted; returning to prompt")
        except Exception as exc:  # noqa: BLE001
            self.notice(f"Agent error: {type(exc).__name__}: {exc}")
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            sys.unraisablehook = previous_hook

        usage = self.agent.usage
        if usage.context_tokens:
            cached = f", {usage.cached_tokens:,} cached" if usage.cached_tokens else ""
            self.notice(
                f"Context: {usage.context_tokens:,} of "
                f"{self.agent.config.max_history_tokens:,} tokens{cached}; "
                f"output: {usage.output_tokens:,} tokens"
            )
        print()

    def _interrupt(self, signum: int, frame: FrameType | None) -> None:
        """Ctrl-C during a turn: stop the agent even if the exception is lost."""
        self.agent.stop_requested = True
        raise KeyboardInterrupt

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
