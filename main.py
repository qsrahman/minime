import sys

from pydantic import ValidationError

from agent import Agent
from config import Config
from repl import Repl


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
