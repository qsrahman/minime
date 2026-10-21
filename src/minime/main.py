import sys
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from minime.agent import Agent
from minime.config import Config
from minime.fs import EditFileTool, ReadFileTool, WriteFileTool
from minime.repl import Repl
from minime.shell import RunCommandTool
from minime.tools import Tool, Toolbox
from minime.web import WebFetchTool, WebSearchTool

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
