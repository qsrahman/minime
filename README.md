# MiniMe

MiniMe is a small, interactive AI coding agent that works in the directory containing the project. It uses the OpenAI Responses API and can inspect and edit files or run shell commands in response to your requests.

## Requirements

- Python 3.14 or newer
- An OpenAI API key

## Setup

Install the project dependencies:

```sh
uv sync
```

Set the required API key, then start the agent:

```sh
export OPENAI_API_KEY="your-api-key"
uv run main.py
```

Alternatively, put `OPENAI_API_KEY=your-api-key` in a `.env` file in the project directory. MiniMe loads that file automatically. Do not commit your API key.

At the `>` prompt, describe a coding task. Enter `exit` or `quit` to end the session. Conversation history is kept for the duration of the process.

## Configuration

Configuration can be provided through environment variables or a `.env` file:

| Variable                  | Required | Default          | Description                                           |
| ------------------------- | -------- | ---------------- | ----------------------------------------------------- |
| `OPENAI_API_KEY`          | Yes      |                  | API key used to authenticate with the model provider. |
| `OPENAI_MODEL`            | No       | `gpt-5.4-mini`   | Model used for responses.                             |
| `OPENAI_BASE_URL`         | No       | Provider default | Optional API-compatible endpoint.                     |
| `MAX_TURNS_PER_RUN`       | No       | `16`             | Maximum model/tool iterations for each user request.  |
| `COMMAND_TIMEOUT_SECONDS` | No       | `30`             | Default timeout for shell commands, in seconds.       |
| `STREAMING`               | No       | `true`           | Print the response as it is generated.                |
| `WORKSPACE_ROOT`          | No       | Current directory | Directory the file tools and commands are limited to. |

## Agent tools

- `read_file` reads a workspace file, optionally selecting a range of lines.
- `write_file` creates or replaces a workspace file, or appends to it.
- `edit_file` replaces a text fragment; by default, the fragment must occur exactly once.
- `run_command` runs a shell command from the workspace, with an optional working directory and timeout.

File and working-directory paths are restricted to the workspace root. This is a path check, **not a security sandbox**: shell commands run with the same permissions as the current user and may access or modify other files or resources. Only use MiniMe with prompts and model providers you trust, and review commands and file changes carefully.

## Development

The project declares its runtime dependencies in `pyproject.toml`. Use `uv sync` to install them and `uv run main.py` to start the agent from the project directory.
