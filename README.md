# MiniMe

MiniMe is a small, interactive AI coding agent that works in the directory you start it from. It uses the OpenAI Responses API and can inspect and edit files, run shell commands, and search and read the web in response to your requests.

## Requirements

- Python 3.14 or newer
- An API key for OpenAI or an OpenAI-compatible endpoint

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

Alternatively, put `OPENAI_API_KEY=your-api-key` in a `.env` file in the MiniMe directory. MiniMe loads that file automatically, wherever it is started from. Do not commit your API key.

At the `>` prompt, describe a coding task. Enter `exit` or `quit` to end the session. Press Ctrl-C while the agent is working to stop the current request and return to the prompt; press it at the prompt to quit. Steps the agent had already completed stay in the conversation, so you can ask it to continue.

Conversation history is kept in memory for the duration of the process, limited by default to the 12 most recent requests and 256 KB (see `MAX_HISTORY_TURNS` and `MAX_HISTORY_BYTES`). Older requests are dropped first; if a single request is still too large, its oldest tool outputs are omitted.

To work on another project, start MiniMe from that project's directory:

```sh
uv run --project /path/to/minime /path/to/minime/main.py
```

## Configuration

Configuration can be provided through environment variables or the `.env` file. Environment variables take precedence:

| Variable                  | Required | Default           | Description                                           |
| ------------------------- | -------- | ----------------- | ----------------------------------------------------- |
| `OPENAI_API_KEY`          | Yes      |                   | API key used to authenticate with the model provider. |
| `OPENAI_MODEL`            | No       | `gpt-5.4-mini`    | Model used for responses.                             |
| `OPENAI_BASE_URL`         | No       | Provider default  | Optional API-compatible endpoint.                     |
| `MAX_TURNS_PER_RUN`       | No       | `16`              | Maximum model/tool iterations for each user request.  |
| `COMMAND_TIMEOUT_SECONDS` | No       | `30`              | Default timeout for shell commands, in seconds.       |
| `MAX_HISTORY_TURNS`       | No       | `12`              | Number of recent requests kept in the conversation.   |
| `MAX_HISTORY_BYTES`       | No       | `262144`          | Size limit for the conversation sent to the model.    |
| `STREAMING`               | No       | `true`            | Print the response as it is generated.                |
| `WORKSPACE_ROOT`          | No       | Current directory | File and working-directory paths must stay inside it. |
| `BRAVE_API_KEY`           | No       |                   | Brave Search API key. Enables the `web_search` tool.  |

## Agent tools

- `read_file` reads a workspace file, optionally selecting a range of lines. Long files are returned up to 16 KB at a time, in whole lines, with the line number to continue from.
- `write_file` creates or replaces a workspace file, or appends to it.
- `edit_file` replaces a text fragment; by default, the fragment must occur exactly once.
- `run_command` runs a shell command from the workspace, with an optional working directory and timeout. Commands get no standard input. When stdout or stderr exceeds 16 KB, the first and last 8 KB are kept and the middle is omitted.
- `web_fetch` downloads a web page (up to 2 MB, within 20 seconds) and returns its text, 16 KB at a time. If the download exceeds 2 MB, the result sets `truncated` to `true`. HTML is reduced to readable text with links kept; navigation, headers, footers and sidebars are dropped.
- `web_search` searches the web with the [Brave Search API](https://brave.com/search/api/). It is only offered to the model when `BRAVE_API_KEY` is set.

File and working-directory paths are restricted to the workspace root. This is a path check, **not a security sandbox**: shell commands run with the same permissions as the current user and may access or modify other files or resources. Only use MiniMe with prompts and model providers you trust, and review commands and file changes carefully.

Web pages and search results are passed to the model as they are. A page can contain text written to steer the model, and the model can run shell commands without asking, so be careful which sites you point it at. `web_fetch` can also reach local and private addresses.

## Development

The project declares its runtime dependencies in `pyproject.toml`. Use `uv sync` to install them and `uv run main.py` to start the agent from the project directory. The code is three modules: `config.py` (settings), `tools.py` (the agent's tools) and `main.py` (the conversation loop).

Run the tests with:

```sh
uv run pytest
```

They need no API key or network access: model responses are scripted and web pages are served from a local test server.
