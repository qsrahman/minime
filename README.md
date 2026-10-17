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

At the `>` prompt, describe a coding task. Enter `exit` or `quit` to end the session, or `/clear` to start a new conversation without restarting. While a streamed response is being prepared, `[thinking…]` is shown whenever the model reasons before answering. Press Ctrl-C while the agent is working to stop the current request and return to the prompt; press it at the prompt to quit. Steps the agent had already completed stay in the conversation, so you can ask it to continue.

Conversation history is kept in memory for the duration of the process, limited by default to 200,000 tokens (see `MAX_HISTORY_TOKENS`). That default suits the default model, which has a 400,000-token context window; set it lower for models with a smaller one. The token limit covers everything sent with a request, including the instructions and tool definitions, and is checked using the token counts the model reports. After each request MiniMe prints how much of it is in use, for example `[Context: 20,208 of 200,000 tokens, 18,944 cached; output: 312 tokens]`. The cached figure is the part of the request the provider had already processed and bills at a reduced rate; it still counts toward the limit. The output figure is everything the model generated during the request, including reasoning. When the limit is reached, history is cut back to three quarters of it, oldest requests first, and the model is asked for a short summary of what was removed, which takes their place; this costs one extra request and can be turned off with `SUMMARIZE=false`. If a single request is still too large, its oldest tool outputs are omitted. Cutting back further than strictly needed keeps the following requests cacheable, since every cut invalidates the provider's cache.

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
| `MAX_HISTORY_TOKENS`      | No       | `200000`          | Token limit for each request sent to the model.       |
| `REASONING_EFFORT`        | No       | Model default     | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. |
| `SUMMARIZE`               | No       | `true`            | Summarise history that is dropped to fit the limit.   |
| `STREAMING`               | No       | `true`            | Print the response as it is generated.                |
| `WORKSPACE_ROOT`          | No       | Current directory | Existing directory that file paths must stay inside.  |
| `BRAVE_API_KEY`           | No       |                   | Brave Search API key. Enables the `web_search` tool.  |

## Agent tools

- `read_file` reads a workspace file, optionally selecting a range of lines. Long files are returned up to 16 KB at a time, in whole lines, with the line number to continue from. A single line longer than that is returned in pieces.
- `write_file` creates or replaces a workspace file, or appends to it.
- `edit_file` replaces a text fragment; by default, the fragment must occur exactly once. Files with Windows (CRLF) line endings can be edited with text copied from `read_file`, and keep their line endings. If the fragment is missing but matches apart from whitespace, the error says so.
- `run_command` runs a shell command from the workspace, with an optional working directory and timeout. Commands get no standard input. stdout and stderr share a 16 KB limit; when a stream is clipped, its start and end are kept and the middle is omitted.
- `web_fetch` downloads a web page (up to 2 MB, within 20 seconds) and returns its text, 16 KB at a time. If the download exceeds 2 MB, the result sets `truncated` to `true`. HTML is reduced to readable text with links kept; navigation, headers, footers and sidebars are dropped.
- `web_search` searches the web with the [Brave Search API](https://brave.com/search/api/). It is only offered to the model when `BRAVE_API_KEY` is set.

File and working-directory paths are restricted to the workspace root. This is a path check, **not a security sandbox**: shell commands run with the same permissions as the current user and may access or modify other files or resources. Only use MiniMe with prompts and model providers you trust, and review commands and file changes carefully.

Web pages and search results are passed to the model as they are. A page can contain text written to steer the model, and the model can run shell commands without asking, so be careful which sites you point it at. `web_fetch` can also reach local and private addresses.

## Development

The project declares its runtime dependencies in `pyproject.toml`. Use `uv sync` to install them and `uv run main.py` to start the agent from the project directory. The code is eight modules: `config.py` (settings), `tools.py` (what all tools share: the `Tool` base class and the `Toolbox` that holds them), `fs.py`, `shell.py` and `web.py` (the file, command and web tools), `context.py` (`Context`, the conversation history), `agent.py` (`Agent`, the tool loop) and `main.py` (`Repl`, the terminal, and the entry point).

Run the tests with:

```sh
uv run pytest
```

They need no API key or network access: model responses are scripted and web pages are served from a local test server.

End-to-end tests against the real model in your `.env` are kept separate, because they make real API calls:

```sh
uv run pytest -m live
```

Lint and format with [ruff](https://docs.astral.sh/ruff/):

```sh
uv run ruff check
uv run ruff format
```
