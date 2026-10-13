# MiniMe

MiniMe is a small, interactive AI coding agent that works in the directory you start it from. It uses the OpenAI Responses API and can inspect and edit files, run shell commands, and search and read the web in response to your requests.

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

Alternatively, put `OPENAI_API_KEY=your-api-key` in a `.env` file next to `main.py`. MiniMe loads that file automatically, wherever it is started from. Do not commit your API key.

At the `>` prompt, describe a coding task. Enter `exit` or `quit` to end the session. Press Ctrl-C while the agent is working to cancel the current request and return to the prompt; press it at the prompt to quit.

Conversation history is kept in memory for the duration of the process, limited to the 12 most recent requests and 256 KB. Older requests are dropped first; if a single request is still too large, its oldest tool outputs are omitted.

To work on another project, start MiniMe from that project's directory:

```sh
uv run --project /path/to/minime /path/to/minime/main.py
```

## Configuration

Configuration can be provided through environment variables or the `.env` file. Environment variables take precedence:

| Variable                  | Required | Default           | Description                                              |
| ------------------------- | -------- | ----------------- | -------------------------------------------------------- |
| `OPENAI_API_KEY`          | Yes      |                   | API key used to authenticate with the model provider.    |
| `OPENAI_MODEL`            | No       | `gpt-5.4-mini`    | Model used for responses.                                |
| `OPENAI_BASE_URL`         | No       | Provider default  | Optional API-compatible endpoint.                        |
| `MAX_TURNS_PER_RUN`       | No       | `16`              | Maximum model/tool iterations for each user request.     |
| `COMMAND_TIMEOUT_SECONDS` | No       | `30`              | Default timeout for shell commands, in seconds.          |
| `STREAMING`               | No       | `true`            | Print the response as it is generated.                   |
| `WORKSPACE_ROOT`          | No       | Current directory | File and working-directory paths must stay inside it.    |
| `BRAVE_API_KEY`           | No       |                   | Brave Search API key. Enables the `web_search` tool.     |

## Agent tools

- `read_file` reads a workspace file, optionally selecting a range of lines. Results over 16 KB are truncated.
- `write_file` creates or replaces a workspace file, or appends to it.
- `edit_file` replaces a text fragment; by default, the fragment must occur exactly once.
- `run_command` runs a shell command from the workspace, with an optional working directory and timeout. Commands get no standard input, and stdout and stderr are each truncated at 16 KB.
- `web_fetch` downloads a web page (up to 2 MB) and returns its text, 16 KB at a time. HTML is reduced to visible text; links are not kept.
- `web_search` searches the web with the [Brave Search API](https://brave.com/search/api/). It is only offered to the model when `BRAVE_API_KEY` is set.

File and working-directory paths are restricted to the workspace root. This is a path check, **not a security sandbox**: shell commands run with the same permissions as the current user and may access or modify other files or resources. Only use MiniMe with prompts and model providers you trust, and review commands and file changes carefully.

Web pages and search results are passed to the model as they are. A page can contain text written to steer the model, and the model can run shell commands without asking, so be careful which sites you point it at. `web_fetch` can also reach local and private addresses.

## Development

The project declares its runtime dependencies in `pyproject.toml`. Use `uv sync` to install them and `uv run main.py` to start the agent from the project directory. There is no test suite.
