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
uv run minime
```

Alternatively, put `OPENAI_API_KEY=your-api-key` in a `.env` file in the MiniMe directory. MiniMe loads that file automatically, wherever it is started from. Do not commit your API key.

MiniMe works in the directory you start it from: relative paths and commands start there, and `~` stands for your home directory.

MiniMe starts with a line naming the model and the working directory. At the `›` prompt, describe a coding task, or enter a command:

| Command | Effect |
| --- | --- |
| `/help` | List the commands. |
| `/context` | Show how much of the context limit the last request used, how much of it was cached, the number of turns and the learned bytes per token. |
| `/model [name]` | Show the model, or switch to another one for the rest of the session. |
| `/effort [level]` | Show the reasoning effort, or set it: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or `default` to leave it to the model. |
| `/cd <dir>` | Change the working directory. |
| `/retry` | Send the last prompt again. |
| `/edit` | Remove the last turn and put its prompt back at the prompt, to change and send again. |
| `/undo` | Remove the last turn from the conversation. |
| `/compact` | Replace the conversation with a summary of it, to free space; it is kept unchanged if the summary fails. |
| `/export [file]` | Save the conversation as Markdown: prompts, answers and one line per tool call. The default file is `minime-<date>-<time>.md` in the working directory. |
| `/reset` | Start a new conversation without restarting (clears the history). |
| `/clear` | Clear the screen; the conversation is kept. |
| `/exit`, `/quit` | End the session. |

In a terminal the prompt has line editing: Up and Down recall earlier input, kept across sessions in `~/.minime_history`, and a matching earlier prompt is suggested in grey as you type (→ accepts it); typing `/` offers the commands with their descriptions, and Tab completes them, and directories after `/cd`; Esc then Enter starts a new line, Ctrl-X Ctrl-E opens the prompt in your `$EDITOR` for longer text, and pasted text stays one prompt however many lines it has. The line at the bottom shows the model, the reasoning effort and how much of the context limit the last request used, with these keys at the right when the terminal is wide enough. While the agent works, a spinner shows what it is waiting for (the model, or the tool call that is running), for how long, and that Ctrl-C stops it. Each tool call is then listed with how it ended and how long it took, such as `✓ read_file: src/app.py  0.0s` or `✗ run_command: pytest -q  exit 1  2.3s`, one row each, with a long description shortened to fit. Notices are dim, warnings (an interrupted or stopped turn) yellow and errors red; a common API error (no connection, a bad key, an unknown model, a rate limit) is followed by a hint on what to check. The terminal tab's title reads `minime · working…` during a turn, and a turn that took longer than 30 seconds rings the terminal bell when it ends, which most terminals show as a badge or notification. Answers stream as plain text and are redrawn as formatted Markdown once complete, at most 100 columns wide and with headings left-aligned, with code in the terminal's own colours; an answer taller than the screen stays plain, since its start has scrolled away.

Command names ignore case. A line that starts with a command runs it, with the rest of the line as its argument. A single word starting with `/` that is not a command is reported as unknown; other input, such as `/usr/bin is missing`, goes to the model. Press Ctrl-C while the agent is working to stop the current request and return to the prompt; at the prompt it clears the line, and a second press within two seconds quits (so does Ctrl-D). With fast-streaming models a Ctrl-C during an answer is sometimes lost (Python prints "Exception ignored … KeyboardInterrupt" and the answer continues); press it again. Steps the agent had already completed stay in the conversation, so you can ask it to continue.

Conversation history is kept in memory for the duration of the process, limited by default to 65,536 tokens (see `MAX_CONTEXT_TOKENS`). That fits most models' context windows; raise it for models with a larger one, such as the default model's 400,000 tokens. The token limit covers everything sent with a request, including the instructions and tool definitions, and is checked using the token counts the model reports. Tokens the provider has cached still count toward the limit. When the limit is reached, history is cut back to three quarters of it, oldest requests first, and the model is asked for a short summary of what was removed, which takes their place; this costs one extra request and can be turned off with `SUMMARIZE=false`. If a single request is still too large, its oldest tool outputs are omitted; if even that is not enough, the request is stopped and the conversation history is cleared. Cutting back further than strictly needed keeps the following requests cacheable, since every cut invalidates the provider's cache.

To work on another project, start MiniMe from that project's directory:

```sh
uv run --project /path/to/minime minime
```

Or install it once as a command you can run anywhere. Use `--editable`, so that it keeps reading the `.env` file in the MiniMe directory and picks up changes to the code:

```sh
uv tool install --editable /path/to/minime
minime
```

## Configuration

Configuration can be provided through command-line flags, environment variables or the `.env` file, in that order of precedence. Each variable below has a matching flag in lower case with dashes, for example `uv run minime --openai-model gpt-5.4 --max-steps-per-turn 8`; switches take `--streaming` / `--no-streaming` form. `uv run minime --help` lists them all. Prefer the environment or `.env` for API keys, since flags show up in shell history and process listings:

| Variable                  | Required | Default           | Description                                           |
| ------------------------- | -------- | ----------------- | ----------------------------------------------------- |
| `OPENAI_API_KEY`          | Yes      |                   | API key used to authenticate with the model provider. |
| `OPENAI_MODEL`            | No       | `gpt-5.4-mini`    | Model used for responses.                             |
| `OPENAI_BASE_URL`         | No       | Provider default  | Optional API-compatible endpoint.                     |
| `MAX_STEPS_PER_TURN`      | No       | `16`              | Maximum model requests (steps) for each user request. |
| `COMMAND_TIMEOUT_SECONDS` | No       | `30`              | Default timeout for shell commands, in seconds.       |
| `MAX_CONTEXT_TOKENS`      | No       | `65536`           | Token limit for each request sent to the model.       |
| `REASONING_EFFORT`        | No       | Model default     | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. |
| `SUMMARIZE`               | No       | `true`            | Summarise history that is dropped to fit the limit.   |
| `STREAMING`               | No       | `true`            | Print the response as it is generated.                |
| `BRAVE_API_KEY`           | No       |                   | Brave Search API key. Enables the `web_search` tool.  |

## Agent tools

- `read_file` reads a file, optionally selecting a range of lines. Long files are returned up to 16 KB at a time, in whole lines, with the line number to continue from. A single line longer than that is returned in pieces. Whenever the file continues past what was returned, including after a requested line range, the result says where to continue.
- `write_file` creates or replaces a file, or appends to it.
- `edit_file` replaces a text fragment; by default, the fragment must occur exactly once. Files with Windows (CRLF) line endings can be edited with text copied from `read_file`, and keep their line endings. If the fragment is missing but matches apart from whitespace, the error says so.
- `run_command` runs a shell command from the current directory, with an optional working directory and timeout. Commands get no standard input. stdout and stderr share a 16 KB limit; when a stream is clipped, its start and end are kept and the middle is omitted.
- `web_fetch` downloads a web page (up to 2 MB, within 20 seconds) and returns its text, 16 KB at a time. If the download exceeds 2 MB, the result sets `truncated` to `true`. HTML is reduced to readable text with links kept; navigation, headers, footers and sidebars are dropped.
- `web_search` searches the web with the [Brave Search API](https://brave.com/search/api/). It is only offered to the model when `BRAVE_API_KEY` is set.

The tools are **not restricted to that directory and not sandboxed**: files and shell commands can be read, written and run anywhere the current user has access. Only use MiniMe with prompts and model providers you trust, and review commands and file changes carefully.

Web pages and search results are passed to the model as they are. A page can contain text written to steer the model, and the model can run shell commands without asking, so be careful which sites you point it at. `web_fetch` can also reach local and private addresses.

## Development

The project declares its runtime dependencies in `pyproject.toml`. Use `uv sync` to install them and `uv run minime` to start the agent from the project directory. The code is the `minime` package in `src/minime/`, ten modules: `config.py` (settings), `tools.py` (what all tools share: the `Tool` base class and the `Toolbox` that holds them), `fs.py`, `shell.py` and `web.py` (the file, command and web tools), `context.py` (`Context`, the conversation history), `agent.py` (`Agent`, the tool loop), `commands.py` (the slash commands), `repl.py` (`Repl`, the terminal, using `prompt_toolkit` for input and `rich` for output) and `main.py` (the entry point, which builds everything and starts the terminal).

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
