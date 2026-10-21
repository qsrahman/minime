import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import Any, cast

from openai.types.responses import (
    EasyInputMessageParam,
    ResponseFunctionToolCall,
    ResponseInputParam,
)

# Used until the model reports real token counts: about four bytes per token.
DEFAULT_TOKENS_PER_BYTE = 0.25
# Once history passes the limit it is cut back to this share of it, not just to
# the limit. Each cut changes the start of the prompt, which discards the
# provider's prompt cache; cutting further leaves many requests before the next.
TRIM_TARGET = 0.75

# The text that stands in for a dropped summary or tool output; the tests
# import it rather than repeating it.
SUMMARY_MARKER = (
    "[Summary of the earlier conversation, which was removed to save space]"
)
OMITTED_TOOL_OUTPUT = (
    "[Tool output omitted to fit the context budget; "
    "call the tool again if it is still needed.]"
)


def json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def is_prompt(item: object) -> bool:
    """Whether a history item is a user message, which starts a turn."""
    return isinstance(item, dict) and item.get("role") == "user"


@dataclass
class Context:
    """The message history and how many tokens the model counts per byte of it.

    Every function_call is followed by its function_call_output, because the
    API rejects history otherwise; add_step records them together.
    """

    history: ResponseInputParam = field(default_factory=list)
    tokens_per_byte: float = DEFAULT_TOKENS_PER_BYTE

    def reset(self) -> None:
        """Forget the history; the learned tokens per byte still apply."""
        self.history.clear()

    def byte_budget(self, max_tokens: int, overhead_bytes: int) -> int:
        """The bytes of history that fit in max_tokens, after overhead_bytes
        sent with every request (instructions and tool schemas)."""
        return int(max_tokens / self.tokens_per_byte) - overhead_bytes

    def learn_tokens_per_byte(self, input_tokens: int, request_bytes: int) -> None:
        """Learn the model's ratio from the tokens it counted for a request.

        Cached tokens are part of input_tokens and still fill the context
        window, so they count. The bounds guard against nonsense reports.
        """
        self.tokens_per_byte = min(1.0, max(0.1, input_tokens / request_bytes))

    def add_prompt(self, prompt: str) -> EasyInputMessageParam:
        """Start a turn with the user's message, which is returned."""
        message: EasyInputMessageParam = {"role": "user", "content": prompt}
        self.history.append(message)
        return message

    def add_step(
        self,
        output: Sequence[Any],
        tool_calls: Sequence[ResponseFunctionToolCall],
        results: Sequence[dict[str, Any]],
    ) -> None:
        """Record a response's output items and one output per tool call.

        Calls without a result, because Ctrl-C interrupted them, are recorded
        as interrupted.
        """
        interrupted = {"ok": False, "message": "Interrupted by the user."}
        self.history.extend(cast(ResponseInputParam, output))
        self.history.extend(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            }
            for call, result in zip(
                tool_calls, chain(results, repeat(interrupted)), strict=False
            )
        )

    def add_summary(self, summary: str, max_bytes: int) -> bool:
        """Put a summary of dropped turns at the start of the history.

        It is a user message, so a later trim() drops it like any turn and the
        next summary covers it. A summary that is empty or would push the
        history past max_bytes is left out. Returns whether it was added.
        """
        summary = summary.strip()
        message: EasyInputMessageParam = {
            "role": "user",
            "content": f"{SUMMARY_MARKER}\n{summary}",
        }
        if not summary or json_bytes([message, *self.history]) > max_bytes:
            return False
        self.history.insert(0, message)
        return True

    def replace_with_summary(self, summary: str, max_bytes: int) -> bool:
        """Replace the whole history with a summary of it, unless the summary
        is empty or does not fit; returns whether it did."""
        previous = self.history[:]
        self.history.clear()
        if self.add_summary(summary, max_bytes):
            return True
        self.history[:] = previous
        return False

    def _turn_starts(self) -> list[int]:
        """Where each turn, a user message, starts in the history."""
        return [index for index, item in enumerate(self.history) if is_prompt(item)]

    def undo(self) -> bool:
        """Drop the last turn: the last user message and everything after it.

        Returns whether there was one to drop.
        """
        starts = self._turn_starts()
        if not starts:
            return False
        del self.history[starts[-1] :]
        return True

    def turns(self) -> int:
        """How many turns the history holds (a leading summary counts as one)."""
        return len(self._turn_starts())

    def discard_unanswered(self, prompt: EasyInputMessageParam) -> None:
        """Drop the turn's prompt if no step was recorded after it."""
        if self.history and self.history[-1] is prompt:
            self.history.pop()

    def trim(self, max_bytes: int) -> tuple[ResponseInputParam, bool, bool]:
        """Trim the history in place to fit max_bytes.

        Returns the whole turns dropped from the front (oldest first), whether
        anything was trimmed, and whether the current request itself had to be
        dropped, in which case the history is emptied and no turns are returned.
        """
        history = self.history
        if json_bytes(history) <= max_bytes:
            return [], False, False

        dropped: ResponseInputParam = []
        trimmed = False
        target_bytes = int(max_bytes * TRIM_TARGET)
        while json_bytes(history) > target_bytes:
            starts = self._turn_starts()
            if len(starts) > 1:
                dropped += history[: starts[1]]
                del history[: starts[1]]
                trimmed = True
                continue

            # Only the current request is left: free space by omitting its oldest
            # tool outputs.
            oldest_output = next(
                (
                    item
                    for item in history
                    if isinstance(item, dict)
                    and item.get("type") == "function_call_output"
                    and item.get("output") != OMITTED_TOOL_OUTPUT
                ),
                None,
            )
            if oldest_output is None:
                break  # nothing left that can be dropped
            cast(dict[str, Any], oldest_output)["output"] = OMITTED_TOOL_OUTPUT
            trimmed = True

        # The request itself is over the limit: give up on it.
        if json_bytes(history) > max_bytes:
            history.clear()
            return [], True, True
        return dropped, trimmed, False
