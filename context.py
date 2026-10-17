import json
from collections.abc import Sequence
from dataclasses import dataclass, field
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


def json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


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
        missing = len(tool_calls) - len(results)
        self.history.extend(cast(ResponseInputParam, output))
        self.history.extend(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            }
            for call, result in zip(
                tool_calls,
                [*results, *[interrupted] * missing],
                strict=True,
            )
        )

    def restore(self, dropped: ResponseInputParam) -> None:
        """Put turns that trim() dropped back at the start of the history."""
        self.history[:0] = dropped

    def add_summary(self, summary: str, max_bytes: int) -> None:
        """Put a summary of dropped turns at the start of the history.

        It is a user message, so a later trim() drops it like any turn and the
        next summary covers it. A summary that is empty or would push the
        history past max_bytes is left out.
        """
        summary = summary.strip()
        message: EasyInputMessageParam = {
            "role": "user",
            "content": "[Summary of the earlier conversation, which was removed "
            f"to save space]\n{summary}",
        }
        if summary and json_bytes([message, *self.history]) <= max_bytes:
            self.history.insert(0, message)

    def discard_unanswered(self, prompt: EasyInputMessageParam) -> None:
        """Drop the turn's prompt if no step was recorded after it."""
        if self.history and self.history[-1] is prompt:
            self.history.pop()

    def trim(self, max_bytes: int) -> tuple[bool, bool]:
        """Trim the history in place to fit max_bytes.

        Returns whether anything was trimmed and whether the current request
        itself had to be dropped.
        """
        history = self.history
        if json_bytes(history) <= max_bytes:
            return False, False

        trimmed = False
        target_bytes = int(max_bytes * TRIM_TARGET)
        omitted = (
            "[Tool output omitted to fit the context budget; "
            "call the tool again if it is still needed.]"
        )
        while json_bytes(history) > target_bytes:
            user_turn_starts = [
                index
                for index, item in enumerate(history)
                if isinstance(item, dict) and item.get("role") == "user"
            ]
            if len(user_turn_starts) > 1:
                del history[: user_turn_starts[1]]
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
                    and item.get("output") != omitted
                ),
                None,
            )
            if oldest_output is None:
                break  # nothing left that can be dropped
            cast(dict[str, Any], oldest_output)["output"] = omitted
            trimmed = True

        # The request itself is over the limit: give up on it.
        if json_bytes(history) > max_bytes and any(
            isinstance(item, dict) and item.get("role") == "user" for item in history
        ):
            replacement: ResponseInputParam = [
                {
                    "role": "user",
                    "content": "[Previous request omitted because its history exceeded the context budget.]",
                }
            ]
            history[:] = replacement if json_bytes(replacement) <= max_bytes else []
            return True, True
        return trimmed, False
