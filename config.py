import sys
from functools import cache
from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for the AI coding agent."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        extra="ignore",
    )
    api_key: str = Field(
        min_length=1,
        validation_alias="OPENAI_API_KEY",
    )
    model: str = Field(
        default="gpt-5.4-mini",
        min_length=1,
        validation_alias="OPENAI_MODEL",
    )
    base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    max_turns: int = Field(
        default=16,
        gt=0,
        validation_alias="MAX_TURNS_PER_RUN",
    )
    command_timeout: int = Field(
        default=30,
        gt=0,
        validation_alias="COMMAND_TIMEOUT_SECONDS",
    )
    max_history_turns: int = Field(
        default=12,
        gt=0,
        validation_alias="MAX_HISTORY_TURNS",
    )
    max_history_bytes: int = Field(
        default=256 * 1024,
        gt=0,
        validation_alias="MAX_HISTORY_BYTES",
    )
    streaming: bool = Field(
        default=True,
        validation_alias="STREAMING",
    )
    brave_api_key: str | None = Field(
        default=None,
        validation_alias="BRAVE_API_KEY",
    )
    workspace_root: Path = Field(
        default_factory=Path.cwd,
        validation_alias="WORKSPACE_ROOT",
    )


@cache
def get_config() -> Config:
    """Load and validate the agent configuration on first use."""
    try:
        return Config()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None
