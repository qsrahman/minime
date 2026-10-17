from pathlib import Path

from openai.types.shared import ReasoningEffort
from pydantic import DirectoryPath, Field
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
    max_history_tokens: int = Field(
        default=200_000,
        gt=0,
        validation_alias="MAX_HISTORY_TOKENS",
    )
    reasoning_effort: ReasoningEffort = Field(
        default=None,
        validation_alias="REASONING_EFFORT",
    )
    summarize: bool = Field(
        default=True,
        validation_alias="SUMMARIZE",
    )
    streaming: bool = Field(
        default=True,
        validation_alias="STREAMING",
    )
    brave_api_key: str | None = Field(
        default=None,
        validation_alias="BRAVE_API_KEY",
    )
    workspace_root: DirectoryPath = Field(
        default_factory=Path.cwd,
        validation_alias="WORKSPACE_ROOT",
    )
