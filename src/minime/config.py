from pathlib import Path

from openai.types.shared import ReasoningEffort
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for the AI coding agent."""

    model_config = SettingsConfigDict(
        # .env stays in the project root, above src/minime/. That holds for
        # the editable install `uv sync` makes; a copied install finds none.
        env_file=Path(__file__).parents[2] / ".env",
        extra="ignore",
        # Command-line flags, when main() asks for them: --max-steps-per-turn 8,
        # --no-streaming. They override the environment and .env.
        cli_prog_name="minime",
        cli_kebab_case=True,
        cli_implicit_flags=True,
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
    max_steps: int = Field(
        default=16,
        gt=0,
        validation_alias="MAX_STEPS_PER_TURN",
    )
    command_timeout: int = Field(
        default=30,
        gt=0,
        validation_alias="COMMAND_TIMEOUT_SECONDS",
    )
    # These fields read the environment variable of the same name (matched
    # case-insensitively), so they need no alias.
    max_context_tokens: int = Field(default=64 * 1024, gt=0)
    reasoning_effort: ReasoningEffort = None
    summarize: bool = True
    streaming: bool = True
    brave_api_key: str | None = None
