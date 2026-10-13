import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import config
from config import Config


def make(**values: object) -> Config:
    return Config(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = make(OPENAI_API_KEY="k")
    assert settings.model == "gpt-5.4-mini"
    assert settings.base_url is None
    assert settings.max_turns == 16
    assert settings.command_timeout == 30
    assert settings.reasoning_effort is None
    assert settings.summarize is True
    assert settings.max_history_tokens == 200_000
    assert settings.streaming is True
    assert settings.brave_api_key is None
    assert settings.workspace_root == Path.cwd()


def test_environment_variables_map_to_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = {
        "OPENAI_API_KEY": "env-key",
        "OPENAI_MODEL": "some-model",
        "OPENAI_BASE_URL": "http://localhost:1234/v1",
        "MAX_TURNS_PER_RUN": "3",
        "COMMAND_TIMEOUT_SECONDS": "7",
        "REASONING_EFFORT": "medium",
        "SUMMARIZE": "false",
        "MAX_HISTORY_TOKENS": "1000",
        "STREAMING": "false",
        "BRAVE_API_KEY": "brave",
        "WORKSPACE_ROOT": str(tmp_path),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = make()
    assert settings.api_key == "env-key"
    assert settings.model == "some-model"
    assert settings.base_url == "http://localhost:1234/v1"
    assert settings.max_turns == 3
    assert settings.command_timeout == 7
    assert settings.reasoning_effort == "medium"
    assert settings.summarize is False
    assert settings.max_history_tokens == 1000
    assert settings.streaming is False
    assert settings.brave_api_key == "brave"
    assert settings.workspace_root == tmp_path


def test_unknown_environment_variables_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOMETHING_ELSE", "x")
    assert make(OPENAI_API_KEY="k").api_key == "k"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENAI_API_KEY", ""),
        ("OPENAI_MODEL", ""),
        ("MAX_TURNS_PER_RUN", "0"),
        ("MAX_TURNS_PER_RUN", "many"),
        ("COMMAND_TIMEOUT_SECONDS", "-1"),
        ("REASONING_EFFORT", "extreme"),
        ("REASONING_EFFORT", ""),
        ("SUMMARIZE", "sometimes"),
        ("MAX_HISTORY_TOKENS", "0"),
        ("STREAMING", "maybe"),
    ],
)
def test_invalid_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        make()


def test_workspace_root_must_be_an_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x")
    for bad in (tmp_path / "missing", tmp_path / "file.txt"):
        with pytest.raises(ValidationError, match="WORKSPACE_ROOT"):
            make(OPENAI_API_KEY="k", WORKSPACE_ROOT=bad)


def test_api_key_is_required() -> None:
    with pytest.raises(ValidationError):
        make()


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the real get_config() at a temporary .env file."""
    path = tmp_path / ".env"
    monkeypatch.setitem(Config.model_config, "env_file", path)
    config.get_config.cache_clear()
    yield path
    config.get_config.cache_clear()


def test_get_config_reads_the_env_file(env_file: Path) -> None:
    env_file.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\n")
    settings = config.get_config()
    assert settings.api_key == "file-key"
    assert settings.model == "file-model"


def test_unknown_keys_in_the_env_file_are_ignored(env_file: Path) -> None:
    env_file.write_text("OPENAI_API_KEY=file-key\nSOMETHING_ELSE=x\n")
    assert config.get_config().api_key == "file-key"


def test_environment_overrides_the_env_file(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\n")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert config.get_config().model == "env-model"


def test_env_file_values_are_not_exported(env_file: Path) -> None:
    env_file.write_text("OPENAI_API_KEY=file-key\nBRAVE_API_KEY=brave-file-key\n")
    assert config.get_config().brave_api_key == "brave-file-key"
    assert "BRAVE_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_get_config_is_cached(env_file: Path) -> None:
    env_file.write_text("OPENAI_API_KEY=first\n")
    first = config.get_config()
    env_file.write_text("OPENAI_API_KEY=second\n")
    assert config.get_config() is first


def test_missing_workspace_is_reported_at_startup(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file.write_text("OPENAI_API_KEY=k\nWORKSPACE_ROOT=/definitely/not/here\n")
    with pytest.raises(SystemExit):
        config.get_config()
    assert (
        "WORKSPACE_ROOT: Path does not point to a directory" in capsys.readouterr().err
    )


def test_get_config_reports_errors_and_exits(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file.write_text("MAX_TURNS_PER_RUN=0\n")
    with pytest.raises(SystemExit) as exit_info:
        config.get_config()
    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Configuration error:" in captured.err
    assert "OPENAI_API_KEY" in captured.err
    assert "MAX_TURNS_PER_RUN" in captured.err
