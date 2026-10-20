import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from minime import main
from minime.config import Config


def make(**values: object) -> Config:
    return Config(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = make(OPENAI_API_KEY="k")
    assert settings.model == "gpt-5.4-mini"
    assert settings.base_url is None
    assert settings.max_steps == 16
    assert settings.command_timeout == 30
    assert settings.reasoning_effort is None
    assert settings.summarize is True
    assert settings.max_context_tokens == 64 * 1024
    assert settings.streaming is True
    assert settings.brave_api_key is None


def test_environment_variables_map_to_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = {
        "OPENAI_API_KEY": "env-key",
        "OPENAI_MODEL": "some-model",
        "OPENAI_BASE_URL": "http://localhost:1234/v1",
        "MAX_STEPS_PER_TURN": "3",
        "COMMAND_TIMEOUT_SECONDS": "7",
        "REASONING_EFFORT": "medium",
        "SUMMARIZE": "false",
        "MAX_CONTEXT_TOKENS": "1000",
        "STREAMING": "false",
        "BRAVE_API_KEY": "brave",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = make()
    assert settings.api_key == "env-key"
    assert settings.model == "some-model"
    assert settings.base_url == "http://localhost:1234/v1"
    assert settings.max_steps == 3
    assert settings.command_timeout == 7
    assert settings.reasoning_effort == "medium"
    assert settings.summarize is False
    assert settings.max_context_tokens == 1000
    assert settings.streaming is False
    assert settings.brave_api_key == "brave"


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
        ("MAX_STEPS_PER_TURN", "0"),
        ("MAX_STEPS_PER_TURN", "many"),
        ("COMMAND_TIMEOUT_SECONDS", "-1"),
        ("REASONING_EFFORT", "extreme"),
        ("REASONING_EFFORT", ""),
        ("SUMMARIZE", "sometimes"),
        ("MAX_CONTEXT_TOKENS", "0"),
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


def test_api_key_is_required() -> None:
    with pytest.raises(ValidationError):
        make()


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Config at a temporary .env file instead of the project's."""
    path = tmp_path / ".env"
    monkeypatch.setitem(Config.model_config, "env_file", path)
    return path


def test_config_reads_the_env_file(env_file: Path) -> None:
    env_file.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\n")
    settings = Config()
    assert settings.api_key == "file-key"
    assert settings.model == "file-model"


def test_unknown_keys_in_the_env_file_are_ignored(env_file: Path) -> None:
    env_file.write_text("OPENAI_API_KEY=file-key\nSOMETHING_ELSE=x\n")
    assert Config().api_key == "file-key"


def test_environment_overrides_the_env_file(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\n")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert Config().model == "env-model"


def test_env_file_values_are_not_exported(env_file: Path) -> None:
    env_file.write_text("OPENAI_API_KEY=file-key\nBRAVE_API_KEY=brave-file-key\n")
    assert Config().brave_api_key == "brave-file-key"
    assert "BRAVE_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_main_reports_configuration_errors_and_exits(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file.write_text("MAX_STEPS_PER_TURN=0\n")
    with pytest.raises(SystemExit) as exit_info:
        main.main()
    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Configuration error:" in captured.err
    assert "OPENAI_API_KEY" in captured.err
    assert "MAX_STEPS_PER_TURN" in captured.err


def test_command_line_flags_override_the_environment(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text("OPENAI_API_KEY=k\nOPENAI_MODEL=file-model\n")
    monkeypatch.setenv("MAX_STEPS_PER_TURN", "9")
    monkeypatch.setattr(
        "sys.argv",
        ["minime", "--openai-model", "flag-model", "--max-steps-per-turn", "3"]
        + ["--no-streaming", "--reasoning-effort", "low"],
    )
    started: list[Config] = []
    monkeypatch.setattr(main, "OpenAI", lambda **kwargs: "client")
    monkeypatch.setattr(main, "Agent", lambda config, client, toolbox: config)
    monkeypatch.setattr(
        main,
        "Repl",
        lambda config: SimpleNamespace(run=lambda: started.append(config)),
    )
    main.main()
    [config] = started
    assert config.api_key == "k"  # from .env, not overridden
    assert config.model == "flag-model"
    assert config.max_steps == 3
    assert config.streaming is False
    assert config.reasoning_effort == "low"


def test_invalid_flag_values_are_reported(
    env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file.write_text("OPENAI_API_KEY=k\n")
    monkeypatch.setattr("sys.argv", ["minime", "--max-steps-per-turn", "0"])
    with pytest.raises(SystemExit) as exit_info:
        main.main()
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "Configuration error:" in err
    assert "greater than 0" in err
