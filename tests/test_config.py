from app.config import load_settings


def test_quoted_values_from_docker_env_files_are_unquoted(monkeypatch):
    # `docker run --env-file` passes KEY='value' with the quotes.
    monkeypatch.setenv("API_KEY", "'sk-test'")
    monkeypatch.setenv("LLM_TEMPERATURE", '"0.5"')
    monkeypatch.setenv("MODEL_NAME", "anthropic/claude-haiku-5.5")

    settings = load_settings()

    assert settings.api_key == "sk-test"
    assert settings.temperature == 0.5
    assert settings.model_name == "anthropic/claude-haiku-5.5"
