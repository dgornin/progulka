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


def test_database_lives_in_data_dir_unless_a_url_is_given(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert load_settings().database_url == f"sqlite+aiosqlite:///{tmp_path / 'progulka.db'}"

    monkeypatch.setenv("DATABASE_URL", "'sqlite+aiosqlite:////srv/other.db'")
    assert load_settings().database_url == "sqlite+aiosqlite:////srv/other.db"


def test_accounts_and_cookie_flags(monkeypatch):
    monkeypatch.setenv("COOKIE_SECURE", "1")
    monkeypatch.setenv("DEMO_EMAIL", '"demo@progulka.app"')
    monkeypatch.setenv("DEMO_PASSWORD", "progulka-demo")
    monkeypatch.delenv("ADMIN_EMAIL", raising=False)

    settings = load_settings()

    assert settings.cookie_secure
    assert (settings.demo_email, settings.demo_password) == ("demo@progulka.app", "progulka-demo")
    assert settings.admin_email == ""
