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


def test_speech_model_has_a_default_and_can_be_changed(monkeypatch):
    monkeypatch.delenv("STT_MODEL", raising=False)
    assert load_settings().stt_model == "qwen/qwen3.8-omni-flash"

    monkeypatch.setenv("STT_MODEL", "google/gemini-3.5-flash")
    assert load_settings().stt_model == "google/gemini-3.5-flash"


def test_secrets_stay_out_of_the_settings_repr(monkeypatch):
    # A traceback or a log line that prints the settings must not print the key.
    monkeypatch.setenv("API_KEY", "sk-secret-value")
    monkeypatch.setenv("DEMO_PASSWORD", "demo-secret-value")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-secret-value")

    shown = repr(load_settings())

    assert "secret-value" not in shown
    assert "model_name=" in shown


def test_the_trip_planner_agent_is_on_unless_turned_off(monkeypatch):
    monkeypatch.delenv("TRIP_PLANNER_AGENT", raising=False)
    assert load_settings().trip_planner_agent

    monkeypatch.setenv("TRIP_PLANNER_AGENT", "0")  # the eval's ablation: code assembly only
    assert not load_settings().trip_planner_agent
