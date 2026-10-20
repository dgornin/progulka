# AGENTS.md

Progulka plans a walking route for each day of a 1–7 day trip in an unfamiliar city, from
real places (OpenStreetMap, Wikipedia, Wikivoyage). Plan-and-Execute: an outline LLM step
splits the trip into days, per-day ReAct agents research in parallel, deterministic code
picks, orders and validates the stops. FastAPI + SQLite, vanilla JS UI; README.md has the why.

## Where things are
- `app/main.py`, `auth.py`, `limits.py`: HTTP pages and API, accounts and sessions, public-deploy limits.
- `app/db.py`, `store.py`: SQLite via SQLAlchemy; users, sessions, saved trips, memory.
- `app/pipeline.py`, `personalization.py`, `profile.py`, `models.py`: the fixed pipeline,
  brief merging (code, not LLM), questionnaire, domain models.
- `app/agent/`: every step where the LLM works (brief, outline, researcher + tools, writer,
  replanner, memory, speech); all system prompts are in `prompts.py`.
- `app/planning/`: deterministic optimizer, validator, opening hours, walking geometry and routing.
- `app/sources/`: open-data clients; all HTTP goes through `sources/http.py` (User-Agent, cache, retries).
  `sources/wikidata.py` scores how well known a found place is (Wikipedia views, Wikidata sitelinks).
- `app/llm.py`, `config.py`: the only LLM wrapper (no framework) and settings from the environment.
- `app/static/`: UI without a build step. `tests/`: offline tests. `eval/`: agent vs one-prompt baseline.
- `Dockerfile`, `docker-compose*.yml`, `deploy/Caddyfile`, `.github/workflows/ci.yml`: image, deploy, CI.

## Commands
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q                                      # offline: no key, no network
.venv/bin/ruff check app tests eval && .venv/bin/ruff format app tests eval
.venv/bin/uvicorn app.main:app --reload                  # http://localhost:8000
docker compose up --build                                # the same app in the production image
.venv/bin/python -m eval.run_eval [--model M] [--only id,id]   # calls the provider: costs money
```

## Conventions
- Docstrings and comments explain why: the design decision or the failure behind it, not what the code does.
- No dead code, no commented-out code, no abandoned drafts: every file must justify itself.
- UI text and API error messages are Russian. Code, comments, docstrings and system prompts are
  English (models follow English instructions best); the texts the traveller reads are generated
  in the brief's language.
- Python 3.12, line length 100 (`ruff.toml`). Dependencies are pinned with `==`; a bump is a deliberate
  edit of `requirements*.txt` followed by `pip install -r requirements-dev.txt` and the full suite.
- Tests are offline. HTTP sources replay recorded answers from `tests/fixtures/` through a fake of
  the shared client's `get_json`/`post_form_json` (the client itself runs on an httpx `MockTransport`);
  the LLM is `FakeLLM` in `tests/test_agent.py`, which returns scripted replies. Never call the provider
  or the network from a test, and never record a fixture that contains a key or a visitor's data.
- Configuration comes only from the environment (`app/config.py`); `.env` is gitignored and never
  committed. A new variable goes into `config.py`, `.env.example` and the README table together.
- `LLM_REPLAY` (`app/llm.py`): `1` stores every chat response in `$CACHE_DIR/llm/<sha256 of the request>.json`
  and answers an identical request (model, messages, temperature, max_tokens, tools, response_format;
  not the timeout) from disk; a miss logs "LLM replay miss", calls the provider and records the answer.
  `0` (default) reads and writes nothing, so a public server keeps no visitor's words. Voice
  transcription is never cached. Any change to a prompt, tool schema, source answer or date changes
  the key, so pin `TODAY=YYYY-MM-DD` to replay a rehearsed demo. Replay is demo insurance, not a test tool.

## Definition of done
- `ruff check app tests eval`, `ruff format --check app tests eval` and `pytest -q` are green
  (CI runs the same three).
- New behaviour has an offline test; docstrings and comments describe the new behaviour.
- README.md is updated when behaviour, the API, environment variables or commands change.
- UI and LLM-facing changes are also run and looked at: green tests are not the same as correct.

## Do not
- Read, print, edit or commit `.env`, or put any key into code, fixtures, logs or cache files you add.
- Change the demo/admin seeding (`lifespan` in `app/main.py`, `auth.seed_account`, `DEMO_*`/`ADMIN_*`)
  unless the task is about the accounts.
- Add a framework or dependency (agent framework, ORM layer, JS build step) without a written reason.
- Make live LLM calls or run `eval/` unless the human asks: both spend the API key.
- Commit or push unless the human asks: the human reviews and commits.

This file is advice; the deterministic gate is CI (`.github/workflows/ci.yml`).
