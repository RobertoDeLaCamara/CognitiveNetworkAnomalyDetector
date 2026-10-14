# 0001. Default local MLflow tracking backend to SQLite

- Status: **Proposed**
- Date: 2026-10-14
- Issue: cognitive-anomaly-detector#1

## Context

`requirements.txt:22` allows `mlflow>=3.5.0,<4.0.0`, so CI resolves to the latest 3.x (3.17.1). That release refuses the filesystem tracking backend (`file://.../mlruns`). According to the issue, setting `MLFLOW_ALLOW_FILE_STORE=true` does not help.

The file store is the default today in two places:

- **Application default**: `src/config/mlflow_config.py:27-33` sets `MLFLOW_TRACKING_URI = f'file://{Path.cwd() / MLFLOW_DIR / "mlruns"}'` when no `MLFLOW_TRACKING_URI` is set at import time, and creates `.mlflow/mlruns` and `.mlflow/artifacts`.
- **Test fixture**: `tests/ml/test_mlflow_integration.py:28-45` (`temp_mlflow_dir`) sets `MLFLOW_TRACKING_URI=file://<tmp>/mlruns`. All 8 tests that use this fixture fail.

Other relevant code:

- `get_tracking_uri()` (`src/config/mlflow_config.py:114-122`) reads the env var at call time and falls back to the module constant. `os.getenv(name, default)` returns `''` when the variable is set but empty. `.env.example:7` ships `MLFLOW_TRACKING_URI=` (empty), and `docs/SETUP.md:129` documents `MLFLOW_TRACKING_URI=""` as the way to force local tracking.
- All callers go through `get_tracking_uri()`: `src/ml/model_trainer.py:80`, `src/ml/isolation_forest_detector.py:513,646`, `src/ml/lstm_autoencoder_detector.py:377`, `scripts/setup_mlflow.py:50`.
- `tests/ml/test_mlflow_integration.py:78-84` asserts that `.mlflow/mlruns` exists.
- `scripts/setup_mlflow.py:43,46,89` and `docs/SETUP.md:132` refer to the `mlruns` file store.
- `.gitignore:143` already ignores `.mlflow/`, so a `.mlflow/mlflow.db` file will not be committed.

## Decision drivers

- AC-1: `tests/ml/test_mlflow_integration.py` passes with mlflow 3.17.x.
- AC-2: the default local tracking URI is a SQLite database.
- AC-3: an explicit `MLFLOW_TRACKING_URI` still takes precedence.
- AC-4: no regressions in the rest of the suite.
- AC-5: the docs explain the default and how to migrate an existing `mlruns/` directory.
- Make the smallest change possible and add no new runtime dependencies.

## Considered options

1. **SQLite default in `mlflow_config.py`, SQLite URI in the test fixture (chosen).** The local default becomes `sqlite:///<cwd>/.mlflow/mlflow.db`, and artifacts stay in `.mlflow/artifacts`.
   - Good: this is the backend MLflow recommends. The change is confined to one config module, one fixture and the docs. SQLAlchemy and Alembic already come with the `mlflow` package.
   - Bad: existing local `mlruns/` history is not visible until the user migrates it.
2. **Opt out with `MLFLOW_ALLOW_FILE_STORE=true`.**
   - Good: no change to the storage format.
   - Bad: the issue reports that it does not fix the failures. It also only postpones the problem, because the file store is in maintenance mode.
3. **Pin `mlflow<3.x`, below the version that rejects the file store.**
   - Good: no code change.
   - Bad: blocks security and feature updates, conflicts with the existing `>=3.5.0` floor, and only postpones the problem.
4. **Require a remote tracking server (no local default).**
   - Bad: breaks offline and local training and the tests. It is far more than the issue asks for.

## Decision

Option 1:

- In `src/config/mlflow_config.py`, add `DEFAULT_LOCAL_TRACKING_URI = f'sqlite:///{(Path.cwd() / MLFLOW_DIR / "mlflow.db").as_posix()}'`. An absolute path gives the correct four-slash SQLite form.
- Use that constant for the local branch of `MLFLOW_TRACKING_URI`. Stop creating `.mlflow/mlruns` and keep creating `.mlflow/` and `.mlflow/artifacts`.
- Change `get_tracking_uri()` to `os.getenv('MLFLOW_TRACKING_URI') or MLFLOW_TRACKING_URI`. A non-empty env var still wins (AC-3). An empty value now falls back to the local SQLite default, which matches `.env.example` and `docs/SETUP.md:129`.
- In `temp_mlflow_dir`, set `MLFLOW_TRACKING_URI=sqlite:///<tmp>/mlflow.db`. Update `test_mlflow_directories_created` to assert `.mlflow/` and `.mlflow/artifacts` instead of `.mlflow/mlruns`.
- Update `scripts/setup_mlflow.py` so it no longer creates or advertises `mlruns`. The `mlflow ui` hint becomes `--backend-store-uri <DEFAULT_LOCAL_TRACKING_URI>`.
- Document the default and the migration (`mlflow migrate-filestore`) in the README Configuration section and in `docs/SETUP.md`.

## Consequences

**Good**
- CI and local tests work with current MLflow 3.x.
- Production local tracking stops relying on a backend in maintenance mode.
- An empty `MLFLOW_TRACKING_URI` now behaves the way the docs already describe.

**Bad / risks**
- Existing local experiment history in `.mlflow/mlruns` is not read automatically. Users have to run `mlflow migrate-filestore`, which the docs will describe. Remote server users, such as `.env` pointing at the MLflow server, are unaffected.
- On first use, MLflow creates the schema in `mlflow.db` through its own Alembic migrations. This is managed by MLflow, not by this repo.
- SQLite does not handle concurrent writers well. That is acceptable for single-user local tracking. Concurrent setups should use the remote server.

**Migration**: none in this repo's schema. Migrating old local history is optional and done by the user (`mlflow migrate-filestore`).

**Rollback**: revert the commit. The old `.mlflow/mlruns` directory is never deleted, so it is still there. Rolling back would only make sense together with pinning an older mlflow.

## Verification

| AC | How it is verified |
|----|--------------------|
| AC-1 | `pytest tests/ml/test_mlflow_integration.py` with mlflow 3.17.x: 0 failures, with the fixture using a SQLite URI |
| AC-2 | New unit test in `tests/ml/test_mlflow_config.py`: with `MLFLOW_TRACKING_URI` unset, and with it set to an empty string, `get_tracking_uri()` starts with `sqlite:///` and ends with `.mlflow/mlflow.db`. A second test asserts that `DEFAULT_LOCAL_TRACKING_URI` does not start with `file:` |
| AC-3 | The existing `test_returns_env_variable_when_set` stays. A new test sets `MLFLOW_TRACKING_URI=sqlite:///custom.db` and asserts it is returned unchanged |
| AC-4 | Full suite: `pytest tests/` |
| AC-5 | Doc review: README Configuration row and section, and the `docs/SETUP.md` "Use remote vs local" section mention `sqlite:///.mlflow/mlflow.db` and `mlflow migrate-filestore` |
