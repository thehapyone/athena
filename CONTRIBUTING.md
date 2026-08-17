# Contributing to Athena

## Development setup

Athena requires Python 3.12 or newer and `uv`.

```bash
uv sync --extra dev
uv run pytest -q
```

The PostgreSQL integration tests run when `ATHENA_TEST_DATABASE_URL` points
to a PostgreSQL database with the `vector` extension. Unit tests use in-memory
fakes and do not require external services.

Before opening a pull request, run the test suite and build the image locally:

```bash
docker build -t athena:local .
```

Keep changes focused, update the README when the API or configuration changes,
and do not commit credentials, source documents, generated files, or local
environment files. Pull requests should explain the user-visible or operational
impact of the change and include relevant test coverage.
