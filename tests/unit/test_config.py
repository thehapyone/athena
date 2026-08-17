"""Configuration validation."""

import pytest

from app.config import ConfigurationError, Settings


def test_defaults_are_applied(base_env: dict[str, str]) -> None:
    env = {
        key: value
        for key, value in base_env.items()
        if key
        in {
            "KNOWLEDGE_DATABASE_URL",
            "KNOWLEDGE_API_TOKEN",
            "KNOWLEDGE_EMBEDDING_BASE_URL",
            "KNOWLEDGE_EMBEDDING_API_KEY",
            "KNOWLEDGE_EMBEDDING_MODEL",
            "KNOWLEDGE_EMBEDDING_DIMENSION",
        }
    }
    settings = Settings.from_env(env)

    assert settings.db_schema == "knowledge"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.retrieval_mode == "hybrid"
    assert settings.vector_table == "knowledge_vectors"


@pytest.mark.parametrize(
    "name",
    [
        "KNOWLEDGE_DATABASE_URL",
        "KNOWLEDGE_API_TOKEN",
        "KNOWLEDGE_EMBEDDING_BASE_URL",
        "KNOWLEDGE_EMBEDDING_API_KEY",
        "KNOWLEDGE_EMBEDDING_MODEL",
        "KNOWLEDGE_EMBEDDING_DIMENSION",
    ],
)
def test_required_variables_are_reported_by_name(base_env: dict[str, str], name: str) -> None:
    base_env.pop(name)
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env(base_env)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"KNOWLEDGE_DATABASE_URL": "mysql://db/knowledge"}, "KNOWLEDGE_DATABASE_URL"),
        ({"KNOWLEDGE_DB_SCHEMA": "public; drop table"}, "KNOWLEDGE_DB_SCHEMA"),
        ({"KNOWLEDGE_API_TOKEN": "short"}, "KNOWLEDGE_API_TOKEN"),
        ({"KNOWLEDGE_EMBEDDING_BASE_URL": "not-a-url"}, "KNOWLEDGE_EMBEDDING_BASE_URL"),
        ({"KNOWLEDGE_EMBEDDING_DIMENSION": "0"}, "KNOWLEDGE_EMBEDDING_DIMENSION"),
        ({"KNOWLEDGE_EMBEDDING_DIMENSION": "abc"}, "KNOWLEDGE_EMBEDDING_DIMENSION"),
        ({"KNOWLEDGE_CHUNK_OVERLAP": "512", "KNOWLEDGE_CHUNK_SIZE": "256"}, "OVERLAP"),
        ({"KNOWLEDGE_DEFAULT_TOP_K": "30", "KNOWLEDGE_MAX_TOP_K": "10"}, "DEFAULT_TOP_K"),
        ({"KNOWLEDGE_RETRIEVAL_MODE": "lexical"}, "KNOWLEDGE_RETRIEVAL_MODE"),
    ],
)
def test_invalid_values_are_rejected(
    base_env: dict[str, str], overrides: dict[str, str], expected: str
) -> None:
    base_env.update(overrides)
    with pytest.raises(ConfigurationError, match=expected):
        Settings.from_env(base_env)


def test_secrets_are_never_echoed(base_env: dict[str, str]) -> None:
    base_env["KNOWLEDGE_API_TOKEN"] = "tiny"
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env(base_env)
    assert "tiny" not in str(error.value)

    settings = Settings.from_env(dict(base_env, KNOWLEDGE_API_TOKEN="a-valid-token-value-1234"))
    redacted = settings.redacted()
    assert "a-valid-token-value-1234" not in str(redacted)
    assert "embedding-key" not in str(redacted)


def test_database_urls_are_translated_for_sqlalchemy(base_env: dict[str, str]) -> None:
    base_env["KNOWLEDGE_DATABASE_URL"] = "postgres://user:pass@db:5432/knowledge"
    settings = Settings.from_env(base_env)

    assert settings.async_database_url == "postgresql+asyncpg://user:pass@db:5432/knowledge"
    assert settings.sync_database_url == "postgresql://user:pass@db:5432/knowledge"


def test_trailing_slash_is_stripped_from_embedding_url(base_env: dict[str, str]) -> None:
    base_env["KNOWLEDGE_EMBEDDING_BASE_URL"] = "https://resource.example.com/openai/v1/"
    assert (
        Settings.from_env(base_env).embedding_base_url == "https://resource.example.com/openai/v1"
    )


def test_upload_and_conversion_defaults(base_env: dict[str, str]) -> None:
    settings = Settings.from_env(base_env)

    assert settings.max_upload_bytes == 20 * 1024 * 1024
    assert settings.max_filename_characters == 255
    assert settings.docling_base_url == ""
    assert settings.docling_timeout_seconds == 660
    assert settings.conversion_configured is False


def test_a_configured_docling_url_enables_conversion(base_env: dict[str, str]) -> None:
    settings = Settings.from_env(
        {**base_env, "DOCLING_BASE_URL": "http://docling:5001/", "DOCLING_TIMEOUT_SECONDS": "45"}
    )

    assert settings.docling_base_url == "http://docling:5001"
    assert settings.docling_timeout_seconds == 45
    assert settings.conversion_configured is True


@pytest.mark.parametrize("value", ["docling:5001", "ftp://docling", "/relative"])
def test_a_docling_url_must_be_absolute_http(base_env: dict[str, str], value: str) -> None:
    with pytest.raises(ConfigurationError, match="DOCLING_BASE_URL"):
        Settings.from_env({**base_env, "DOCLING_BASE_URL": value})


def test_upload_limits_are_bounded(base_env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError, match="KNOWLEDGE_MAX_UPLOAD_BYTES"):
        Settings.from_env({**base_env, "KNOWLEDGE_MAX_UPLOAD_BYTES": "1"})
    with pytest.raises(ConfigurationError, match="KNOWLEDGE_MAX_UPLOAD_BYTES"):
        Settings.from_env({**base_env, "KNOWLEDGE_MAX_UPLOAD_BYTES": "not-a-number"})
    with pytest.raises(ConfigurationError, match="KNOWLEDGE_MAX_UPLOAD_BYTES"):
        Settings.from_env(
            {
                **base_env,
                "KNOWLEDGE_MAX_UPLOAD_BYTES": str(20 * 1024 * 1024 + 1),
            }
        )


def test_redacted_settings_never_include_secrets(base_env: dict[str, str]) -> None:
    settings = Settings.from_env({**base_env, "DOCLING_BASE_URL": "http://docling:5001"})

    redacted = settings.redacted()

    assert redacted["docling_base_url"] == "http://docling:5001"
    assert redacted["max_upload_bytes"] == 20 * 1024 * 1024
    serialized = str(redacted)
    assert settings.api_token not in serialized
    assert settings.embedding_api_key not in serialized
