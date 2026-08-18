"""Configuration validation."""

import pytest

from app.config import ConfigurationError, Settings


def test_defaults_are_applied(base_env: dict[str, str]) -> None:
    env = {
        key: value
        for key, value in base_env.items()
        if key
        in {
            "ATHENA_DATABASE_URL",
            "ATHENA_API_TOKEN",
            "ATHENA_EMBEDDING_BASE_URL",
            "ATHENA_EMBEDDING_API_KEY",
            "ATHENA_EMBEDDING_MODEL",
            "ATHENA_EMBEDDING_DIMENSION",
        }
    }
    settings = Settings.from_env(env)

    assert settings.db_schema == "athena"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.retrieval_mode == "hybrid"
    assert settings.vector_table == "athena_vectors"


@pytest.mark.parametrize(
    "name",
    [
        "ATHENA_DATABASE_URL",
        "ATHENA_API_TOKEN",
        "ATHENA_EMBEDDING_BASE_URL",
        "ATHENA_EMBEDDING_API_KEY",
        "ATHENA_EMBEDDING_MODEL",
        "ATHENA_EMBEDDING_DIMENSION",
    ],
)
def test_required_variables_are_reported_by_name(base_env: dict[str, str], name: str) -> None:
    base_env.pop(name)
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env(base_env)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"ATHENA_DATABASE_URL": "mysql://db/athena"}, "ATHENA_DATABASE_URL"),
        ({"ATHENA_DB_SCHEMA": "public; drop table"}, "ATHENA_DB_SCHEMA"),
        ({"ATHENA_API_TOKEN": "short"}, "ATHENA_API_TOKEN"),
        ({"ATHENA_EMBEDDING_BASE_URL": "not-a-url"}, "ATHENA_EMBEDDING_BASE_URL"),
        ({"ATHENA_EMBEDDING_DIMENSION": "0"}, "ATHENA_EMBEDDING_DIMENSION"),
        ({"ATHENA_EMBEDDING_DIMENSION": "abc"}, "ATHENA_EMBEDDING_DIMENSION"),
        ({"ATHENA_CHUNK_OVERLAP": "512", "ATHENA_CHUNK_SIZE": "256"}, "OVERLAP"),
        ({"ATHENA_DEFAULT_TOP_K": "30", "ATHENA_MAX_TOP_K": "10"}, "DEFAULT_TOP_K"),
        ({"ATHENA_RETRIEVAL_MODE": "lexical"}, "ATHENA_RETRIEVAL_MODE"),
    ],
)
def test_invalid_values_are_rejected(
    base_env: dict[str, str], overrides: dict[str, str], expected: str
) -> None:
    base_env.update(overrides)
    with pytest.raises(ConfigurationError, match=expected):
        Settings.from_env(base_env)


def test_secrets_are_never_echoed(base_env: dict[str, str]) -> None:
    base_env["ATHENA_API_TOKEN"] = "tiny"
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env(base_env)
    assert "tiny" not in str(error.value)

    settings = Settings.from_env(dict(base_env, ATHENA_API_TOKEN="a-valid-token-value-1234"))
    redacted = settings.redacted()
    assert "a-valid-token-value-1234" not in str(redacted)
    assert "embedding-key" not in str(redacted)


def test_database_urls_are_translated_for_sqlalchemy(base_env: dict[str, str]) -> None:
    base_env["ATHENA_DATABASE_URL"] = "postgres://user:pass@db:5432/athena"
    settings = Settings.from_env(base_env)

    assert settings.async_database_url == "postgresql+asyncpg://user:pass@db:5432/athena"
    assert settings.sync_database_url == "postgresql://user:pass@db:5432/athena"


def test_trailing_slash_is_stripped_from_embedding_url(base_env: dict[str, str]) -> None:
    base_env["ATHENA_EMBEDDING_BASE_URL"] = "https://resource.example.com/openai/v1/"
    assert (
        Settings.from_env(base_env).embedding_base_url == "https://resource.example.com/openai/v1"
    )


def test_upload_and_conversion_defaults(base_env: dict[str, str]) -> None:
    settings = Settings.from_env(base_env)

    assert settings.max_upload_bytes == 50 * 1024 * 1024
    assert settings.max_filename_characters == 255
    assert settings.document_converter == "docling"
    assert settings.docling_base_url == ""
    assert settings.docling_timeout_seconds == 120
    # An hour, so a slow scanned PDF is not cut off by Athena while Docling is
    # still working on it.
    assert settings.docling_conversion_deadline_seconds == 3_600
    assert settings.docling_poll_interval_seconds == 5
    assert settings.azure_ocr_endpoint == ""
    assert settings.azure_ocr_model_id == "prebuilt-layout"
    assert settings.azure_ocr_timeout_seconds == 300
    assert settings.conversion_configured is False


def test_a_configured_docling_url_enables_conversion(base_env: dict[str, str]) -> None:
    settings = Settings.from_env(
        {**base_env, "DOCLING_BASE_URL": "http://docling:5001/", "DOCLING_TIMEOUT_SECONDS": "45"}
    )

    assert settings.docling_base_url == "http://docling:5001"
    assert settings.docling_timeout_seconds == 45
    assert settings.conversion_configured is True


def test_async_conversion_bounds_are_configurable(base_env: dict[str, str]) -> None:
    settings = Settings.from_env(
        {
            **base_env,
            "DOCLING_CONVERSION_DEADLINE_SECONDS": "7200",
            "DOCLING_POLL_INTERVAL_SECONDS": "15",
        }
    )

    assert settings.docling_conversion_deadline_seconds == 7_200
    assert settings.docling_poll_interval_seconds == 15


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DOCLING_CONVERSION_DEADLINE_SECONDS", "30"),
        ("DOCLING_CONVERSION_DEADLINE_SECONDS", "90000"),
        ("DOCLING_CONVERSION_DEADLINE_SECONDS", "an hour"),
        ("DOCLING_POLL_INTERVAL_SECONDS", "0"),
        ("DOCLING_POLL_INTERVAL_SECONDS", "600"),
    ],
)
def test_out_of_range_async_conversion_bounds_are_rejected(
    base_env: dict[str, str], name: str, value: str
) -> None:
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({**base_env, name: value})


def test_the_deadline_must_cover_a_single_request(base_env: dict[str, str]) -> None:
    """A per-request timeout longer than the whole deadline can never be reached."""
    with pytest.raises(ConfigurationError, match="DOCLING_CONVERSION_DEADLINE_SECONDS"):
        Settings.from_env(
            {
                **base_env,
                "DOCLING_TIMEOUT_SECONDS": "600",
                "DOCLING_CONVERSION_DEADLINE_SECONDS": "120",
            }
        )


def test_the_deadline_is_reported_without_secrets(base_env: dict[str, str]) -> None:
    redacted = Settings.from_env(base_env).redacted()

    assert redacted["docling_conversion_deadline_seconds"] == 3_600
    assert redacted["docling_poll_interval_seconds"] == 5


@pytest.mark.parametrize("value", ["docling:5001", "ftp://docling", "/relative"])
def test_a_docling_url_must_be_absolute_http(base_env: dict[str, str], value: str) -> None:
    with pytest.raises(ConfigurationError, match="DOCLING_BASE_URL"):
        Settings.from_env({**base_env, "DOCLING_BASE_URL": value})


def test_an_unknown_document_converter_is_rejected(base_env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError, match="DOCUMENT_CONVERTER"):
        Settings.from_env({**base_env, "DOCUMENT_CONVERTER": "textract"})


def test_azure_document_converter_requires_endpoint_and_key(base_env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError, match="AZURE_OCR_ENDPOINT"):
        Settings.from_env({**base_env, "DOCUMENT_CONVERTER": "azure"})

    with pytest.raises(ConfigurationError, match="AZURE_OCR_ENDPOINT"):
        Settings.from_env(
            {
                **base_env,
                "DOCUMENT_CONVERTER": "azure",
                "AZURE_OCR_API_KEY": "a-key",
            }
        )

    with pytest.raises(ConfigurationError, match="AZURE_OCR_ENDPOINT"):
        Settings.from_env(
            {
                **base_env,
                "DOCUMENT_CONVERTER": "azure",
                "AZURE_OCR_ENDPOINT": "https://resource.cognitiveservices.azure.com",
            }
        )


def test_a_configured_azure_converter_enables_conversion(base_env: dict[str, str]) -> None:
    settings = Settings.from_env(
        {
            **base_env,
            "DOCUMENT_CONVERTER": "azure",
            "AZURE_OCR_ENDPOINT": "https://resource.cognitiveservices.azure.com/",
            "AZURE_OCR_API_KEY": "a-key",
            "AZURE_OCR_MODEL_ID": "prebuilt-read",
            "AZURE_OCR_TIMEOUT_SECONDS": "45",
        }
    )

    assert settings.document_converter == "azure"
    assert settings.azure_ocr_endpoint == "https://resource.cognitiveservices.azure.com"
    assert settings.azure_ocr_api_key == "a-key"
    assert settings.azure_ocr_model_id == "prebuilt-read"
    assert settings.azure_ocr_timeout_seconds == 45
    assert settings.conversion_configured is True
    # Docling stays optional and unrelated when Azure is the selected backend.
    assert settings.docling_base_url == ""


@pytest.mark.parametrize("value", ["resource.cognitiveservices.azure.com", "ftp://resource", "/relative"])
def test_an_azure_endpoint_must_be_absolute_http(base_env: dict[str, str], value: str) -> None:
    with pytest.raises(ConfigurationError, match="AZURE_OCR_ENDPOINT"):
        Settings.from_env({**base_env, "AZURE_OCR_ENDPOINT": value})


def test_upload_limits_are_bounded(base_env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError, match="ATHENA_MAX_UPLOAD_BYTES"):
        Settings.from_env({**base_env, "ATHENA_MAX_UPLOAD_BYTES": "1"})
    with pytest.raises(ConfigurationError, match="ATHENA_MAX_UPLOAD_BYTES"):
        Settings.from_env({**base_env, "ATHENA_MAX_UPLOAD_BYTES": "not-a-number"})
    with pytest.raises(ConfigurationError, match="ATHENA_MAX_UPLOAD_BYTES"):
        Settings.from_env(
            {
                **base_env,
                "ATHENA_MAX_UPLOAD_BYTES": str(50 * 1024 * 1024 + 1),
            }
        )


def test_redacted_settings_never_include_secrets(base_env: dict[str, str]) -> None:
    settings = Settings.from_env(
        {
            **base_env,
            "DOCLING_BASE_URL": "http://docling:5001",
            "AZURE_OCR_ENDPOINT": "https://resource.cognitiveservices.azure.com",
            "AZURE_OCR_API_KEY": "a-secret-azure-key",
        }
    )

    redacted = settings.redacted()

    assert redacted["docling_base_url"] == "http://docling:5001"
    assert redacted["document_converter"] == "docling"
    assert redacted["azure_ocr_endpoint"] == "https://resource.cognitiveservices.azure.com"
    assert "azure_ocr_api_key" not in redacted
    assert redacted["max_upload_bytes"] == 50 * 1024 * 1024
    serialized = str(redacted)
    assert settings.api_token not in serialized
    assert settings.embedding_api_key not in serialized
    assert settings.azure_ocr_api_key not in serialized
