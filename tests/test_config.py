"""
Unit tests for RAGnarok configuration and settings subsystem.
Validates dynamic config loading, environment secret parsing, schema constraints,
and error handling for missing or invalid settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ragnarok.core.config import (
    AppConfig,
    DatabaseConfig,
    GenerationConfig,
    load_config,
)


def test_default_config_loading() -> None:
    """Verifies that config.toml loads cleanly with all expected sections."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.general.app_name == "RAGnarok"
    assert config.database.min_connections >= 1
    assert config.chunking.parent_target_size > config.chunking.child_target_size
    assert config.dense_retrieval.vector_dimension == 384
    assert config.hybrid_retrieval.k == 60
    assert config.hybrid_retrieval.reranker.enabled is True
    assert config.context_management.max_tokens == 4000
    assert config.generation.temperature <= 0.2  # Grounding invariant
    assert len(config.generation.fallbacks.models) > 0

def test_custom_toml_loading(tmp_path: Path) -> None:
    """Verifies loading from an explicit custom TOML file."""
    custom_toml = tmp_path / "custom_config.toml"
    custom_toml.write_text(
        """
[general]
app_name = "CustomRAG"
environment = "staging"

[chunking]
parent_target_size = 1500
child_target_size = 300

[dense_retrieval]
provider = "sentence-transformers"
model = "custom-embed-model"
vector_dimension = 768

[generation]
model = "custom-llm"
temperature = 0.05
max_tokens = 2000
""",
        encoding="utf-8",
    )

    config = load_config(config_path=custom_toml)
    assert config.general.app_name == "CustomRAG"
    assert config.general.environment == "staging"
    assert config.chunking.parent_target_size == 1500
    assert config.chunking.child_target_size == 300
    assert config.dense_retrieval.model == "custom-embed-model"
    assert config.dense_retrieval.vector_dimension == 768
    # Dense retrieval and embedding sync check
    assert config.embedding.model == "custom-embed-model"
    assert config.generation.temperature == 0.05

def test_nonexistent_config_raises_error(tmp_path: Path) -> None:
    """Verifies that an explicit nonexistent path raises FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.toml"
    with pytest.raises(FileNotFoundError):
        load_config(config_path=missing_file)

def test_invalid_settings_validation_errors() -> None:
    """Verifies that out-of-bound settings trigger Pydantic validation errors."""
    with pytest.raises(ValidationError):
        DatabaseConfig.model_validate({"min_connections": 0})  # ge=1 required

    with pytest.raises(ValidationError):
        GenerationConfig.model_validate({"temperature": 2.5})  # le=2.0 required

def test_env_secrets_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that secrets are loaded dynamically from an env file with zero hardcoded values."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    custom_env = tmp_path / ".custom_env"
    custom_env.write_text(
        """
DATABASE_URL="postgresql://custom_user:custom_pass@localhost:5433/custom_db"
OPENAI_API_KEY="sk-test-secret-key-12345"
GROQ_API_KEY="gsk-test-groq-key-67890"
""",
        encoding="utf-8",
    )

    config = load_config(env_file=custom_env)
    assert (
        config.secrets.database_url
        == "postgresql://custom_user:custom_pass@localhost:5433/custom_db"
    )
    assert config.secrets.openai_api_key is not None
    assert (
        config.secrets.openai_api_key.get_secret_value()
        == "sk-test-secret-key-12345"
    )
    assert config.secrets.groq_api_key is not None
    assert (
        config.secrets.groq_api_key.get_secret_value()
        == "gsk-test-groq-key-67890"
    )
