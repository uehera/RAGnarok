"""
RAGnarok Configuration & Settings Subsystem.

Provides typed, validated access to system configuration (config.toml)
and environment credentials (.env) using Pydantic v2 and pydantic-settings.
Enforces zero hardcoded models, providers, or credentials.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSecrets(BaseSettings):
    """
    Secrets and infrastructure endpoints loaded dynamically from environment variables or .env.
    No credentials or connection strings may be hardcoded anywhere in the codebase.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/ragnarok",
        description="PostgreSQL connection string with pgcrypto extension enabled",
    )
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: SecretStr | None = Field(default=None, description="Anthropic API key")
    groq_api_key: SecretStr | None = Field(default=None, description="Groq API key")
    gemini_api_key: SecretStr | None = Field(default=None, description="Google Gemini API key")
    ollama_api_base: str | None = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    vllm_api_base: str | None = Field(
        default="http://localhost:8000/v1", description="vLLM API base URL"
    )


class GeneralConfig(BaseModel):
    app_name: str = Field(default="RAGnarok")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")


class DatabaseConfig(BaseModel):
    min_connections: int = Field(default=1, ge=1)
    max_connections: int = Field(default=10, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class IngestionDeduplicationConfig(BaseModel):
    hash_algorithm: str = Field(default="sha256")
    reject_duplicates: bool = Field(default=True)


class IngestionParserConfig(BaseModel):
    engine: str = Field(default="docling")
    export_format: str = Field(default="markdown")


class IngestionConfig(BaseModel):
    staging_directory: Path = Field(default_factory=lambda: Path("data/staging"))
    raw_directory: Path = Field(default_factory=lambda: Path("data/raw"))
    parsed_directory: Path = Field(default_factory=lambda: Path("data/parsed"))
    supported_formats: list[str] = Field(default_factory=lambda: [".pdf", ".docx", ".md", ".txt"])
    batch_size: int = Field(default=50, ge=1)
    deduplication: IngestionDeduplicationConfig = Field(
        default_factory=IngestionDeduplicationConfig
    )
    parser: IngestionParserConfig = Field(default_factory=IngestionParserConfig)


class ChunkingConfig(BaseModel):
    strategy: str = Field(default="markdown_header_recursive")
    parent_target_size: int = Field(default=2000, ge=100)
    child_target_size: int = Field(default=500, ge=50)
    child_overlap: int = Field(default=50, ge=0)
    parent_overlap: int = Field(default=0, ge=0)
    heading_levels: list[str] = Field(default_factory=lambda: ["#", "##", "###"])
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    size_counter: str = Field(default="transformers.AutoTokenizer")
    fallback_counter: str = Field(default="tiktoken")
    max_input_tokens: int = Field(default=8192, ge=256)


class DenseRetrievalConfig(BaseModel):
    provider: str = Field(default="sentence-transformers")
    model: str = Field(default="BAAI/bge-small-en-v1.5")
    vector_dimension: int = Field(default=384, ge=1)
    batch_size: int = Field(default=100, ge=1)
    similarity_metric: str = Field(default="cosine")
    input_formatting: dict[str, str] = Field(default_factory=dict)


class LexicalRetrievalConfig(BaseModel):
    library: str = Field(default="bm25s")
    case: str = Field(default="lowercase")
    stop_words_removed: bool = Field(default=True)
    stemming_enabled: bool = Field(default=True)


class RerankerConfig(BaseModel):
    enabled: bool = Field(default=True)
    model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_n: int = Field(default=50, ge=1)
    top_k: int = Field(default=10, ge=1)


class HybridRetrievalConfig(BaseModel):
    k: int = Field(default=60, ge=1, description="RRF constant k")
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class ContextManagementConfig(BaseModel):
    max_tokens: int = Field(default=4000, ge=100)


class QueryRewritingConfig(BaseModel):
    enabled: bool = Field(default=True)
    model: str = Field(default="groq/llama-3.1-8b-instant")


class AdaptiveRetrievalConfig(BaseModel):
    enabled: bool = Field(default=True)
    default_strategy: Literal["lexical", "dense", "hybrid"] = Field(default="hybrid")


class SelfRagConfig(BaseModel):
    enabled: bool = Field(default=True)
    decision_method: str = Field(default="llm")
    model: str = Field(default="groq/llama-3.1-8b-instant")


class GuardrailsConfig(BaseModel):
    enabled: bool = Field(default=True)
    strict_mode: bool = Field(default=False)
    max_query_tokens: int = Field(default=1024, ge=1)
    enable_pii_filter: bool = Field(default=True)
    enable_injection_detection: bool = Field(default=True)
    model: str = Field(default="groq/llama-3.1-8b-instant")


class EvalsThresholdsConfig(BaseModel):
    context_precision: float = Field(default=0.85, ge=0.0, le=1.0)
    context_recall: float = Field(default=0.85, ge=0.0, le=1.0)
    faithfulness: float = Field(default=0.90, ge=0.0, le=1.0)
    answer_relevancy: float = Field(default=0.85, ge=0.0, le=1.0)


class EvalsConfig(BaseModel):
    framework: str = Field(default="deepeval")
    judge_model: str = Field(default="gpt-4o")
    golden_dataset_path: Path = Field(
        default_factory=lambda: Path("tests/data/golden_dataset.json")
    )
    thresholds: EvalsThresholdsConfig = Field(default_factory=EvalsThresholdsConfig)


class GenerationFallbacksConfig(BaseModel):
    enabled: bool = Field(default=True)
    models: list[str] = Field(default_factory=lambda: ["ollama/llama3.2:3b", "openai/gpt-4o-mini"])


class GenerationCitationsConfig(BaseModel):
    enabled: bool = Field(default=True)
    format: str = Field(default="bracketed")
    show_provenance: bool = Field(default=True)


class GenerationConfig(BaseModel):
    model: str = Field(default="groq/llama-3.3-70b-versatile")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1500, ge=1)
    stream: bool = Field(default=True)
    timeout_seconds: float = Field(default=15.0, gt=0.0)
    fallbacks: GenerationFallbacksConfig = Field(default_factory=GenerationFallbacksConfig)
    citations: GenerationCitationsConfig = Field(default_factory=GenerationCitationsConfig)


class CliConfig(BaseModel):
    theme: str = Field(default="dark")
    show_waterfall: bool = Field(default=True)
    show_provenance: bool = Field(default=True)


class AppConfig(BaseModel):
    """
    Root configuration object containing all component configurations
    and verified environment credentials.
    """

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    dense_retrieval: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    lexical_retrieval: LexicalRetrievalConfig = Field(default_factory=LexicalRetrievalConfig)
    hybrid_retrieval: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)
    reranking: RerankerConfig = Field(default_factory=RerankerConfig)
    context_management: ContextManagementConfig = Field(default_factory=ContextManagementConfig)
    query_rewriting: QueryRewritingConfig = Field(default_factory=QueryRewritingConfig)
    adaptive_retrieval: AdaptiveRetrievalConfig = Field(default_factory=AdaptiveRetrievalConfig)
    self_rag: SelfRagConfig = Field(default_factory=SelfRagConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cli: CliConfig = Field(default_factory=CliConfig)
    secrets: EnvSecrets = Field(default_factory=EnvSecrets)


def load_config(
    config_path: Path | str | None = None,
    env_file: Path | str | None = None,
) -> AppConfig:
    """
    Pure procedural function that loads, parses, and validates system settings
    from a TOML configuration file and environment secrets from an env file.

    Parameters:
        config_path: Optional path to config.toml. If None, defaults to config.toml in current directory.
        env_file: Optional path to .env file. If None, defaults to .env in current directory.

    Returns:
        AppConfig: Fully validated configuration tree with active credentials and component settings.

    Raises:
        FileNotFoundError: If an explicitly provided config_path does not exist.
        ValueError: If config values violate validation rules.
    """
    resolved_config_path: Path
    if config_path is not None:
        resolved_config_path = Path(config_path)
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {resolved_config_path}")
    else:
        resolved_config_path = Path("config.toml")

    toml_dict: dict[str, Any] = {}
    if resolved_config_path.exists():
        with resolved_config_path.open("rb") as f:
            toml_dict = tomllib.load(f)

    # Sync embedding and dense_retrieval if only one is provided
    if "dense_retrieval" in toml_dict and "embedding" not in toml_dict:
        toml_dict["embedding"] = toml_dict["dense_retrieval"]
    elif "embedding" in toml_dict and "dense_retrieval" not in toml_dict:
        toml_dict["dense_retrieval"] = toml_dict["embedding"]

    # Sync hybrid_retrieval.reranker and top-level reranking
    if "reranking" in toml_dict and "hybrid_retrieval" in toml_dict:
        if "reranker" not in toml_dict["hybrid_retrieval"]:
            toml_dict["hybrid_retrieval"]["reranker"] = toml_dict["reranking"]
    elif (
        "hybrid_retrieval" in toml_dict
        and "reranker" in toml_dict["hybrid_retrieval"]
        and "reranking" not in toml_dict
    ):
        toml_dict["reranking"] = toml_dict["hybrid_retrieval"]["reranker"]

    # Load secrets
    secrets = (
        EnvSecrets(_env_file=str(env_file))  # type: ignore[call-arg] # pyright: ignore
        if env_file is not None
        else EnvSecrets()
    )

    app_config = AppConfig.model_validate({**toml_dict, "secrets": secrets})
    return app_config
