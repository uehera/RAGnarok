"""Document ingestion, layout parsing, and parent-child chunking subsystem."""

from ragnarok.ingestion.chunker import (
    AutoTokenizerCounter,
    TiktokenCounter,
    TokenCounterProtocol,
    TwoStageChunker,
    resolve_token_counter,
)
from ragnarok.ingestion.parser import (
    DoclingParserEngine,
    DocumentParser,
    ParserEngineProtocol,
)
from ragnarok.ingestion.pipeline import IngestionPipeline
from ragnarok.ingestion.staging import (
    DuplicateDocumentError,
    StagingResult,
    StagingService,
    compute_sha256,
    detect_file_type,
    validate_staging_file,
)

__all__ = [
    "AutoTokenizerCounter",
    "DoclingParserEngine",
    "DocumentParser",
    "DuplicateDocumentError",
    "IngestionPipeline",
    "ParserEngineProtocol",
    "StagingResult",
    "StagingService",
    "TiktokenCounter",
    "TokenCounterProtocol",
    "TwoStageChunker",
    "compute_sha256",
    "detect_file_type",
    "resolve_token_counter",
    "validate_staging_file",
]
