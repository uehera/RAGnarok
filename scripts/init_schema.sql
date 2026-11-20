-- RAGnarok Relational Schema DDL
-- Authoritative Schema for Embedded DuckDB Metadata Store

CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    raw_file_path TEXT NOT NULL,
    parsed_file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    checksum TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'stored'
        CHECK (status IN ('stored', 'parsing', 'parsed', 'failed')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_chunk_id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    parent_index INTEGER NOT NULL,
    heading_path TEXT[],
    UNIQUE (document_id, parent_index)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    parent_chunk_id UUID NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    heading_path TEXT[],
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued'
        CHECK (stage IN ('queued', 'parsing', 'chunking', 'indexing', 'completed', 'failed')),
    error_message TEXT,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents(checksum);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_document_id ON parent_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_chunk_id ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_document_id ON processing_jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_stage ON processing_jobs(stage);
