-- Документы
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(500) NOT NULL,
    filetype VARCHAR(50) NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP,
    hash VARCHAR(64),
    metadata JSON
);

-- Чанки документов
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    vector_id VARCHAR(100),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- История запросов
CREATE TABLE IF NOT EXISTS query_history (
    id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    sources JSON,
    rating INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Неотвеченные вопросы
CREATE TABLE IF NOT EXISTS unanswered_queries (
    id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    reason VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE
);

-- Настройки
CREATE TABLE IF NOT EXISTS settings (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Обратная связь
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    query_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES query_history(id) ON DELETE CASCADE
);

-- Статус индексации
CREATE TABLE IF NOT EXISTS indexing_status (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    message TEXT,
    start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    total_chunks INTEGER DEFAULT 0,
    processed_chunks INTEGER DEFAULT 0,
    error TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Комментарии обратной связи (для API)
CREATE TABLE IF NOT EXISTS feedback_comments (
    id INTEGER PRIMARY KEY,
    query_id VARCHAR(100) NOT NULL,
    comment TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для ускорения поиска
CREATE INDEX IF NOT EXISTS idx_documents_filetype ON documents(filetype);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_id ON document_chunks(vector_id);
CREATE INDEX IF NOT EXISTS idx_query_history_created_at ON query_history(created_at);
CREATE INDEX IF NOT EXISTS idx_unanswered_queries_resolved ON unanswered_queries(resolved);
CREATE INDEX IF NOT EXISTS idx_feedback_comments_query_id ON feedback_comments(query_id);