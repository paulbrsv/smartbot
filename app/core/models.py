from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """
    Перечисление типов документов.
    """
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    XLSX = "xlsx"
    HTML = "html"
    MD = "md"
    RTF = "rtf"
    IMAGE = "image"
    ZIP = "zip"
    RAR = "rar"
    API = "api"
    UNKNOWN = "unknown"


class ChunkMetadata(BaseModel):
    """
    Модель метаданных чанка документа.
    """
    document_id: str
    document_name: str
    document_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    chunk_order: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True


class DocumentChunk(BaseModel):
    """
    Модель чанка документа.
    """
    id: Optional[int] = None
    document_id: str
    chunk_text: str
    chunk_order: int
    vector_id: Optional[str] = None
    metadata: Optional[Union[ChunkMetadata, Dict[str, Any]]] = None
    
    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    """
    Модель документа.
    """
    id: Optional[int] = None
    filename: str
    filepath: str
    filetype: str
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.now)
    indexed_at: Optional[datetime] = None
    hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    chunks: Optional[List[DocumentChunk]] = []
    
    class Config:
        arbitrary_types_allowed = True


class SourceDocument(BaseModel):
    """
    Модель исходного документа для возврата в результатах поиска.
    """
    document_id: str
    document_name: str
    document_type: str
    chunk_text: str
    score: float = 0.0
    page_number: Optional[int] = None
    section: Optional[str] = None
    url: Optional[str] = None


class QueryRequest(BaseModel):
    """
    Модель запроса пользователя.
    """
    query_text: str
    history_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    k: int = 5
    use_hybrid_search: bool = True
    hybrid_alpha: float = 0.5  # 0 - полнотекстовый, 1 - векторный


class QueryResponse(BaseModel):
    """
    Модель ответа на запрос пользователя.
    """
    query_id: str
    query_text: str
    response_text: str
    sources: List[SourceDocument] = []
    created_at: datetime = Field(default_factory=datetime.now)
    processing_time: float = 0.0
    model_name: str = ""
    model_provider: str = ""
    
    class Config:
        protected_namespaces = ()  # Отключаем защиту пространства имен "model_"


class UnansweredQuery(BaseModel):
    """
    Модель неотвеченного запроса.
    """
    id: Optional[int] = None
    query_text: str
    reason: str
    created_at: datetime = Field(default_factory=datetime.now)
    resolved: bool = False


class QueryHistory(BaseModel):
    """
    Модель истории запросов.
    """
    id: Optional[int] = None
    query_text: str
    response_text: str
    sources: Optional[List[Dict[str, Any]]] = []
    rating: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ModelProvider(str, Enum):
    """
    Перечисление провайдеров моделей.
    """
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class ModelConfig(BaseModel):
    """
    Модель конфигурации языковой модели.
    """
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2000
    stop_sequences: Optional[List[str]] = None
    timeout: int = 60


class ModelsConfig(BaseModel):
    """
    Модель конфигурации языковых моделей.
    """
    default_provider: ModelProvider = ModelProvider.OLLAMA
    ollama: Optional[ModelConfig] = None
    openai: Optional[ModelConfig] = None
    anthropic: Optional[ModelConfig] = None
    google: Optional[ModelConfig] = None


class VectorStoreConfig(BaseModel):
    """
    Модель конфигурации векторного хранилища.
    """
    type: str = "chromadb"
    path: str = "./data/vector_store"
    collection_name: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    distance_metric: str = "cosine"


class DatabaseConfig(BaseModel):
    """
    Модель конфигурации базы данных.
    """
    type: str = "sqlite"
    path: str = "./data/app.db"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


class ProcessingConfig(BaseModel):
    """
    Модель конфигурации обработки файлов.
    """
    max_file_size_mb: int = 500
    chunk_size: int = 1000
    chunk_overlap: int = 200
    supported_extensions: List[str] = [".txt", ".pdf", ".docx", ".json", ".csv"]


class OCRConfig(BaseModel):
    """
    Модель конфигурации OCR.
    """
    enabled: bool = True
    language: str = "rus+eng"
    dpi: int = 300


class TranslationConfig(BaseModel):
    """
    Модель конфигурации переводчика.
    """
    enabled: bool = False
    provider: str = "google"
    target_language: str = "ru"


class InterfaceConfig(BaseModel):
    """
    Модель конфигурации интерфейса.
    """
    theme: str = "dark"
    language: str = "ru"
    max_history: int = 100


class APIParserConfig(BaseModel):
    """
    Модель конфигурации парсера API.
    """
    api_name: str
    endpoint: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    response_parser: Dict[str, Any]


class Settings(BaseModel):
    """
    Модель настроек приложения.
    """
    models: ModelsConfig
    vector_store: VectorStoreConfig
    database: DatabaseConfig
    processing: ProcessingConfig
    ocr: OCRConfig
    translation: TranslationConfig
    interface: InterfaceConfig


class Feedback(BaseModel):
    """
    Модель обратной связи от пользователя.
    """
    query_id: str
    rating: int
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class IndexingStatus(BaseModel):
    """
    Модель статуса индексации.
    """
    document_id: str
    status: str
    message: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_chunks: int = 0
    processed_chunks: int = 0
    error: Optional[str] = None


class IndexingStatistics(BaseModel):
    """
    Модель статистики индексации.
    """
    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    average_chunks_per_document: float = 0.0
    average_tokens_per_chunk: float = 0.0
    index_size_bytes: int = 0
    index_size_mb: float = 0.0
    last_indexed_at: Optional[datetime] = None


class IndexingBatch(BaseModel):
    """
    Модель пакета индексации.
    """
    batch_id: str
    documents: List[Document]
    status: str
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ModelRequest(BaseModel):
    """
    Модель запроса к языковой модели.
    """
    prompt: str
    model_name: str = "gpt-3.5-turbo"
    provider: ModelProvider = ModelProvider.OLLAMA
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.95
    stop_sequences: Optional[List[str]] = None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    stream: bool = False
    timeout: int = 60
    
    class Config:
        protected_namespaces = ()  # Отключаем защиту пространства имен "model_"


class ModelResponse(BaseModel):
    """
    Модель ответа от языковой модели.
    """
    request_id: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str
    provider: str
    created_at: datetime = Field(default_factory=datetime.now)
    processing_time: float = 0.0
    finish_reason: Optional[str] = None
    
    class Config:
        protected_namespaces = ()  # Отключаем защиту пространства имен "model_"