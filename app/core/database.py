import os
import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import config
from app.core.models import (
    Document, DocumentChunk, QueryHistory, UnansweredQuery, 
    Feedback, IndexingStatus, ChunkMetadata
)

logger = logging.getLogger("smartbot.database")

Base = declarative_base()


class Database:
    """
    Класс для работы с базой данных.
    Поддерживает SQLite и PostgreSQL.
    """

    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.initialize_engine()

    def initialize_engine(self):
        """
        Инициализирует соединение с базой данных на основе конфигурации.
        """
        db_config = config.settings.database
        
        if db_config.type == "sqlite":
            # Создаем директорию для БД, если она не существует
            db_path = Path(db_config.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Создаем строку подключения SQLite
            connection_string = f"sqlite:///{db_config.path}"
            self.engine = create_engine(connection_string, connect_args={"check_same_thread": False})
            
        elif db_config.type == "postgresql":
            # Создаем строку подключения PostgreSQL
            connection_string = (
                f"postgresql://{db_config.username}:{db_config.password}@"
                f"{db_config.host}:{db_config.port}/{db_config.database}"
            )
            self.engine = create_engine(connection_string)
        else:
            raise ValueError(f"Неподдерживаемый тип базы данных: {db_config.type}")
        
        # Создаем фабрику сессий
        self.session_factory = sessionmaker(bind=self.engine)
        
        logger.info(f"Соединение с БД {db_config.type} успешно инициализировано")

    def initialize_database(self):
        """
        Инициализирует базу данных, создавая необходимые таблицы.
        """
        try:
            # Чтение SQL-скрипта для инициализации БД
            script_path = Path(__file__).parent / "db_init.sql"
            
            if not script_path.exists():
                logger.error(f"SQL-скрипт инициализации не найден: {script_path}")
                return False
                
            with open(script_path, "r", encoding="utf-8") as f:
                sql_script = f.read()
                
            # Выполнение SQL-скрипта
            with self.engine.connect() as connection:
                # Разделение скрипта на отдельные запросы и выполнение
                for statement in sql_script.split(";"):
                    if statement.strip():
                        connection.execute(text(statement))
                        
            logger.info("База данных успешно инициализирована")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {str(e)}")
            return False

    def get_session(self) -> Session:
        """
        Возвращает новую сессию базы данных.
        """
        return self.session_factory()
        
    def cursor(self):
        """
        Возвращает курсор для выполнения прямых SQL-запросов.
        Используется для совместимости с кодом, который работает через курсор.
        """
        self._sqlite_conn = sqlite3.connect(self.engine.url.database)
        return self._sqlite_conn.cursor()
        
    def close(self):
        """
        Закрывает соединение с базой данных.
        Используется для совместимости с кодом, который работает через cursor().
        """
        if hasattr(self, '_sqlite_conn'):
            self._sqlite_conn.close()

    # CRUD операции для Document

    def create_document(self, document: Document) -> Optional[int]:
        """
        Создает новый документ в базе данных.
        Возвращает ID созданного документа.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO documents (filename, filepath, filetype, size_bytes, created_at, hash, metadata)
                    VALUES (:filename, :filepath, :filetype, :size_bytes, :created_at, :hash, :metadata)
                    RETURNING id
                """)
                
                result = session.execute(
                    query,
                    {
                        "filename": document.filename,
                        "filepath": document.filepath,
                        "filetype": document.filetype,
                        "size_bytes": document.size_bytes,
                        "created_at": document.created_at,
                        "hash": document.hash,
                        "metadata": json.dumps(document.metadata) if document.metadata else None
                    }
                )
                
                document_id = result.scalar()
                session.commit()
                
                logger.info(f"Документ создан: {document.filename}, ID: {document_id}")
                return document_id
                
        except Exception as e:
            logger.error(f"Ошибка при создании документа {document.filename}: {str(e)}")
            return None

    def update_document_indexed(self, document_id: int, indexed_at: datetime = None) -> bool:
        """
        Обновляет статус индексации документа.
        """
        try:
            with self.get_session() as session:
                indexed_at = indexed_at or datetime.now()
                
                query = text("""
                    UPDATE documents
                    SET indexed_at = :indexed_at
                    WHERE id = :document_id
                """)
                
                session.execute(
                    query,
                    {"document_id": document_id, "indexed_at": indexed_at}
                )
                
                session.commit()
                
                logger.info(f"Обновлен статус индексации документа ID: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении статуса индексации документа {document_id}: {str(e)}")
            return False

    def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """
        Получает документ по ID.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT id, filename, filepath, filetype, size_bytes, created_at, indexed_at, hash, metadata
                    FROM documents
                    WHERE id = :document_id
                """)
                
                result = session.execute(query, {"document_id": document_id}).fetchone()
                
                if not result:
                    return None
                    
                document = Document(
                    id=result[0],
                    filename=result[1],
                    filepath=result[2],
                    filetype=result[3],
                    size_bytes=result[4],
                    created_at=result[5],
                    indexed_at=result[6],
                    hash=result[7],
                    metadata=json.loads(result[8]) if result[8] else {}
                )
                
                return document
                
        except Exception as e:
            logger.error(f"Ошибка при получении документа {document_id}: {str(e)}")
            return None

    def get_document_by_hash(self, hash_value: str) -> Optional[Document]:
        """
        Получает документ по хешу содержимого.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT id, filename, filepath, filetype, size_bytes, created_at, indexed_at, hash, metadata
                    FROM documents
                    WHERE hash = :hash_value
                """)
                
                result = session.execute(query, {"hash_value": hash_value}).fetchone()
                
                if not result:
                    return None
                    
                document = Document(
                    id=result[0],
                    filename=result[1],
                    filepath=result[2],
                    filetype=result[3],
                    size_bytes=result[4],
                    created_at=result[5],
                    indexed_at=result[6],
                    hash=result[7],
                    metadata=json.loads(result[8]) if result[8] else {}
                )
                
                return document
                
        except Exception as e:
            logger.error(f"Ошибка при получении документа по хешу {hash_value}: {str(e)}")
            return None

    def get_all_documents(self, filetype: Optional[str] = None) -> List[Document]:
        """
        Получает список всех документов.
        Можно фильтровать по типу файла.
        """
        try:
            with self.get_session() as session:
                if filetype:
                    query = text("""
                        SELECT id, filename, filepath, filetype, size_bytes, created_at, indexed_at, hash, metadata
                        FROM documents
                        WHERE filetype = :filetype
                        ORDER BY created_at DESC
                    """)
                    results = session.execute(query, {"filetype": filetype}).fetchall()
                else:
                    query = text("""
                        SELECT id, filename, filepath, filetype, size_bytes, created_at, indexed_at, hash, metadata
                        FROM documents
                        ORDER BY created_at DESC
                    """)
                    results = session.execute(query).fetchall()
                
                documents = []
                for result in results:
                    document = Document(
                        id=result[0],
                        filename=result[1],
                        filepath=result[2],
                        filetype=result[3],
                        size_bytes=result[4],
                        created_at=result[5],
                        indexed_at=result[6],
                        hash=result[7],
                        metadata=json.loads(result[8]) if result[8] else {}
                    )
                    documents.append(document)
                
                return documents
                
        except Exception as e:
            logger.error(f"Ошибка при получении списка документов: {str(e)}")
            return []

    def delete_document(self, document_id: int) -> bool:
        """
        Удаляет документ из базы данных.
        """
        try:
            with self.get_session() as session:
                # Сначала удаляем чанки документа
                chunk_query = text("""
                    DELETE FROM document_chunks
                    WHERE document_id = :document_id
                """)
                
                session.execute(chunk_query, {"document_id": document_id})
                
                # Затем удаляем сам документ
                doc_query = text("""
                    DELETE FROM documents
                    WHERE id = :document_id
                """)
                
                session.execute(doc_query, {"document_id": document_id})
                
                session.commit()
                
                logger.info(f"Документ ID: {document_id} успешно удален")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при удалении документа {document_id}: {str(e)}")
            return False

    # CRUD операции для DocumentChunk

    def create_document_chunk(self, chunk: DocumentChunk) -> Optional[int]:
        """
        Создает новый чанк документа в базе данных.
        Возвращает ID созданного чанка.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO document_chunks (document_id, chunk_text, chunk_order, vector_id)
                    VALUES (:document_id, :chunk_text, :chunk_order, :vector_id)
                    RETURNING id
                """)
                
                result = session.execute(
                    query,
                    {
                        "document_id": chunk.document_id,
                        "chunk_text": chunk.chunk_text,
                        "chunk_order": chunk.chunk_order,
                        "vector_id": chunk.vector_id
                    }
                )
                
                chunk_id = result.scalar()
                session.commit()
                
                logger.debug(f"Чанк документа создан: документ ID: {chunk.document_id}, чанк ID: {chunk_id}")
                return chunk_id
                
        except Exception as e:
            logger.error(f"Ошибка при создании чанка документа {chunk.document_id}: {str(e)}")
            return None

    def create_document_chunks_batch(self, chunks: List[DocumentChunk]) -> bool:
        """
        Создает несколько чанков документа в одной транзакции.
        """
        if not chunks:
            return True
            
        try:
            with self.get_session() as session:
                for chunk in chunks:
                    query = text("""
                        INSERT INTO document_chunks (document_id, chunk_text, chunk_order, vector_id)
                        VALUES (:document_id, :chunk_text, :chunk_order, :vector_id)
                    """)
                    
                    session.execute(
                        query,
                        {
                            "document_id": chunk.document_id,
                            "chunk_text": chunk.chunk_text,
                            "chunk_order": chunk.chunk_order,
                            "vector_id": chunk.vector_id
                        }
                    )
                
                session.commit()
                
                logger.info(f"Создано {len(chunks)} чанков для документа ID: {chunks[0].document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при создании чанков документа: {str(e)}")
            return False

    def get_chunks_by_document_id(self, document_id: int) -> List[DocumentChunk]:
        """
        Получает все чанки определенного документа.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT id, document_id, chunk_text, chunk_order, vector_id
                    FROM document_chunks
                    WHERE document_id = :document_id
                    ORDER BY chunk_order
                """)
                
                results = session.execute(query, {"document_id": document_id}).fetchall()
                
                chunks = []
                for result in results:
                    chunk = DocumentChunk(
                        id=result[0],
                        document_id=result[1],
                        chunk_text=result[2],
                        chunk_order=result[3],
                        vector_id=result[4]
                    )
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            logger.error(f"Ошибка при получении чанков документа {document_id}: {str(e)}")
            return []

    def delete_chunks_by_document_id(self, document_id: int) -> bool:
        """
        Удаляет все чанки определенного документа.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    DELETE FROM document_chunks
                    WHERE document_id = :document_id
                """)
                
                session.execute(query, {"document_id": document_id})
                
                session.commit()
                
                logger.info(f"Удалены все чанки документа ID: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при удалении чанков документа {document_id}: {str(e)}")
            return False

    # CRUD операции для QueryHistory

    def create_query_history(self, query_history: QueryHistory) -> Optional[int]:
        """
        Создает новую запись в истории запросов.
        Возвращает ID созданной записи.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO query_history (query_text, response_text, sources, rating, created_at)
                    VALUES (:query_text, :response_text, :sources, :rating, :created_at)
                    RETURNING id
                """)
                
                result = session.execute(
                    query,
                    {
                        "query_text": query_history.query_text,
                        "response_text": query_history.response_text,
                        "sources": json.dumps(query_history.sources) if query_history.sources else None,
                        "rating": query_history.rating,
                        "created_at": query_history.created_at
                    }
                )
                
                history_id = result.scalar()
                session.commit()
                
                logger.info(f"Создана запись в истории запросов ID: {history_id}")
                return history_id
                
        except Exception as e:
            logger.error(f"Ошибка при создании записи в истории запросов: {str(e)}")
            return None

    def get_query_history(self, limit: int = 100) -> List[QueryHistory]:
        """
        Получает историю запросов.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT id, query_text, response_text, sources, rating, created_at
                    FROM query_history
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                
                results = session.execute(query, {"limit": limit}).fetchall()
                
                history_items = []
                for result in results:
                    item = QueryHistory(
                        id=result[0],
                        query_text=result[1],
                        response_text=result[2],
                        sources=json.loads(result[3]) if result[3] else None,
                        rating=result[4],
                        created_at=result[5]
                    )
                    history_items.append(item)
                
                return history_items
                
        except Exception as e:
            logger.error(f"Ошибка при получении истории запросов: {str(e)}")
            return []

# Глобальный экземпляр базы данных
_db_instance = None

def get_db_connection() -> Database:
    """
    Возвращает экземпляр класса Database.
    Если экземпляр еще не создан, создает его.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance

def init_database():
    """
    Инициализирует базу данных.
    Функция-обертка для совместимости с импортом в main.py
    """
    return get_db_connection().initialize_database()

    # CRUD операции для UnansweredQuery

    def create_unanswered_query(self, query: UnansweredQuery) -> Optional[int]:
        """
        Создает новую запись о неотвеченном запросе.
        Возвращает ID созданной записи.
        """
        try:
            with self.get_session() as session:
                query_sql = text("""
                    INSERT INTO unanswered_queries (query_text, reason, created_at, resolved)
                    VALUES (:query_text, :reason, :created_at, :resolved)
                    RETURNING id
                """)
                
                result = session.execute(
                    query_sql,
                    {
                        "query_text": query.query_text,
                        "reason": query.reason,
                        "created_at": query.created_at,
                        "resolved": query.resolved
                    }
                )
                
                query_id = result.scalar()
                session.commit()
                
                logger.info(f"Создана запись о неотвеченном запросе ID: {query_id}")
                return query_id
                
        except Exception as e:
            logger.error(f"Ошибка при создании записи о неотвеченном запросе: {str(e)}")
            return None

    def get_unanswered_queries(self, resolved: bool = False, limit: int = 100) -> List[UnansweredQuery]:
        """
        Получает список неотвеченных запросов.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT id, query_text, reason, created_at, resolved
                    FROM unanswered_queries
                    WHERE resolved = :resolved
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                
                results = session.execute(query, {"resolved": resolved, "limit": limit}).fetchall()
                
                queries = []
                for result in results:
                    item = UnansweredQuery(
                        id=result[0],
                        query_text=result[1],
                        reason=result[2],
                        created_at=result[3],
                        resolved=result[4]
                    )
                    queries.append(item)
                
                return queries
                
        except Exception as e:
            logger.error(f"Ошибка при получении неотвеченных запросов: {str(e)}")
            return []

    def mark_query_as_resolved(self, query_id: int) -> bool:
        """
        Отмечает неотвеченный запрос как решенный.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    UPDATE unanswered_queries
                    SET resolved = TRUE
                    WHERE id = :query_id
                """)
                
                session.execute(query, {"query_id": query_id})
                
                session.commit()
                
                logger.info(f"Запрос ID: {query_id} отмечен как решенный")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении статуса запроса {query_id}: {str(e)}")
            return False

    # CRUD операции для Feedback

    def create_feedback(self, feedback: Feedback) -> Optional[int]:
        """
        Создает новую запись обратной связи.
        Возвращает ID созданной записи.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO feedback (query_id, rating, comment, created_at)
                    VALUES (:query_id, :rating, :comment, :created_at)
                    RETURNING id
                """)
                
                result = session.execute(
                    query,
                    {
                        "query_id": feedback.query_id,
                        "rating": feedback.rating,
                        "comment": feedback.comment,
                        "created_at": feedback.created_at
                    }
                )
                
                feedback_id = result.scalar()
                session.commit()
                
                # Обновляем рейтинг в истории запросов
                update_query = text("""
                    UPDATE query_history
                    SET rating = :rating
                    WHERE id = :query_id
                """)
                
                session.execute(update_query, {"rating": feedback.rating, "query_id": feedback.query_id})
                
                session.commit()
                
                logger.info(f"Создана запись обратной связи ID: {feedback_id}")
                return feedback_id
                
        except Exception as e:
            logger.error(f"Ошибка при создании записи обратной связи: {str(e)}")
            return None

    # CRUD операции для IndexingStatus

    def create_indexing_status(self, status: IndexingStatus) -> Optional[int]:
        """
        Создает новую запись о статусе индексации.
        Возвращает ID созданной записи.
        """
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO indexing_status (document_id, status, message, start_time, total_chunks, processed_chunks)
                    VALUES (:document_id, :status, :message, :start_time, :total_chunks, :processed_chunks)
                    RETURNING id
                """)
                
                result = session.execute(
                    query,
                    {
                        "document_id": status.document_id,
                        "status": status.status,
                        "message": status.message,
                        "start_time": status.start_time,
                        "total_chunks": status.total_chunks,
                        "processed_chunks": status.processed_chunks
                    }
                )
                
                status_id = result.scalar()
                session.commit()
                
                logger.info(f"Создана запись о статусе индексации ID: {status_id}")
                return status_id
                
        except Exception as e:
            logger.error(f"Ошибка при создании записи о статусе индексации: {str(e)}")
            return None

    def update_indexing_status(self, document_id: str, status: str, 
                               end_time: Optional[datetime] = None,
                               processed_chunks: Optional[int] = None,
                               error: Optional[str] = None) -> bool:
        """
        Обновляет статус индексации документа.
        """
        try:
            with self.get_session() as session:
                update_values = {"document_id": document_id, "status": status}
                update_fields = "status = :status"
                
                if end_time:
                    update_values["end_time"] = end_time
                    update_fields += ", end_time = :end_time"
                    
                if processed_chunks is not None:
                    update_values["processed_chunks"] = processed_chunks
                    update_fields += ", processed_chunks = :processed_chunks"
                    
                if error:
                    update_values["error"] = error
                    update_fields += ", error = :error"
                
                query = text(f"""
                    UPDATE indexing_status
                    SET {update_fields}
                    WHERE document_id = :document_id
                    AND end_time IS NULL
                """)
                
                session.execute(query, update_values)
                
                session.commit()
                
                logger.info(f"Обновлен статус индексации документа ID: {document_id}, статус: {status}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении статуса индексации документа {document_id}: {str(e)}")
            return False

    def get_indexing_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику индексации.
        """
        try:
            with self.get_session() as session:
                # Общее количество документов
                doc_query = text("SELECT COUNT(*) FROM documents")
                total_documents = session.execute(doc_query).scalar() or 0
                
                # Общее количество чанков
                chunk_query = text("SELECT COUNT(*) FROM document_chunks")
                total_chunks = session.execute(chunk_query).scalar() or 0
                
                # Средний размер документа в чанках
                if total_documents > 0:
                    avg_chunks_query = text("SELECT AVG(chunks) FROM (SELECT document_id, COUNT(*) as chunks FROM document_chunks GROUP BY document_id) as chunk_counts")
                    avg_chunks = session.execute(avg_chunks_query).scalar() or 0
                else:
                    avg_chunks = 0
                
                # Последняя индексация
                last_indexed_query = text("SELECT MAX(indexed_at) FROM documents WHERE indexed_at IS NOT NULL")
                last_indexed = session.execute(last_indexed_query).scalar()
                
                return {
                    "total_documents": total_documents,
                    "total_chunks": total_chunks,
                    "average_chunks_per_document": avg_chunks,
                    "last_indexed_at": last_indexed
                }
                
        except Exception as e:
            logger.error(f"Ошибка при получении статистики индексации: {str(e)}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "average_chunks_per_document": 0,
                "last_indexed_at": None
            }

    # Настройки

    def set_setting(self, key: str, value: Any) -> bool:
        """
        Устанавливает значение настройки.
        """
        try:
            with self.get_session() as session:
                # Проверяем, существует ли настройка
                check_query = text("SELECT key FROM settings WHERE key = :key")
                exists = session.execute(check_query, {"key": key}).fetchone() is not None
                
                if exists:
                    # Обновляем существующую настройку
                    update_query = text("""
                        UPDATE settings
                        SET value = :value, updated_at = :updated_at
                        WHERE key = :key
                    """)
                    
                    session.execute(
                        update_query,
                        {"key": key, "value": json.dumps(value), "updated_at": datetime.now()}
                    )
                else:
                    # Создаем новую настройку
                    insert_query = text("""
                        INSERT INTO settings (key, value, updated_at)
                        VALUES (:key, :value, :updated_at)
                    """)
                    
                    session.execute(
                        insert_query,
                        {"key": key, "value": json.dumps(value), "updated_at": datetime.now()}
                    )
                
                session.commit()
                
                logger.info(f"Настройка {key} успешно установлена")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при установке настройки {key}: {str(e)}")
            return False

    def get_setting(self, key: str, default_value: Any = None) -> Any:
        """
        Получает значение настройки.
        """
        try:
            with self.get_session() as session:
                query = text("SELECT value FROM settings WHERE key = :key")
                
                result = session.execute(query, {"key": key}).fetchone()
                
                if result:
                    return json.loads(result[0])
                else:
                    return default_value
                    
        except Exception as e:
            logger.error(f"Ошибка при получении настройки {key}: {str(e)}")
            return default_value

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Получает все настройки.
        """
        try:
            with self.get_session() as session:
                query = text("SELECT key, value FROM settings")
                
                results = session.execute(query).fetchall()
                
                settings = {}
                for key, value in results:
                    settings[key] = json.loads(value)
                
                return settings
                
        except Exception as e:
            logger.error(f"Ошибка при получении всех настроек: {str(e)}")
            return {}


# Создаем экземпляр базы данных
db = Database()