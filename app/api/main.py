"""
Главный API модуль приложения SmartBot.
Реализует REST API эндпоинты для взаимодействия с системой.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from app.core.config import get_settings
from app.core.database import get_db_connection, init_database
from app.core.vector_store import VectorStore
from app.indexing.indexer import FileIndexer
from app.rag.engine import RAGEngine
from app.rag.multimodal import MultimodalProcessor
from app.models.llm_provider import get_available_models

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Получаем настройки
settings = get_settings()

# Создаем приложение FastAPI
app = FastAPI(
    title="SmartBot RAG Assistant API",
    description="API для универсального чат-бота с индексацией и поиском данных",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные объекты
vector_store = None
file_indexer = None
rag_engine = None
multimodal_processor = None


# Pydantic модели для API
class ChatRequest(BaseModel):
    """Модель запроса для чата"""
    query: str = Field(..., description="Текст запроса пользователя")
    image_data: Optional[List[str]] = Field(None, description="Данные изображений в base64")
    search_options: Optional[Dict[str, Any]] = Field(None, description="Параметры поиска")
    llm_settings: Optional[Dict[str, Any]] = Field(None, description="Настройки языковой модели")
    chat_history: Optional[List[Dict[str, Any]]] = Field(None, description="История чата")
    user_id: Optional[str] = Field(None, description="ID пользователя")


class ChatResponse(BaseModel):
    """Модель ответа чата"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    execution_time: float
    metadata: Dict[str, Any]


class DocumentInfo(BaseModel):
    """Информация о документе"""
    id: int
    filename: str
    filepath: str
    filetype: str
    size_bytes: int
    created_at: datetime
    indexed_at: datetime
    metadata: Dict[str, Any]


class IndexingStatus(BaseModel):
    """Статус индексации"""
    total_files: int
    indexed_files: int
    failed_files: int
    in_progress: bool
    current_file: Optional[str]


class SettingsUpdate(BaseModel):
    """Модель для обновления настроек"""
    key: str
    value: Any


class ReindexRequest(BaseModel):
    """Запрос на переиндексацию"""
    document_ids: Optional[List[int]] = Field(None, description="ID документов для переиндексации")
    reindex_all: bool = Field(False, description="Переиндексировать все документы")


class FeedbackRequest(BaseModel):
    """Запрос обратной связи"""
    query_id: str
    rating: int = Field(..., ge=1, le=5, description="Оценка от 1 до 5")
    comment: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    global vector_store, file_indexer, rag_engine, multimodal_processor
    
    try:
        # Инициализируем базу данных
        init_database()
        logger.info("Database initialized")
        
        # Создаем объекты системы
        vector_store = VectorStore()
        file_indexer = FileIndexer(vector_store=vector_store)
        multimodal_processor = MultimodalProcessor(vector_store=vector_store)
        rag_engine = RAGEngine(
            vector_store=vector_store,
            multimodal_processor=multimodal_processor
        )
        
        logger.info("SmartBot API started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "SmartBot RAG Assistant API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/api/upload", response_model=Dict[str, Any])
async def upload_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Загрузка файлов для индексации"""
    try:
        upload_results = []
        
        for file in files:
            # Проверяем размер файла
            file_size = 0
            content = await file.read()
            file_size = len(content)
            await file.seek(0)
            
            max_size = settings.processing.max_file_size_mb * 1024 * 1024
            if file_size > max_size:
                upload_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"File size exceeds limit of {settings.processing.max_file_size_mb}MB"
                })
                continue
            
            # Сохраняем файл
            upload_dir = os.path.join(os.path.expanduser("~"), ".smartbot", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Добавляем задачу индексации в фоновые задачи
            background_tasks.add_task(
                file_indexer.index_file,
                file_path=file_path
            )
            
            upload_results.append({
                "filename": file.filename,
                "status": "queued",
                "message": "File queued for indexing"
            })
        
        return {
            "status": "success",
            "files": upload_results
        }
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_documents(
    limit: int = 100,
    offset: int = 0,
    filetype: Optional[str] = None
):
    """Получение списка документов"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Формируем запрос
        query = "SELECT * FROM documents"
        params = []
        
        if filetype:
            query += " WHERE filetype = ?"
            params.append(filetype)
        
        query += " ORDER BY indexed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        documents = cursor.fetchall()
        
        # Преобразуем в модели
        result = []
        for doc in documents:
            result.append(DocumentInfo(
                id=doc[0],
                filename=doc[1],
                filepath=doc[2],
                filetype=doc[3],
                size_bytes=doc[4],
                created_at=datetime.fromisoformat(doc[5]),
                indexed_at=datetime.fromisoformat(doc[6]),
                metadata={}
            ))
        
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Отправка вопроса и получение ответа"""
    try:
        # Обрабатываем запрос через RAG engine
        result = rag_engine.process_query(
            query_text=request.query,
            image_data=request.image_data,
            search_options=request.search_options,
            llm_settings=request.llm_settings,
            chat_history=request.chat_history,
            user_id=request.user_id
        )
        
        # Преобразуем результат в модель ответа
        return ChatResponse(
            query=result.query_text,
            response=result.response_text,
            sources=result.sources,
            execution_time=result.processing_time,
            metadata={}
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Получение истории диалогов"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM query_history"
        params = []
        
        if user_id:
            query += " WHERE metadata LIKE ?"
            params.append(f'%"user_id": "{user_id}"%')
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        history = cursor.fetchall()
        
        result = []
        for item in history:
            result.append({
                "id": item[0],
                "query": item[1],
                "response": item[2],
                "sources": item[3],
                "rating": item[4],
                "created_at": item[5]
            })
        
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex")
async def reindex_documents(
    request: ReindexRequest,
    background_tasks: BackgroundTasks
):
    """Переиндексация документов"""
    try:
        if request.reindex_all:
            # Переиндексируем все документы
            background_tasks.add_task(file_indexer.reindex_all)
            return {
                "status": "success",
                "message": "All documents queued for reindexing"
            }
        elif request.document_ids:
            # Переиндексируем выбранные документы
            for doc_id in request.document_ids:
                background_tasks.add_task(
                    file_indexer.reindex_document,
                    document_id=doc_id
                )
            return {
                "status": "success",
                "message": f"{len(request.document_ids)} documents queued for reindexing"
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Either reindex_all or document_ids must be provided"
            )
    except Exception as e:
        logger.error(f"Error reindexing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings")
async def get_settings():
    """Получение настроек"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT key, value FROM settings")
        settings_data = cursor.fetchall()
        
        result = {}
        for key, value in settings_data:
            try:
                # Пытаемся распарсить JSON
                import json
                result[key] = json.loads(value)
            except:
                result[key] = value
        
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/settings")
async def update_settings(update: SettingsUpdate):
    """Обновление настроек"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Сериализуем значение в JSON если это объект
        import json
        if isinstance(update.value, (dict, list)):
            value_str = json.dumps(update.value)
        else:
            value_str = str(update.value)
        
        # Обновляем или вставляем настройку
        cursor.execute("""
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (update.key, value_str, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Setting updated"}
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: int):
    """Удаление документа"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Получаем информацию о документе
        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        doc = cursor.fetchone()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Удаляем из векторного хранилища
        cursor.execute(
            "SELECT vector_id FROM document_chunks WHERE document_id = ?",
            (document_id,)
        )
        vector_ids = [row[0] for row in cursor.fetchall()]
        
        for vector_id in vector_ids:
            vector_store.delete(vector_id)
        
        # Удаляем из базы данных
        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Document deleted"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Отправка обратной связи"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Обновляем рейтинг в истории запросов
        cursor.execute("""
            UPDATE query_history 
            SET rating = ?
            WHERE metadata LIKE ?
        """, (feedback.rating, f'%"query_id": "{feedback.query_id}"%'))
        
        # Если есть комментарий, сохраняем его отдельно
        if feedback.comment:
            cursor.execute("""
                INSERT INTO feedback_comments (query_id, comment, created_at)
                VALUES (?, ?, ?)
            """, (feedback.query_id, feedback.comment, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Feedback submitted"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """Получение статуса системы"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Получаем статистику
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_documents = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM query_history")
        total_queries = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM unanswered_queries WHERE resolved = 0")
        unanswered_queries = cursor.fetchone()[0]
        
        conn.close()
        
        # Проверяем доступность моделей
        available_models = get_available_models()
        
        return {
            "status": "running",
            "statistics": {
                "total_documents": total_documents,
                "total_queries": total_queries,
                "unanswered_queries": unanswered_queries
            },
            "available_models": available_models,
            "vector_store": {
                "type": settings.vector_store.type,
                "status": "connected" if vector_store else "disconnected"
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_api():
    """Запуск API сервера"""
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    run_api()