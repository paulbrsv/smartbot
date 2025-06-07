import os
import logging
import time
import hashlib
import json
import re
import concurrent.futures
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from datetime import datetime
import traceback
import magic
import shutil
from pathlib import Path
import sqlite3
from urllib.parse import urlparse

from app.core.config import get_settings
from app.core.models import Document
from app.core.database import get_db_connection
from app.core.vector_store import VectorStore
from app.indexing.parsers.base import ParserFactory, ParsedDocument

logger = logging.getLogger(__name__)
settings = get_settings()


class Indexer:
    """
    Основной класс для индексации данных.
    
    Отвечает за обнаружение, классификацию и обработку файлов,
    а также за сохранение извлеченных данных в БД и векторное хранилище.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        db_connection: Optional[sqlite3.Connection] = None
    ):
        """
        Инициализирует индексатор.
        
        Args:
            vector_store: Объект векторного хранилища
            db_connection: Соединение с базой данных
        """
        self.vector_store = vector_store or VectorStore()
        self.db_connection = db_connection or get_db_connection()
        self.temp_dir = os.path.join(os.path.expanduser("~"), ".smartbot", "temp")
        
        # Создаем временную директорию, если её нет
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        
        # Инициализируем счетчик для отслеживания прогресса
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.current_operation = "idle"
        self.indexing_status = {}
        self.lock = threading.Lock()
        
        # Загружаем все парсеры для проверки их доступности
        self.parser_factory = ParserFactory()
        self.available_parsers = ParserFactory.get_all_parsers()
        # Добавляем парсеры в экземпляр фабрики
        self.parser_factory.parsers = self.available_parsers
        logger.info(f"Initialized indexer with {len(self.available_parsers)} available parsers")
    
    def index_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size_mb: Optional[int] = None,
        max_files: Optional[int] = None,
        callback: Optional[Callable[[str, float], None]] = None,
        parallel: bool = True,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Индексирует все файлы в указанной директории.
        
        Args:
            directory_path: Путь к директории
            recursive: Индексировать рекурсивно
            file_patterns: Список паттернов имен файлов для включения
            exclude_patterns: Список паттернов имен файлов для исключения
            max_file_size_mb: Максимальный размер файла в МБ
            max_files: Максимальное количество файлов для индексации
            callback: Функция обратного вызова для отслеживания прогресса
            parallel: Использовать параллельную обработку
            max_workers: Максимальное количество рабочих потоков
            
        Returns:
            Словарь с результатами индексации
        """
        try:
            # Обновляем статус индексации
            self.current_operation = "indexing_directory"
            self.processed_files = 0
            self.failed_files = 0
            
            # Устанавливаем значения по умолчанию
            if max_file_size_mb is None:
                max_file_size_mb = settings.processing.max_file_size_mb
            
            start_time = time.time()
            
            # Получаем список файлов для индексации
            files_to_index = self._discover_files(
                directory_path,
                recursive=recursive,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                max_file_size_mb=max_file_size_mb,
                max_files=max_files
            )
            
            self.total_files = len(files_to_index)
            logger.info(f"Found {self.total_files} files to index in {directory_path}")
            
            # Обновляем статус
            self.indexing_status = {
                "status": "indexing",
                "total_files": self.total_files,
                "processed_files": 0,
                "failed_files": 0,
                "start_time": start_time,
                "directory": directory_path
            }
            
            # Создаем функцию для обработки файла
            def process_file(file_path):
                try:
                    # Индексируем файл
                    result = self.index_file(file_path)
                    
                    with self.lock:
                        self.processed_files += 1
                        
                        if not result["success"]:
                            self.failed_files += 1
                        
                        # Обновляем статус
                        progress = self.processed_files / self.total_files if self.total_files > 0 else 0
                        self.indexing_status.update({
                            "processed_files": self.processed_files,
                            "failed_files": self.failed_files,
                            "progress": progress,
                            "current_file": file_path
                        })
                        
                        # Вызываем колбэк, если он предоставлен
                        if callback:
                            callback(file_path, progress)
                    
                    return result
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    logger.error(traceback.format_exc())
                    
                    with self.lock:
                        self.processed_files += 1
                        self.failed_files += 1
                        
                        # Обновляем статус
                        progress = self.processed_files / self.total_files if self.total_files > 0 else 0
                        self.indexing_status.update({
                            "processed_files": self.processed_files,
                            "failed_files": self.failed_files,
                            "progress": progress,
                            "current_file": file_path
                        })
                        
                        # Вызываем колбэк, если он предоставлен
                        if callback:
                            callback(file_path, progress)
                    
                    return {
                        "file_path": file_path,
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
            
            # Обрабатываем файлы
            results = []
            
            if parallel and self.total_files > 1:
                # Параллельная обработка
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_file, file_path) for file_path in files_to_index]
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Error in worker thread: {e}")
                            logger.error(traceback.format_exc())
            else:
                # Последовательная обработка
                for file_path in files_to_index:
                    result = process_file(file_path)
                    results.append(result)
            
            # Вычисляем статистику
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            successful_files = sum(1 for result in results if result["success"])
            
            # Обновляем финальный статус
            self.indexing_status.update({
                "status": "completed",
                "end_time": end_time,
                "elapsed_time": elapsed_time,
                "successful_files": successful_files,
                "failed_files": self.failed_files
            })
            
            # Готовим результат
            index_result = {
                "directory": directory_path,
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "successful_files": successful_files,
                "failed_files": self.failed_files,
                "elapsed_time": elapsed_time,
                "files": results
            }
            
            logger.info(f"Indexing completed: {successful_files} successful, {self.failed_files} failed, {elapsed_time:.2f} seconds")
            
            # Сбрасываем текущую операцию
            self.current_operation = "idle"
            
            return index_result
        except Exception as e:
            logger.error(f"Error indexing directory {directory_path}: {e}")
            logger.error(traceback.format_exc())
            
            # Обновляем статус ошибки
            self.indexing_status.update({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            # Сбрасываем текущую операцию
            self.current_operation = "idle"
            
            return {
                "directory": directory_path,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def index_file(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
        parser_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Индексирует указанный файл.
        
        Args:
            file_path: Путь к файлу
            custom_metadata: Пользовательские метаданные
            parser_options: Опции для парсера
            
        Returns:
            Словарь с результатами индексации
        """
        try:
            logger.info(f"Starting indexing of file: {file_path}")
            
            # Проверяем существование файла
            if not os.path.exists(file_path):
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": "File not found"
                }
            
            # Проверяем, что это файл, а не директория
            if os.path.isdir(file_path):
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": "Path is a directory, not a file"
                }
            
            # Проверяем размер файла
            file_size = os.path.getsize(file_path)
            max_size_bytes = settings.processing.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size_bytes} bytes)"
                }
            
            # Вычисляем хеш файла
            logger.info(f"Computing file hash for: {file_path}")
            file_hash = self._compute_file_hash(file_path)
            logger.info(f"File hash computed: {file_hash}")
            
            # Проверяем, есть ли файл уже в базе данных
            logger.info(f"Checking if file already exists in database")
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT id, hash, indexed_at FROM documents WHERE filepath = ? OR hash = ?",
                (file_path, file_hash)
            )
            existing_doc = cursor.fetchone()
            
            # Проверяем, изменился ли файл с момента последней индексации
            if existing_doc:
                doc_id, doc_hash, indexed_at = existing_doc
                
                # Если хеш не изменился, проверяем, не нужно ли обновить метаданные
                if doc_hash == file_hash and not custom_metadata:
                    # Файл не изменился, можно пропустить повторную индексацию
                    logger.info(f"File {file_path} already indexed with same hash, skipping")
                    return {
                        "file_path": file_path,
                        "success": True,
                        "document_id": doc_id,
                        "status": "unchanged",
                        "message": f"File already indexed at {indexed_at} and hasn't changed"
                    }
            
            # Определяем тип файла
            file_type = self._determine_file_type(file_path)
            
            # Подбираем парсер для файла
            logger.info(f"Getting parser for file: {file_path}, file type: {file_type}")
            parser = self.parser_factory.get_parser(file_path)
            
            if not parser:
                logger.error(f"No parser found for file type: {file_type}")
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": f"No parser found for file type: {file_type}"
                }
            
            logger.info(f"Using parser: {parser.__class__.__name__} for file: {file_path}")
            
            # Готовим опции парсера
            options = {
                "chunk_size": settings.processing.chunk_size,
                "chunk_overlap": settings.processing.chunk_overlap
            }
            
            if parser_options:
                options.update(parser_options)
            
            # Парсим файл
            logger.info(f"Starting to parse file: {file_path} with options: {options}")
            parsed_doc = parser.parse(file_path, **options)
            logger.info(f"Parse completed. Content length: {len(parsed_doc.content) if parsed_doc.content else 0}, Chunks: {len(parsed_doc.chunks) if parsed_doc.chunks else 0}")
            
            # Подготавливаем метаданные
            metadata = parsed_doc.metadata or {}
            
            if custom_metadata:
                metadata.update(custom_metadata)
            
            # Добавляем базовые метаданные
            metadata.update({
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "filetype": file_type,
                "size_bytes": file_size,
                "hash": file_hash,
                "parser": parser.__class__.__name__,
                "indexed_at": datetime.now().isoformat()
            })
            
            # Сохраняем документ в базе данных
            logger.info(f"Saving document to database: {file_path}")
            document_id = self._save_document_to_db(
                file_path=file_path,
                content=parsed_doc.content,
                metadata=metadata,
                chunks=parsed_doc.chunks,
                existing_doc_id=existing_doc[0] if existing_doc else None
            )
            logger.info(f"Document saved to database with ID: {document_id}")
            
            # Добавляем документ в векторное хранилище
            if parsed_doc.chunks:
                logger.info(f"Adding {len(parsed_doc.chunks)} chunks to vector store")
                vector_ids = self._add_to_vector_store(
                    document_id=document_id,
                    chunks=parsed_doc.chunks,
                    metadata=metadata
                )
                
                # Обновляем информацию о векторных ID в базе данных
                self._update_vector_ids(document_id, vector_ids)
            
            return {
                "file_path": file_path,
                "success": True,
                "document_id": document_id,
                "chunk_count": len(parsed_doc.chunks) if parsed_doc.chunks else 0,
                "status": "updated" if existing_doc else "created",
                "file_type": file_type,
                "parser": parser.__class__.__name__
            }
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "file_path": file_path,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def index_api(
        self,
        api_config: Dict[str, Any],
        api_name: Optional[str] = None,
        parser_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Индексирует данные из API.
        
        Args:
            api_config: Конфигурация API
            api_name: Название API (если не указано, берется из конфигурации)
            parser_options: Опции для парсера
            
        Returns:
            Словарь с результатами индексации
        """
        try:
            # Получаем имя API
            if not api_name and "api_name" in api_config:
                api_name = api_config["api_name"]
            else:
                # Если имя API не указано, генерируем его из URL
                if "endpoint" in api_config:
                    url = api_config["endpoint"]
                    parsed_url = urlparse(url)
                    api_name = f"api_{parsed_url.netloc.replace('.', '_')}"
                else:
                    api_name = f"api_{int(time.time())}"
            
            # Создаем временный файл конфигурации
            temp_file_path = os.path.join(self.temp_dir, f"{api_name}.apiconfig")
            
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(api_config, f, ensure_ascii=False, indent=2)
            
            # Готовим опции парсера
            options = {
                "chunk_size": settings.processing.chunk_size,
                "chunk_overlap": settings.processing.chunk_overlap,
                "api_config": api_config
            }
            
            if parser_options:
                options.update(parser_options)
            
            # Получаем парсер API
            from app.indexing.parsers.api import APIParser
            parser = APIParser()
            
            # Парсим API
            parsed_doc = parser.parse(temp_file_path, **options)
            
            # Подготавливаем метаданные
            metadata = parsed_doc.metadata or {}
            
            # Добавляем базовые метаданные
            metadata.update({
                "api_name": api_name,
                "content_type": "api_data",
                "endpoint": api_config.get("endpoint", ""),
                "method": api_config.get("method", "GET"),
                "indexed_at": datetime.now().isoformat()
            })
            
            # Генерируем фиктивный путь к файлу для хранения в БД
            file_path = f"api://{api_name}"
            
            # Вычисляем хеш конфигурации API
            api_hash = hashlib.md5(json.dumps(api_config, sort_keys=True).encode()).hexdigest()
            
            # Проверяем, есть ли API уже в базе данных
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT id FROM documents WHERE filepath = ?",
                (file_path,)
            )
            existing_doc = cursor.fetchone()
            
            # Сохраняем документ в базе данных
            document_id = self._save_document_to_db(
                file_path=file_path,
                content=parsed_doc.content,
                metadata=metadata,
                chunks=parsed_doc.chunks,
                existing_doc_id=existing_doc[0] if existing_doc else None
            )
            
            # Добавляем документ в векторное хранилище
            if parsed_doc.chunks:
                vector_ids = self._add_to_vector_store(
                    document_id=document_id,
                    chunks=parsed_doc.chunks,
                    metadata=metadata
                )
                
                # Обновляем информацию о векторных ID в базе данных
                self._update_vector_ids(document_id, vector_ids)
            
            # Удаляем временный файл
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Error removing temporary API config file: {e}")
            
            return {
                "api_name": api_name,
                "success": True,
                "document_id": document_id,
                "chunk_count": len(parsed_doc.chunks) if parsed_doc.chunks else 0,
                "status": "updated" if existing_doc else "created",
                "endpoint": api_config.get("endpoint", "")
            }
        except Exception as e:
            logger.error(f"Error indexing API {api_name}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "api_name": api_name,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def remove_document(self, document_id: int) -> Dict[str, Any]:
        """
        Удаляет документ из индекса.
        
        Args:
            document_id: ID документа
            
        Returns:
            Словарь с результатами удаления
        """
        try:
            # Проверяем существование документа
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT filepath FROM documents WHERE id = ?",
                (document_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return {
                    "document_id": document_id,
                    "success": False,
                    "error": "Document not found"
                }
            
            file_path = result[0]
            
            # Получаем ID векторов для удаления из векторного хранилища
            cursor.execute(
                "SELECT vector_id FROM document_chunks WHERE document_id = ?",
                (document_id,)
            )
            vector_ids = [row[0] for row in cursor.fetchall() if row[0]]
            
            # Удаляем документ из базы данных
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_id = ?",
                (document_id,)
            )
            
            cursor.execute(
                "DELETE FROM documents WHERE id = ?",
                (document_id,)
            )
            
            self.db_connection.commit()
            
            # Удаляем векторы из векторного хранилища
            if vector_ids:
                for vector_id in vector_ids:
                    try:
                        self.vector_store.delete(vector_id)
                    except Exception as e:
                        logger.warning(f"Error deleting vector {vector_id}: {e}")
            
            return {
                "document_id": document_id,
                "file_path": file_path,
                "success": True,
                "removed_vectors": len(vector_ids)
            }
        except Exception as e:
            logger.error(f"Error removing document {document_id}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "document_id": document_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def reindex_document(
        self,
        document_id: int,
        parser_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Переиндексирует документ.
        
        Args:
            document_id: ID документа
            parser_options: Опции для парсера
            
        Returns:
            Словарь с результатами переиндексации
        """
        try:
            # Проверяем существование документа
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT filepath, metadata FROM documents WHERE id = ?",
                (document_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return {
                    "document_id": document_id,
                    "success": False,
                    "error": "Document not found"
                }
            
            file_path, metadata_json = result
            
            # Если это API, обрабатываем его специальным образом
            if file_path.startswith("api://"):
                # Извлекаем имя API
                api_name = file_path.replace("api://", "")
                
                # Получаем конфигурацию API из метаданных
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                if "api_config" in metadata:
                    api_config = metadata["api_config"]
                else:
                    return {
                        "document_id": document_id,
                        "success": False,
                        "error": "API configuration not found in metadata"
                    }
                
                # Удаляем документ
                self.remove_document(document_id)
                
                # Индексируем API заново
                return self.index_api(
                    api_config=api_config,
                    api_name=api_name,
                    parser_options=parser_options
                )
            
            # Проверяем существование файла
            if not os.path.exists(file_path):
                return {
                    "document_id": document_id,
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Удаляем документ
            self.remove_document(document_id)
            
            # Индексируем файл заново
            return self.index_file(
                file_path=file_path,
                parser_options=parser_options
            )
        except Exception as e:
            logger.error(f"Error reindexing document {document_id}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "document_id": document_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус индексации.
        
        Returns:
            Словарь с информацией о статусе индексации
        """
        with self.lock:
            return {
                "operation": self.current_operation,
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "failed_files": self.failed_files,
                "progress": self.processed_files / self.total_files if self.total_files > 0 else 0,
                **self.indexing_status
            }
    
    def get_document_metadata(self, document_id: int) -> Optional[Dict[str, Any]]:
        """
        Возвращает метаданные документа.
        
        Args:
            document_id: ID документа
            
        Returns:
            Словарь с метаданными или None, если документ не найден
        """
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT metadata, filepath, filetype, size_bytes, created_at, indexed_at, hash "
                "FROM documents WHERE id = ?",
                (document_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return None
            
            metadata_json, filepath, filetype, size_bytes, created_at, indexed_at, file_hash = result
            
            # Парсим метаданные
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            # Добавляем базовую информацию
            metadata.update({
                "document_id": document_id,
                "filepath": filepath,
                "filetype": filetype,
                "size_bytes": size_bytes,
                "created_at": created_at,
                "indexed_at": indexed_at,
                "hash": file_hash
            })
            
            # Получаем информацию о чанках
            cursor.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id = ?",
                (document_id,)
            )
            chunk_count = cursor.fetchone()[0]
            metadata["chunk_count"] = chunk_count
            
            return metadata
        except Exception as e:
            logger.error(f"Error getting document metadata for {document_id}: {e}")
            return None
    
    def _discover_files(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size_mb: Optional[int] = None,
        max_files: Optional[int] = None
    ) -> List[str]:
        """
        Обнаруживает файлы в директории согласно заданным критериям.
        
        Args:
            directory_path: Путь к директории
            recursive: Искать рекурсивно
            file_patterns: Список паттернов имен файлов для включения
            exclude_patterns: Список паттернов имен файлов для исключения
            max_file_size_mb: Максимальный размер файла в МБ
            max_files: Максимальное количество файлов для возврата
            
        Returns:
            Список путей к файлам
        """
        if max_file_size_mb is None:
            max_file_size_mb = settings.processing.max_file_size_mb
        
        max_size_bytes = max_file_size_mb * 1024 * 1024
        discovered_files = []
        
        # Преобразуем паттерны в регулярные выражения
        include_patterns = None
        exclude_patterns_regex = None
        
        if file_patterns:
            include_patterns = [self._glob_to_regex(pattern) for pattern in file_patterns]
        
        if exclude_patterns:
            exclude_patterns_regex = [self._glob_to_regex(pattern) for pattern in exclude_patterns]
        
        # Получаем все парсеры
        parsers = ParserFactory.get_all_parsers()
        supported_extensions = set()
        
        for parser in parsers:
            supported_extensions.update(parser.supported_extensions)
        
        # Обходим директорию
        for root, _, files in os.walk(directory_path):
            # Если не рекурсивный режим и это не корневая директория, пропускаем
            if not recursive and root != directory_path:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Проверяем, соответствует ли файл паттернам включения
                if include_patterns:
                    if not any(pattern.match(file) for pattern in include_patterns):
                        continue
                
                # Проверяем, соответствует ли файл паттернам исключения
                if exclude_patterns_regex:
                    if any(pattern.match(file) for pattern in exclude_patterns_regex):
                        continue
                
                # Проверяем расширение файла
                _, ext = os.path.splitext(file)
                if ext.lower() not in supported_extensions and not include_patterns:
                    continue
                
                # Проверяем размер файла
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > max_size_bytes:
                        continue
                    
                    # Файл прошел все проверки, добавляем его в список
                    discovered_files.append(file_path)
                    
                    # Если достигли максимального количества файлов, останавливаемся
                    if max_files and len(discovered_files) >= max_files:
                        return discovered_files
                except Exception as e:
                    logger.warning(f"Error checking file {file_path}: {e}")
        
        return discovered_files
    
    def _determine_file_type(self, file_path: str) -> str:
        """
        Определяет тип файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Строка с типом файла
        """
        try:
            # Получаем MIME-тип файла
            mime_type = magic.from_file(file_path, mime=True)
            
            # Получаем расширение файла
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # Для некоторых типов файлов предпочитаем использовать расширение
            if ext in [".docx", ".xlsx", ".pptx"]:
                if ext == ".docx":
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif ext == ".xlsx":
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif ext == ".pptx":
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            
            return mime_type
        except Exception as e:
            logger.warning(f"Error determining file type for {file_path}: {e}")
            
            # Если не удалось определить MIME-тип, используем расширение
            _, ext = os.path.splitext(file_path)
            return f"application/octet-stream{ext}"
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Вычисляет хеш файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Строка с хешем файла
        """
        hash_obj = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Для больших файлов читаем только части
            if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB
                # Читаем первые 5MB
                hash_obj.update(f.read(5 * 1024 * 1024))
                
                # Перемещаемся в конец файла и читаем последние 5MB
                f.seek(-5 * 1024 * 1024, os.SEEK_END)
                hash_obj.update(f.read())
            else:
                # Для небольших файлов читаем всё содержимое
                hash_obj.update(f.read())
        
        return hash_obj.hexdigest()
    
    def _save_document_to_db(
        self,
        file_path: str,
        content: str,
        metadata: Dict[str, Any],
        chunks: List[str],
        existing_doc_id: Optional[int] = None
    ) -> int:
        """
        Сохраняет документ в базе данных.
        
        Args:
            file_path: Путь к файлу
            content: Содержимое документа
            metadata: Метаданные документа
            chunks: Чанки документа
            existing_doc_id: ID существующего документа (если обновляем)
            
        Returns:
            ID документа
        """
        cursor = self.db_connection.cursor()
        
        # Преобразуем метаданные в JSON
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        
        # Получаем базовую информацию
        filename = os.path.basename(file_path)
        filetype = metadata.get("filetype", "unknown")
        size_bytes = metadata.get("size_bytes", 0)
        file_hash = metadata.get("hash", "")
        
        if existing_doc_id:
            # Обновляем существующий документ
            cursor.execute(
                "UPDATE documents SET filename = ?, filepath = ?, filetype = ?, "
                "size_bytes = ?, indexed_at = CURRENT_TIMESTAMP, hash = ?, metadata = ? "
                "WHERE id = ?",
                (filename, file_path, filetype, size_bytes, file_hash, metadata_json, existing_doc_id)
            )
            
            # Удаляем старые чанки
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_id = ?",
                (existing_doc_id,)
            )
            
            document_id = existing_doc_id
        else:
            # Создаем новый документ
            cursor.execute(
                "INSERT INTO documents (filename, filepath, filetype, size_bytes, "
                "created_at, indexed_at, hash, metadata) "
                "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)",
                (filename, file_path, filetype, size_bytes, file_hash, metadata_json)
            )
            
            document_id = cursor.lastrowid
        
        # Сохраняем чанки
        for i, chunk in enumerate(chunks):
            cursor.execute(
                "INSERT INTO document_chunks (document_id, chunk_text, chunk_order, vector_id) "
                "VALUES (?, ?, ?, ?)",
                (document_id, chunk, i, "")
            )
        
        self.db_connection.commit()
        return document_id
    
    def _add_to_vector_store(
        self,
        document_id: int,
        chunks: List[str],
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Добавляет документ в векторное хранилище.
        
        Args:
            document_id: ID документа
            chunks: Чанки документа
            metadata: Метаданные документа
            
        Returns:
            Список ID векторов
        """
        vector_ids = []
        
        # Подготавливаем метаданные для векторного хранилища
        vector_metadata = {
            "document_id": document_id,
            "filename": metadata.get("filename", ""),
            "filepath": metadata.get("filepath", ""),
            "filetype": metadata.get("filetype", "unknown"),
            "content_type": metadata.get("content_type", "text")
        }
        
        # Добавляем дополнительные метаданные
        if "title" in metadata:
            vector_metadata["title"] = metadata["title"]
        
        if "author" in metadata:
            vector_metadata["author"] = metadata["author"]
        
        if "created_at" in metadata:
            vector_metadata["created_at"] = metadata["created_at"]
        
        # Добавляем каждый чанк в векторное хранилище
        for i, chunk in enumerate(chunks):
            chunk_metadata = vector_metadata.copy()
            chunk_metadata["chunk_index"] = i
            
            # Добавляем чанк в векторное хранилище
            vector_id = self.vector_store.add(
                text=chunk,
                metadata=chunk_metadata
            )
            
            vector_ids.append(vector_id)
        
        return vector_ids
    
    def _update_vector_ids(self, document_id: int, vector_ids: List[str]) -> None:
        """
        Обновляет информацию о векторных ID в базе данных.
        
        Args:
            document_id: ID документа
            vector_ids: Список ID векторов
        """
        cursor = self.db_connection.cursor()
        
        # Получаем все чанки документа
        cursor.execute(
            "SELECT id, chunk_order FROM document_chunks WHERE document_id = ? ORDER BY chunk_order",
            (document_id,)
        )
        chunks = cursor.fetchall()
        
        # Обновляем ID векторов
        for (chunk_id, chunk_order), vector_id in zip(chunks, vector_ids):
            cursor.execute(
                "UPDATE document_chunks SET vector_id = ? WHERE id = ?",
                (vector_id, chunk_id)
            )
        
        self.db_connection.commit()
    
    def _glob_to_regex(self, pattern: str) -> re.Pattern:
        """
        Преобразует glob-паттерн в регулярное выражение.
        
        Args:
            pattern: Glob-паттерн
            
        Returns:
            Скомпилированное регулярное выражение
        """
        # Экранируем специальные символы regex
        regex = re.escape(pattern)
        
        # Заменяем glob-специальные символы на соответствующие regex
        regex = regex.replace('\\*', '.*')  # * -> .*
        regex = regex.replace('\\?', '.')   # ? -> .
        regex = regex.replace('\\[', '[')   # [ -> [
        regex = regex.replace('\\]', ']')   # ] -> ]
        
        # Добавляем привязку к началу и концу строки
        regex = f"^{regex}$"
        
        return re.compile(regex)
class FileIndexer(Indexer):
    """
    Класс для индексации файлов и работы с файловой системой.
    Расширяет базовый класс Indexer, добавляя специфичные методы для работы с файлами.
    """
    
    def __init__(self, vector_store=None, db_connection=None):
        """
        Инициализирует индексатор файлов.
        
        Args:
            vector_store: Объект векторного хранилища
            db_connection: Соединение с базой данных
        """
        super().__init__(vector_store, db_connection)
    
    def reindex_all(self):
        """
        Переиндексирует все документы в базе данных.
        
        Returns:
            Dict[str, Any]: Результат переиндексации
        """
        try:
            # Получаем все документы из базы данных
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT id, filepath FROM documents")
            documents = cursor.fetchall()
            
            logger.info(f"Reindexing all documents ({len(documents)} total)")
            
            results = []
            for doc_id, filepath in documents:
                try:
                    result = self.reindex_document(doc_id)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error reindexing document {doc_id}: {e}")
                    logger.error(traceback.format_exc())
                    results.append({
                        "document_id": doc_id,
                        "filepath": filepath,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "status": "completed",
                "total": len(documents),
                "successful": sum(1 for r in results if r.get("success", False)),
                "failed": sum(1 for r in results if not r.get("success", False)),
                "results": results
            }
        except Exception as e:
            logger.error(f"Error in reindex_all: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def reindex_document(self, document_id):
        """
        Переиндексирует указанный документ.
        
        Args:
            document_id: ID документа
            
        Returns:
            Dict[str, Any]: Результат переиндексации
        """
        try:
            # Получаем информацию о документе
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            doc = cursor.fetchone()
            
            if not doc:
                return {
                    "document_id": document_id,
                    "success": False,
                    "error": "Document not found"
                }
            
            # Получаем путь к файлу
            filepath = doc[2]  # Индекс 2 - поле filepath в таблице documents
            
            if not os.path.exists(filepath):
                return {
                    "document_id": document_id,
                    "filepath": filepath,
                    "success": False,
                    "error": "File not found on disk"
                }
            
            # Переиндексируем файл
            result = self.index_file(filepath)
            
            return {
                "document_id": document_id,
                "filepath": filepath,
                "success": result["success"],
                "status": result.get("status", "unknown"),
                "message": result.get("message", ""),
                "error": result.get("error", "")
            }
        except Exception as e:
            logger.error(f"Error reindexing document {document_id}: {e}")
            logger.error(traceback.format_exc())
            return {
                "document_id": document_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }