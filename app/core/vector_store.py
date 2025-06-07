import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from app.core.config import config
from app.core.models import Document, DocumentChunk, ChunkMetadata, SourceDocument

logger = logging.getLogger("smartbot.vector_store")


class VectorStore:
    """
    Класс для работы с векторным хранилищем.
    Использует ChromaDB для хранения и поиска векторных представлений.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.embedding_model = None
        self.initialize_store()

    def initialize_store(self):
        """
        Инициализирует векторное хранилище на основе конфигурации.
        """
        try:
            vector_config = config.settings.vector_store
            
            # Создаем директорию для векторного хранилища, если она не существует
            vector_path = Path(vector_config.path)
            vector_path.mkdir(parents=True, exist_ok=True)
            
            # Инициализируем клиент ChromaDB
            self.client = chromadb.PersistentClient(
                path=vector_config.path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Инициализируем модель эмбеддингов
            model_name = vector_config.embedding_model
            try:
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели эмбеддингов {model_name}: {str(e)}")
                # Используем fallback на локальную функцию эмбеддингов
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            # Получаем или создаем коллекцию
            collection_name = vector_config.collection_name
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Коллекция {collection_name} успешно загружена")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": vector_config.distance_metric}
                )
                logger.info(f"Коллекция {collection_name} успешно создана")
            
            logger.info("Векторное хранилище успешно инициализировано")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации векторного хранилища: {str(e)}")
            raise

    def add_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Добавляет чанки документа в векторное хранилище.
        Возвращает список идентификаторов добавленных чанков.
        """
        if not chunks:
            return []
            
        try:
            # Подготавливаем данные для добавления в ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Генерируем уникальный ID для чанка
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                
                # Добавляем текст чанка
                documents.append(chunk.chunk_text)
                
                # Подготавливаем метаданные
                metadata = {}
                if chunk.metadata:
                    if isinstance(chunk.metadata, ChunkMetadata):
                        metadata = chunk.metadata.dict()
                    else:
                        metadata = chunk.metadata
                else:
                    # Базовые метаданные, если не указаны
                    metadata = {
                        "document_id": chunk.document_id,
                        "chunk_order": chunk.chunk_order
                    }
                
                metadatas.append(metadata)
            
            # Добавляем данные в коллекцию
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Добавлено {len(chunks)} чанков в векторное хранилище")
            
            # Возвращаем ID для обновления в базе данных
            return ids
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении чанков в векторное хранилище: {str(e)}")
            return []

    def search_chunks(
        self, 
        query: str, 
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        use_hybrid_search: bool = True,
        hybrid_alpha: float = 0.5
    ) -> List[SourceDocument]:
        """
        Ищет релевантные чанки в векторном хранилище.
        
        Args:
            query: Текст запроса.
            k: Количество результатов.
            filter_criteria: Критерии фильтрации результатов.
            use_hybrid_search: Использовать гибридный поиск (векторный + полнотекстовый).
            hybrid_alpha: Вес векторного поиска в гибридном поиске (0-1).
            
        Returns:
            Список найденных документов с метаданными.
        """
        try:
            where = filter_criteria if filter_criteria else None
            
            # ChromaDB не поддерживает параметр alpha, поэтому игнорируем use_hybrid_search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            source_documents = []
            
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    try:
                        document_text = results["documents"][0][i]
                        metadata = results["metadatas"][0][i]
                        distance = results["distances"][0][i] if "distances" in results else 0.0
                        
                        # Вычисляем score из distance (ChromaDB возвращает расстояние, а не score)
                        # Для cosine similarity: score = 1 - distance
                        score = 1.0 - distance if distance <= 1.0 else 0.0
                        
                        source_doc = SourceDocument(
                            document_id=metadata.get("document_id", ""),
                            document_name=metadata.get("document_name", ""),
                            document_type=metadata.get("document_type", ""),
                            chunk_text=document_text,
                            score=score,
                            page_number=metadata.get("page_number"),
                            section=metadata.get("section"),
                            url=metadata.get("url")
                        )
                        
                        source_documents.append(source_doc)
                    except Exception as e:
                        logger.error(f"Ошибка при обработке результата поиска {doc_id}: {str(e)}")
            
            logger.info(f"Найдено {len(source_documents)} релевантных чанков для запроса")
            return source_documents
            
        except Exception as e:
            logger.error(f"Ошибка при поиске в векторном хранилище: {str(e)}")
            return []

    def delete_document_chunks(self, document_id: str) -> bool:
        """
        Удаляет все чанки документа из векторного хранилища.
        """
        try:
            # Фильтр по ID документа
            self.collection.delete(
                where={"document_id": document_id}
            )
            
            logger.info(f"Чанки документа {document_id} удалены из векторного хранилища")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при удалении чанков документа {document_id} из векторного хранилища: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получает статистику коллекции.
        """
        try:
            count = self.collection.count()
            
            return {
                "count": count,
                "name": self.collection.name,
                "metadata": self.collection.metadata
            }
            
        except Exception as e:
            logger.error(f"Ошибка при получении статистики коллекции: {str(e)}")
            return {"count": 0, "name": "", "metadata": {}}

    def get_embedding(self, text: str) -> List[float]:
        """
        Получает эмбеддинг для текста.
        """
        try:
            if self.embedding_model:
                return self.embedding_model.encode(text).tolist()
            else:
                # Используем функцию эмбеддинга из ChromaDB
                return self.embedding_function([text])[0]
                
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {str(e)}")
            # Возвращаем пустой вектор
            vector_config = config.settings.vector_store
            return [0.0] * vector_config.embedding_dimension
            
    def reset_collection(self) -> bool:
        """
        Полностью сбрасывает коллекцию.
        """
        try:
            vector_config = config.settings.vector_store
            collection_name = vector_config.collection_name
            
            # Удаляем коллекцию
            self.client.delete_collection(collection_name)
            
            # Создаем новую коллекцию
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": vector_config.distance_metric}
            )
            
            logger.info(f"Коллекция {collection_name} успешно сброшена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сбросе коллекции: {str(e)}")
            return False

    def update_document_chunks(self, document_id: str, new_chunks: List[DocumentChunk]) -> List[str]:
        """
        Обновляет чанки документа в векторном хранилище.
        """
        try:
            # Сначала удаляем существующие чанки
            self.delete_document_chunks(document_id)
            
            # Затем добавляем новые
            return self.add_chunks(new_chunks)
            
        except Exception as e:
            logger.error(f"Ошибка при обновлении чанков документа {document_id}: {str(e)}")
            return []


# Создаем экземпляр векторного хранилища
vector_store = VectorStore()