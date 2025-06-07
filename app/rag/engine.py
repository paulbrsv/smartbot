import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
import re
import traceback
import uuid

from app.core.config import get_settings
from app.core.models import Document, QueryResponse as QueryResult
from app.core.database import get_db_connection
from app.core.vector_store import VectorStore
from app.models.llm_provider import get_llm_provider
from app.rag.multimodal import MultimodalProcessor

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGEngine:
    """
    Основной класс RAG-системы (Retrieval Augmented Generation).
    
    Отвечает за поиск релевантной информации в индексированных данных
    и генерацию ответов на основе найденной информации с использованием
    языковых моделей.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        multimodal_processor: Optional[MultimodalProcessor] = None
    ):
        """
        Инициализирует RAG-систему.
        
        Args:
            vector_store: Объект векторного хранилища
            multimodal_processor: Процессор для мультимодальных запросов
        """
        self.vector_store = vector_store or VectorStore()
        self.db_connection = get_db_connection()
        self.multimodal_processor = multimodal_processor or MultimodalProcessor(vector_store=self.vector_store)
        
        # Устанавливаем значения по умолчанию из конфига
        self.default_search_limit = 5
        self.default_reranking = True
        self.citation_style = "inline"  # "inline", "footnote", "endnote"
    
    def process_query(
        self,
        query_text: str,
        image_data: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        search_options: Optional[Dict[str, Any]] = None,
        llm_settings: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None
    ) -> QueryResult:
        """
        Обрабатывает запрос пользователя и генерирует ответ.
        
        Args:
            query_text: Текст запроса
            image_data: Данные изображений (путь к файлу, bytes или base64)
            search_options: Параметры поиска
                - limit: Максимальное количество результатов (по умолчанию 5)
                - filter_params: Параметры фильтрации
                - reranking: Использовать ли переранжирование (по умолчанию True)
                - hybrid_search: Использовать ли гибридный поиск (по умолчанию True)
                - semantic_weight: Вес семантического поиска (0.0-1.0, по умолчанию 0.7)
            llm_settings: Настройки языковой модели
                - provider: Провайдер модели (openai, anthropic, ollama, google)
                - model: Название модели
                - temperature: Температура генерации
                - max_tokens: Максимальное количество токенов
                - top_p: Параметр top-p
            chat_history: История чата для сохранения контекста диалога
            user_id: Идентификатор пользователя
            
        Returns:
            Результат запроса с ответом и метаданными
        """
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        
        try:
            # Подготавливаем параметры запроса
            search_params = self._prepare_search_params(search_options)
            llm_params = self._prepare_llm_params(llm_settings)
            
            # Проверяем, является ли запрос мультимодальным
            is_multimodal = image_data is not None
            
            # Проверяем, содержит ли запрос ключевые слова, связанные с системными командами
            system_command = self._check_system_command(query_text)
            if system_command:
                # Обрабатываем системную команду
                result = self._process_system_command(
                    command=system_command["command"],
                    parameters=system_command["parameters"],
                    user_id=user_id
                )
                
                # Добавляем метаданные к результату
                # Поле query не существует в QueryResponse, используется query_text
                # result.query = query_text
                result.execution_time = (datetime.now() - start_time).total_seconds()
                result.metadata["query_id"] = query_id
                result.metadata["is_system_command"] = True
                
                # Сохраняем запрос в истории
                self._save_query_to_history(result, user_id)
                
                return result
            
            # Если запрос мультимодальный, обрабатываем его с помощью мультимодального процессора
            if is_multimodal:
                result = self.multimodal_processor.process_query(
                    query_text=query_text,
                    image_data=image_data,
                    chat_history=chat_history,
                    llm_settings=llm_params
                )
                
                # Добавляем метаданные к результату
                result.metadata["query_id"] = query_id
                result.metadata["is_multimodal"] = True
                
                # Сохраняем запрос в истории
                self._save_query_to_history(result, user_id)
                
                return result
            
            # Выполняем поиск по индексированным данным
            search_results = self._search(query_text, search_params)
            
            # Если результаты поиска не найдены, отправляем запрос без контекста
            if not search_results:
                logger.info(f"No search results found for query: {query_text}")
                
                result = self._generate_response_without_context(
                    query_text=query_text,
                    llm_params=llm_params,
                    chat_history=chat_history
                )
                
                # Сохраняем запрос в истории
                self._save_query_to_history(result, user_id)
                
                # Сохраняем неотвеченный запрос для анализа
                self._save_unanswered_query(
                    query_text=query_text,
                    reason="no_results",
                    user_id=user_id
                )
                
                return result
            
            # Генерируем ответ на основе найденных данных
            result = self._generate_response(
                query_text=query_text,
                documents=search_results,
                llm_params=llm_params,
                chat_history=chat_history
            )
            
            # Добавляем информацию об источниках
            result.sources = self._prepare_sources(search_results)
            
            # Добавляем метаданные к результату
            result.metadata["query_id"] = query_id
            result.metadata["result_count"] = len(search_results)
            
            # Сохраняем запрос в истории
            self._save_query_to_history(result, user_id)
            
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            
            # Возвращаем результат с ошибкой
            error_result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                response_text=f"Произошла ошибка при обработке запроса: {str(e)}",
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_name=llm_settings.get("model", ""),
                model_provider=llm_settings.get("provider", "")
            )
            
            # Сохраняем ошибку в истории
            try:
                self._save_query_to_history(error_result, user_id)
            except Exception as save_error:
                logger.error(f"Error saving query to history: {save_error}")
            
            return error_result
    
    def _search(
        self,
        query_text: str,
        search_params: Dict[str, Any]
    ) -> List[Document]:
        """
        Выполняет поиск по индексированным данным.
        
        Args:
            query_text: Текст запроса
            search_params: Параметры поиска
            
        Returns:
            Список найденных документов
        """
        try:
            # Извлекаем параметры поиска
            limit = search_params.get("limit", self.default_search_limit)
            filter_params = search_params.get("filter_params", {})
            reranking = search_params.get("reranking", self.default_reranking)
            hybrid_search = search_params.get("hybrid_search", True)
            semantic_weight = search_params.get("semantic_weight", 0.7)
            
            # Выполняем поиск в векторном хранилище
            if hybrid_search:
                # Гибридный поиск (векторный + полнотекстовый)
                results = self.vector_store.search_chunks(
                    query=query_text,
                    k=limit,
                    filter_criteria=filter_params,
                    use_hybrid_search=True,
                    hybrid_alpha=semantic_weight
                )
            else:
                # Только векторный поиск
                results = self.vector_store.search_chunks(
                    query=query_text,
                    k=limit,
                    filter_criteria=filter_params,
                    use_hybrid_search=False
                )
            
            # Если включено переранжирование, выполняем его
            if reranking and len(results) > 1:
                results = self._rerank_results(query_text, results)
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _rerank_results(
        self,
        query_text: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Переранжирует результаты поиска.
        
        Args:
            query_text: Текст запроса
            documents: Список документов
            
        Returns:
            Переранжированный список документов
        """
        try:
            # Получаем провайдера языковой модели для переранжирования
            llm_provider = get_llm_provider(
                provider=settings.models.default_provider,
                model=getattr(settings.models, settings.models.default_provider).model
            )
            
            # Создаем промпт для переранжирования
            prompt = self._create_reranking_prompt(query_text, documents)
            
            # Получаем ответ модели
            response = llm_provider.generate_response(
                query=prompt,
                context="",
                temperature=0.0  # Используем низкую температуру для детерминированного ответа
            )
            
            # Парсим ответ для получения новых рангов
            reranked_indices = self._parse_reranking_response(response, len(documents))
            
            # Если не удалось получить новый порядок, возвращаем исходный список
            if not reranked_indices:
                return documents
            
            # Переупорядочиваем документы
            reranked_documents = [documents[i] for i in reranked_indices if i < len(documents)]
            
            # Добавляем оставшиеся документы, если не все были переранжированы
            if len(reranked_documents) < len(documents):
                remaining_indices = set(range(len(documents))) - set(reranked_indices)
                for i in remaining_indices:
                    reranked_documents.append(documents[i])
            
            return reranked_documents
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            logger.error(traceback.format_exc())
            return documents
    
    def _create_reranking_prompt(
        self,
        query_text: str,
        documents: List[Document]
    ) -> str:
        """
        Создает промпт для переранжирования.
        
        Args:
            query_text: Текст запроса
            documents: Список документов
            
        Returns:
            Текст промпта
        """
        prompt = f"Задача: Переранжировать результаты поиска на основе их релевантности запросу.\n\n"
        prompt += f"Запрос: {query_text}\n\n"
        prompt += "Результаты поиска:\n\n"
        
        for i, doc in enumerate(documents):
            # Формируем заголовок документа
            title = doc.metadata.get("title", f"Документ {i+1}")
            
            # Извлекаем короткий фрагмент содержимого
            content_snippet = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
            
            prompt += f"Документ {i+1}: {title}\n"
            prompt += f"Содержимое: {content_snippet}\n\n"
        
        prompt += "Оцени релевантность каждого документа запросу на основе его содержания. "
        prompt += "Верни список индексов документов в порядке убывания релевантности. "
        prompt += "Используй следующий формат: 'Ранжирование: [индекс1, индекс2, ...]' "
        prompt += "где индексы начинаются с 0."
        
        return prompt
    
    def _parse_reranking_response(
        self,
        response: str,
        doc_count: int
    ) -> List[int]:
        """
        Парсит ответ модели для получения нового порядка документов.
        
        Args:
            response: Ответ модели
            doc_count: Количество документов
            
        Returns:
            Список индексов в новом порядке
        """
        try:
            # Ищем ранжирование в ответе
            match = re.search(r'Ранжирование:\s*\[([\d\s,]+)\]', response)
            if not match:
                # Пробуем другие форматы
                match = re.search(r'(\[[\d\s,]+\])', response)
            
            if match:
                # Извлекаем список индексов
                indices_str = match.group(1)
                indices = [int(idx.strip()) for idx in re.findall(r'\d+', indices_str)]
                
                # Проверяем валидность индексов
                valid_indices = [idx for idx in indices if 0 <= idx < doc_count]
                
                # Удаляем дубликаты
                unique_indices = []
                for idx in valid_indices:
                    if idx not in unique_indices:
                        unique_indices.append(idx)
                
                return unique_indices
            
            return []
        except Exception as e:
            logger.error(f"Error parsing reranking response: {e}")
            return []
    
    def _generate_response(
        self,
        query_text: str,
        documents: List[Document],
        llm_params: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> QueryResult:
        """
        Генерирует ответ на основе найденных документов.
        
        Args:
            query_text: Текст запроса
            documents: Список документов
            llm_params: Параметры языковой модели
            chat_history: История чата
            
        Returns:
            Результат запроса
        """
        start_time = datetime.now()
        
        try:
            # Получаем провайдера языковой модели
            llm_provider = get_llm_provider(
                provider=llm_params["provider"],
                model=llm_params["model"]
            )
            
            # Подготавливаем контекст на основе найденных документов
            context = self._prepare_context(documents)
            
            # Генерируем ответ
            response = llm_provider.generate_response(
                query=query_text,
                context=context,
                chat_history=chat_history,
                temperature=llm_params.get("temperature", 0.7),
                max_tokens=llm_params.get("max_tokens", 1000),
                top_p=llm_params.get("top_p", 0.95)
            )
            
            # Постобработка ответа (добавление цитат, форматирование)
            processed_response = self._postprocess_response(response, documents)
            
            # Формируем результат
            result = QueryResult(
                query_id=str(uuid.uuid4()),
                query_text=query_text,
                response_text=processed_response,
                sources=[],  # Заполняется в вызывающем методе
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_name=llm_params["model"],
                model_provider=llm_params["provider"]
            )
            
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            
            # Возвращаем ошибку
            return QueryResult(
                query_id=str(uuid.uuid4()),
                query_text=query_text,
                response_text=f"Произошла ошибка при генерации ответа: {str(e)}",
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_name=llm_params.get("model", ""),
                model_provider=llm_params.get("provider", "")
            )
    
    def _generate_response_without_context(
        self,
        query_text: str,
        llm_params: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> QueryResult:
        """
        Генерирует ответ без контекста документов.
        
        Args:
            query_text: Текст запроса
            llm_params: Параметры языковой модели
            chat_history: История чата
            
        Returns:
            Результат запроса
        """
        start_time = datetime.now()
        
        try:
            # Получаем провайдера языковой модели
            llm_provider = get_llm_provider(
                provider=llm_params["provider"],
                model=llm_params["model"]
            )
            
            # Генерируем ответ
            response = llm_provider.generate_response(
                query=query_text,
                context="",
                chat_history=chat_history,
                temperature=llm_params.get("temperature", 0.7),
                max_tokens=llm_params.get("max_tokens", 1000),
                top_p=llm_params.get("top_p", 0.95)
            )
            
            # Формируем результат
            result = QueryResult(
                query_id=str(uuid.uuid4()),
                query_text=query_text,
                response_text=response,
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_name=llm_params["model"],
                model_provider=llm_params["provider"]
            )
            
            return result
        except Exception as e:
            logger.error(f"Error generating response without context: {e}")
            logger.error(traceback.format_exc())
            
            # Возвращаем ошибку
            return QueryResult(
                query_id=str(uuid.uuid4()),
                query_text=query_text,
                response_text=f"Произошла ошибка при генерации ответа: {str(e)}",
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_name=llm_params.get("model", ""),
                model_provider=llm_params.get("provider", "")
            )
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Подготавливает контекст на основе найденных документов.
        
        Args:
            documents: Список документов
            
        Returns:
            Строка с контекстом
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Формируем заголовок документа
            header = f"ДОКУМЕНТ {i+1}"
            
            if "title" in doc.metadata:
                header += f": {doc.metadata['title']}"
            
            # Добавляем информацию об источнике
            source_info = ""
            
            if "source" in doc.metadata:
                source_info += f"Источник: {doc.metadata['source']}\n"
            
            if "filepath" in doc.metadata:
                source_info += f"Файл: {doc.metadata['filepath']}\n"
            
            if "created_at" in doc.metadata:
                source_info += f"Дата создания: {doc.metadata['created_at']}\n"
            
            # Формируем полный контекст документа
            doc_context = f"{header}\n{source_info}\n{doc.content}\n"
            
            context_parts.append(doc_context)
        
        # Объединяем все части контекста
        context = "\n\n".join(context_parts)
        
        return context
    
    def _postprocess_response(
        self,
        response: str,
        documents: List[Document]
    ) -> str:
        """
        Выполняет постобработку ответа.
        
        Args:
            response: Сгенерированный ответ
            documents: Список документов
            
        Returns:
            Обработанный ответ
        """
        # Если указан стиль цитирования, добавляем цитаты
        if self.citation_style == "inline":
            # Добавляем встроенные цитаты
            processed_response = self._add_inline_citations(response, documents)
        elif self.citation_style == "footnote":
            # Добавляем сноски
            processed_response = self._add_footnote_citations(response, documents)
        elif self.citation_style == "endnote":
            # Добавляем концевые сноски
            processed_response = self._add_endnote_citations(response, documents)
        else:
            # Не добавляем цитаты
            processed_response = response
        
        return processed_response
    
    def _add_inline_citations(
        self,
        response: str,
        documents: List[Document]
    ) -> str:
        """
        Добавляет встроенные цитаты к ответу.
        
        Args:
            response: Сгенерированный ответ
            documents: Список документов
            
        Returns:
            Ответ с цитатами
        """
        # Базовая реализация - просто добавляем список источников в конце ответа
        sources_list = []
        
        for i, doc in enumerate(documents):
            # Формируем заголовок источника
            source_title = doc.metadata.get("title", f"Документ {i+1}")
            
            # Добавляем информацию об источнике
            source_info = source_title
            
            if "source" in doc.metadata:
                source_info += f" ({doc.metadata['source']})"
            
            sources_list.append(f"[{i+1}] {source_info}")
        
        # Добавляем список источников в конце ответа
        if sources_list:
            sources_text = "\n\n**Источники:**\n" + "\n".join(sources_list)
            processed_response = response + sources_text
        else:
            processed_response = response
        
        return processed_response
    
    def _add_footnote_citations(
        self,
        response: str,
        documents: List[Document]
    ) -> str:
        """
        Добавляет сноски к ответу.
        
        Args:
            response: Сгенерированный ответ
            documents: Список документов
            
        Returns:
            Ответ со сносками
        """
        # Примечание: это упрощенная реализация для демонстрации
        # В реальном приложении нужно более сложная логика для добавления сносок
        
        # Создаем список сносок
        footnotes = []
        
        for i, doc in enumerate(documents):
            # Формируем заголовок источника
            source_title = doc.metadata.get("title", f"Документ {i+1}")
            
            # Добавляем информацию об источнике
            source_info = source_title
            
            if "source" in doc.metadata:
                source_info += f", {doc.metadata['source']}"
            
            if "created_at" in doc.metadata:
                source_info += f", {doc.metadata['created_at']}"
            
            footnotes.append(f"[{i+1}] {source_info}")
        
        # Добавляем сноски в конце ответа
        if footnotes:
            footnotes_text = "\n\n---\n" + "\n".join(footnotes)
            processed_response = response + footnotes_text
        else:
            processed_response = response
        
        return processed_response
    
    def _add_endnote_citations(
        self,
        response: str,
        documents: List[Document]
    ) -> str:
        """
        Добавляет концевые сноски к ответу.
        
        Args:
            response: Сгенерированный ответ
            documents: Список документов
            
        Returns:
            Ответ с концевыми сносками
        """
        # Реализация аналогична сноскам, но с другим форматированием
        endnotes = []
        
        for i, doc in enumerate(documents):
            # Формируем заголовок источника
            source_title = doc.metadata.get("title", f"Документ {i+1}")
            
            # Добавляем информацию об источнике
            source_info = source_title
            
            if "source" in doc.metadata:
                source_info += f", {doc.metadata['source']}"
            
            if "filepath" in doc.metadata:
                source_info += f", {doc.metadata['filepath']}"
            
            endnotes.append(f"[{i+1}] {source_info}")
        
        # Добавляем концевые сноски в конце ответа
        if endnotes:
            endnotes_text = "\n\n**Примечания:**\n" + "\n".join(endnotes)
            processed_response = response + endnotes_text
        else:
            processed_response = response
        
        return processed_response
    
    def _prepare_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Подготавливает информацию об источниках.
        
        Args:
            documents: Список документов
            
        Returns:
            Список источников
        """
        sources = []
        
        for doc in documents:
            source = {
                "id": doc.id,
                "title": doc.metadata.get("title", "Без названия"),
                "source": doc.metadata.get("source", "Неизвестный источник"),
                "content_type": doc.metadata.get("content_type", "text"),
                "relevance": doc.metadata.get("score", 1.0)
            }
            
            # Добавляем дополнительные метаданные
            if "filepath" in doc.metadata:
                source["filepath"] = doc.metadata["filepath"]
            
            if "filetype" in doc.metadata:
                source["filetype"] = doc.metadata["filetype"]
            
            if "created_at" in doc.metadata:
                source["created_at"] = doc.metadata["created_at"]
            
            sources.append(source)
        
        return sources
    
    def _save_query_to_history(
        self,
        result: QueryResult,
        user_id: Optional[str] = None
    ) -> None:
        """
        Сохраняет запрос в истории.
        
        Args:
            result: Результат запроса
            user_id: Идентификатор пользователя
        """
        try:
            # Проверяем, что соединение с БД активно
            if not self.db_connection:
                logger.warning("Database connection not available, cannot save query to history")
                return
            
            # Подготавливаем данные для сохранения
            query_text = result.query_text
            response_text = result.response_text
            
            # Преобразуем источники в JSON
            sources_json = json.dumps(result.sources, ensure_ascii=False) if result.sources else None
            
            # Создаем метаданные
            metadata = {
                "model_name": result.model_name,
                "model_provider": result.model_provider,
                "processing_time": result.processing_time
            }
            
            if user_id:
                metadata["user_id"] = user_id
            
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            
            # Сохраняем запрос в истории
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT INTO query_history (query_text, response_text, sources, metadata, created_at) "
                "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (query_text, response_text, sources_json, metadata_json)
            )
            
            self.db_connection._sqlite_conn.commit()
            self.db_connection.close()
        except Exception as e:
            logger.error(f"Error saving query to history: {e}")
            logger.error(traceback.format_exc())
    
    def _save_unanswered_query(
        self,
        query_text: str,
        reason: str,
        user_id: Optional[str] = None
    ) -> None:
        """
        Сохраняет неотвеченный запрос для анализа.
        
        Args:
            query_text: Текст запроса
            reason: Причина отсутствия ответа
            user_id: Идентификатор пользователя
        """
        try:
            # Проверяем, что соединение с БД активно
            if not self.db_connection:
                logger.warning("Database connection not available, cannot save unanswered query")
                return
            
            # Подготавливаем метаданные
            metadata = {}
            
            if user_id:
                metadata["user_id"] = user_id
            
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
            
            # Сохраняем запрос
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT INTO unanswered_queries (query_text, reason, created_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)",
                (query_text, reason)
            )
            
            self.db_connection._sqlite_conn.commit()
            self.db_connection.close()
        except Exception as e:
            logger.error(f"Error saving unanswered query: {e}")
            logger.error(traceback.format_exc())
    
    def _check_system_command(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Проверяет, является ли запрос системной командой.
        
        Args:
            query_text: Текст запроса
            
        Returns:
            Словарь с информацией о команде или None, если это не команда
        """
        # Определяем ключевые слова для системных команд
        system_commands = {
            "помощь": "help",
            "справка": "help",
            "команды": "help",
            "настройки": "settings",
            "параметры": "settings",
            "конфиг": "settings",
            "очистить историю": "clear_history",
            "удалить историю": "clear_history",
            "статистика": "stats",
            "статус": "status",
            "информация": "info"
        }
        
        # Проверяем, начинается ли запрос с системной команды
        query_lower = query_text.lower()
        
        for keyword, command in system_commands.items():
            if query_lower.startswith(keyword):
                # Извлекаем параметры команды
                parameters = query_text[len(keyword):].strip()
                
                return {
                    "command": command,
                    "parameters": parameters
                }
        
        return None
    
    def _process_system_command(
        self,
        command: str,
        parameters: str,
        user_id: Optional[str] = None
    ) -> QueryResult:
        """
        Обрабатывает системную команду.
        
        Args:
            command: Команда
            parameters: Параметры команды
            user_id: Идентификатор пользователя
            
        Returns:
            Результат обработки команды
        """
        start_time = datetime.now()
        
        try:
            if command == "help":
                # Команда помощи
                response = self._get_help_text()
            elif command == "settings":
                # Команда настроек
                response = self._get_settings_text()
            elif command == "clear_history":
                # Команда очистки истории
                response = self._clear_history(user_id)
            elif command == "stats":
                # Команда статистики
                response = self._get_stats_text()
            elif command == "status":
                # Команда статуса
                response = self._get_status_text()
            elif command == "info":
                # Команда информации
                response = self._get_info_text()
            else:
                # Неизвестная команда
                response = f"Неизвестная команда: {command}"
            
            # Формируем результат
            result = QueryResult(
                query_id=str(uuid.uuid4()),
                query_text=f"/{command} {parameters}",
                response_text=response,
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_name="system",
                model_provider="internal"
            )
            
            return result
        except Exception as e:
            logger.error(f"Error processing system command: {e}")
            logger.error(traceback.format_exc())
            
            # Возвращаем ошибку
            return QueryResult(
                query_id=str(uuid.uuid4()),
                query_text=f"/{command} {parameters}",
                response_text=f"Произошла ошибка при выполнении команды: {str(e)}",
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_name="system",
                model_provider="internal"
            )
    
    def _get_help_text(self) -> str:
        """
        Возвращает текст справки.
        
        Returns:
            Текст справки
        """
        help_text = """
## Справка по командам SmartBot

### Основные команды:
- **помощь** - показать эту справку
- **настройки** - показать текущие настройки
- **статистика** - показать статистику индексации и запросов
- **статус** - показать текущий статус системы
- **информация** - показать информацию о системе
- **очистить историю** - очистить историю запросов

### Работа с данными:
- Задавайте вопросы в свободной форме, система найдет релевантную информацию в индексированных данных
- Загружайте файлы для анализа (перетащите их в область загрузки)
- Задавайте вопросы по изображениям, прикрепляя их к запросу

### Примеры запросов:
- "Что такое RAG-подход?"
- "Найди информацию о векторных базах данных"
- "Объясни, что показано на этом изображении" (с прикрепленным изображением)
- "Какие форматы файлов поддерживаются для индексации?"

### Справка по API:
Для получения информации о доступных API-эндпоинтах, используйте команду "информация api"
        """
        
        return help_text
    
    def _get_settings_text(self) -> str:
        """
        Возвращает текст с настройками.
        
        Returns:
            Текст с настройками
        """
        # Получаем текущие настройки
        current_settings = settings
        
        settings_text = """
## Текущие настройки SmartBot

### Модели:
- Провайдер по умолчанию: {default_provider}
- Модель: {model}

### Обработка данных:
- Максимальный размер файла: {max_file_size} MB
- Размер чанка: {chunk_size} символов
- Перекрытие чанков: {chunk_overlap} символов
- Поддерживаемые расширения: {supported_extensions}

### Векторное хранилище:
- Тип: {vector_store_type}
- Коллекция: {collection_name}

### Интерфейс:
- Тема: {theme}
- Язык: {language}
- Максимальная история: {max_history} сообщений
        """.format(
            default_provider=current_settings.models.default_provider,
            model=getattr(current_settings.models, current_settings.models.default_provider).model,
            max_file_size=current_settings.processing.max_file_size_mb,
            chunk_size=current_settings.processing.chunk_size,
            chunk_overlap=current_settings.processing.chunk_overlap,
            supported_extensions=", ".join(current_settings.processing.supported_extensions),
            vector_store_type=current_settings.vector_store.type,
            collection_name=current_settings.vector_store.collection_name,
            theme=current_settings.interface.theme,
            language=current_settings.interface.language,
            max_history=current_settings.interface.max_history
        )
        
        return settings_text
    
    def _clear_history(self, user_id: Optional[str] = None) -> str:
        """
        Очищает историю запросов.
        
        Args:
            user_id: Идентификатор пользователя
            
        Returns:
            Текст с результатом очистки
        """
        try:
            # Проверяем, что соединение с БД активно
            if not self.db_connection:
                return "Ошибка: соединение с базой данных недоступно"
            
            cursor = self.db_connection.cursor()
            
            if user_id:
                # Очищаем историю только для указанного пользователя
                cursor.execute(
                    "DELETE FROM query_history WHERE json_extract(metadata, '$.user_id') = ?",
                    (user_id,)
                )
                
                affected_rows = cursor.rowcount
                self.db_connection.commit()
                
                return f"История запросов очищена. Удалено {affected_rows} записей."
            else:
                # Очищаем всю историю
                cursor.execute("DELETE FROM query_history")
                
                affected_rows = cursor.rowcount
                self.db_connection.commit()
                
                return f"Вся история запросов очищена. Удалено {affected_rows} записей."
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            logger.error(traceback.format_exc())
            
            return f"Ошибка при очистке истории: {str(e)}"
    
    def _get_stats_text(self) -> str:
        """
        Возвращает текст со статистикой.
        
        Returns:
            Текст со статистикой
        """
        try:
            # Проверяем, что соединение с БД активно
            if not self.db_connection:
                return "Ошибка: соединение с базой данных недоступно"
            
            cursor = self.db_connection.cursor()
            
            # Получаем статистику по документам
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Получаем статистику по запросам
            cursor.execute("SELECT COUNT(*) FROM query_history")
            query_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM unanswered_queries")
            unanswered_count = cursor.fetchone()[0]
            
            # Получаем статистику по типам файлов
            cursor.execute(
                "SELECT filetype, COUNT(*) FROM documents GROUP BY filetype ORDER BY COUNT(*) DESC"
            )
            filetype_stats = cursor.fetchall()
            
            # Формируем текст статистики
            stats_text = """
## Статистика SmartBot

### Индексированные данные:
- Всего документов: {doc_count}
- Всего чанков: {chunk_count}

### Запросы:
- Всего запросов: {query_count}
- Неотвеченных запросов: {unanswered_count}

### Типы файлов:
{filetype_list}
            """.format(
                doc_count=doc_count,
                chunk_count=chunk_count,
                query_count=query_count,
                unanswered_count=unanswered_count,
                filetype_list="\n".join([f"- {filetype}: {count}" for filetype, count in filetype_stats])
            )
            
            return stats_text
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            logger.error(traceback.format_exc())
            
            return f"Ошибка при получении статистики: {str(e)}"
    
    def _get_status_text(self) -> str:
        """
        Возвращает текст с текущим статусом системы.
        
        Returns:
            Текст с текущим статусом
        """
        try:
            # Получаем информацию о векторном хранилище
            vector_store_info = self.vector_store.get_info()
            
            # Получаем информацию о доступных моделях
            from app.models.llm_provider import get_available_models
            available_models = get_available_models()
            
            # Формируем текст статуса
            status_text = """
## Текущий статус SmartBot

### Векторное хранилище:
- Тип: {vector_store_type}
- Статус: {vector_store_status}
- Количество векторов: {vector_count}

### Доступные модели:
{models_list}

### Системная информация:
- Версия SmartBot: {version}
- Время работы: {uptime}
            """.format(
                vector_store_type=vector_store_info.get("type", "неизвестно"),
                vector_store_status=vector_store_info.get("status", "неизвестно"),
                vector_count=vector_store_info.get("count", 0),
                models_list="\n".join([f"- {provider}: {', '.join(models)}" for provider, models in available_models.items()]),
                version=getattr(settings, "version", "1.0.0"),
                uptime="неизвестно"  # В реальном приложении здесь будет информация о времени работы
            )
            
            return status_text
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            logger.error(traceback.format_exc())
            
            return f"Ошибка при получении статуса: {str(e)}"
    
    def _get_info_text(self) -> str:
        """
        Возвращает информацию о системе.
        
        Returns:
            Текст с информацией
        """
        info_text = """
## SmartBot RAG Assistant

### Описание:
SmartBot RAG Assistant - универсальный помощник, способный индексировать любые типы файлов, понимать вопросы пользователя и предоставлять релевантные ответы на основе проиндексированных данных с использованием RAG-подхода.

### Возможности:
- Индексация различных типов файлов (текстовые документы, структурированные данные, изображения)
- Семантический и гибридный поиск по индексированным данным
- Генерация ответов на основе найденной информации
- Обработка мультимодальных запросов (текст + изображения)
- Поддержка различных языковых моделей (OpenAI, Anthropic, Ollama, Google)

### Поддерживаемые форматы файлов:
- Текстовые документы: TXT, DOCX, DOC, PDF, RTF
- Структурированные данные: JSON, XML, CSV, XLSX
- Медиафайлы: изображения (JPG, PNG, TIFF) с OCR-извлечением текста
- Веб-данные: HTML, MD (Markdown)
- API-данные: REST API endpoints с настраиваемыми парсерами
- Архивы: ZIP, RAR (с автоматической распаковкой)

### Разработчик:
SmartBot RAG Assistant разработан командой Kilo Code.
Версия: {version}
        """.format(
            version=getattr(settings, "version", "1.0.0")
        )
        
        return info_text
    
    def _prepare_search_params(
        self,
        search_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Подготавливает параметры поиска.
        
        Args:
            search_options: Параметры поиска
            
        Returns:
            Словарь с параметрами поиска
        """
        # Параметры по умолчанию
        default_params = {
            "limit": self.default_search_limit,
            "filter_params": {},
            "reranking": self.default_reranking,
            "hybrid_search": True,
            "semantic_weight": 0.7
        }
        
        # Если параметры не указаны, возвращаем значения по умолчанию
        if not search_options:
            return default_params
        
        # Объединяем параметры по умолчанию с указанными параметрами
        params = default_params.copy()
        params.update(search_options)
        
        return params
    
    def _prepare_llm_params(
        self,
        llm_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Подготавливает параметры языковой модели.
        
        Args:
            llm_settings: Настройки языковой модели
            
        Returns:
            Словарь с параметрами языковой модели
        """
        # Параметры по умолчанию
        default_params = {
            "provider": settings.models.default_provider,
            "model": getattr(settings.models, settings.models.default_provider).model_name,
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.95
        }
        
        # Если параметры не указаны, возвращаем значения по умолчанию
        if not llm_settings:
            return default_params
        
        # Объединяем параметры по умолчанию с указанными параметрами
        params = default_params.copy()
        params.update(llm_settings)
        
        return params