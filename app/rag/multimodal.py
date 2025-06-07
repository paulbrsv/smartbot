import logging
import base64
import io
import os
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import traceback
import uuid
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings
from app.core.models import Document, QueryResponse as QueryResult
from app.core.vector_store import VectorStore
from app.models.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)
settings = get_settings()


class MultimodalProcessor:
    """
    Класс для обработки мультимодальных запросов (текст + изображения).
    
    Отвечает за извлечение информации из изображений, их анализ
    и формирование ответов на запросы, содержащие визуальные данные.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        ocr_enabled: bool = True
    ):
        """
        Инициализирует мультимодальный процессор.
        
        Args:
            vector_store: Объект векторного хранилища
            ocr_enabled: Флаг включения OCR
        """
        self.vector_store = vector_store or VectorStore()
        self.ocr_enabled = ocr_enabled
        
        # Создаем временную директорию для изображений
        self.temp_dir = os.path.join(os.path.expanduser("~"), ".smartbot", "temp", "images")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Настройки OCR из конфигурации
        self.ocr_language = settings.ocr.language if hasattr(settings, 'ocr') and hasattr(settings.ocr, 'language') else "rus+eng"
        self.ocr_dpi = settings.ocr.dpi if hasattr(settings, 'ocr') and hasattr(settings.ocr, 'dpi') else 300
        
        # Настройки для обработки изображений
        self.image_size_limit = 10 * 1024 * 1024  # 10 МБ
        self.supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        
        logger.info("Initialized multimodal processor")
    
    def process_query(
        self,
        query_text: str,
        image_data: Union[str, bytes, List[Union[str, bytes]]],
        llm_settings: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> QueryResult:
        """
        Обрабатывает мультимодальный запрос (текст + изображения).
        
        Args:
            query_text: Текст запроса
            image_data: Данные изображений (путь к файлу, bytes или base64)
            llm_settings: Настройки языковой модели
            chat_history: История чата
            
        Returns:
            Результат запроса
        """
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        
        try:
            # Подготавливаем данные изображения
            processed_images = self._process_image_data(image_data)
            
            if not processed_images:
                # Если не удалось обработать изображения, возвращаем ошибку
                return QueryResult(
                    query=query_text,
                    response="Не удалось обработать предоставленные изображения. Пожалуйста, убедитесь, что изображения в поддерживаемом формате и попробуйте снова.",
                    sources=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={
                        "error": "Failed to process images",
                        "query_id": query_id
                    }
                )
            
            # Подготавливаем настройки модели
            if not llm_settings:
                llm_settings = {
                    "provider": settings.models.default_provider,
                    "model": getattr(settings.models, settings.models.default_provider).model,
                    "temperature": 0.7
                }
            
            # Проверяем поддерживает ли выбранная модель мультимодальные запросы
            multimodal_provider = self._get_multimodal_provider(llm_settings)
            
            if not multimodal_provider:
                # Если модель не поддерживает мультимодальные запросы,
                # используем OCR и обрабатываем запрос обычным способом
                return self._process_via_ocr(
                    query_text=query_text,
                    processed_images=processed_images,
                    llm_settings=llm_settings,
                    chat_history=chat_history,
                    query_id=query_id,
                    start_time=start_time
                )
            
            # Подготавливаем изображения для мультимодальной модели
            image_inputs = []
            for img_info in processed_images:
                # В зависимости от формата, который ожидает модель
                if img_info["format"] == "path":
                    image_inputs.append(img_info["data"])
                elif img_info["format"] == "pil":
                    image_inputs.append(img_info["pil_image"])
                elif img_info["format"] == "base64":
                    image_inputs.append(img_info["base64"])
                else:
                    # Используем данные изображения напрямую
                    image_inputs.append(img_info["data"])
            
            # Генерируем ответ с помощью мультимодальной модели
            response = multimodal_provider.generate_multimodal_response(
                query=query_text,
                images=image_inputs,
                chat_history=chat_history
            )
            
            # Параллельно извлекаем текст из изображений через OCR
            ocr_texts = []
            if self.ocr_enabled:
                for img_info in processed_images:
                    try:
                        ocr_text = self._extract_text_from_image(img_info["pil_image"])
                        if ocr_text:
                            ocr_texts.append({
                                "image_id": img_info["id"],
                                "text": ocr_text
                            })
                    except Exception as e:
                        logger.warning(f"OCR extraction failed for image {img_info['id']}: {e}")
            
            # Формируем результат
            result = QueryResult(
                query=query_text,
                response=response,
                sources=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "query_id": query_id,
                    "is_multimodal": True,
                    "image_count": len(processed_images),
                    "images": [
                        {
                            "id": img["id"],
                            "filename": img.get("filename", ""),
                            "size": img.get("size", 0),
                            "format": img.get("mime_type", "unknown")
                        } for img in processed_images
                    ]
                }
            )
            
            # Добавляем данные OCR в метаданные, если они есть
            if ocr_texts:
                result.metadata["ocr_data"] = ocr_texts
            
            return result
        except Exception as e:
            logger.error(f"Error processing multimodal query: {e}")
            logger.error(traceback.format_exc())
            
            # Возвращаем ошибку
            return QueryResult(
                query=query_text,
                response=f"Произошла ошибка при обработке мультимодального запроса: {str(e)}",
                sources=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "error": str(e),
                    "query_id": query_id
                }
            )
    
    def index_image(
        self,
        image_data: Union[str, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        extract_text: bool = True,
        custom_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Индексирует изображение для последующего поиска.
        
        Args:
            image_data: Данные изображения (путь к файлу, bytes или base64)
            metadata: Дополнительные метаданные
            extract_text: Извлекать ли текст из изображения
            custom_id: Пользовательский ID изображения
            
        Returns:
            Результат индексации
        """
        try:
            # Подготавливаем данные изображения
            processed_images = self._process_image_data(image_data)
            
            if not processed_images:
                return {
                    "success": False,
                    "error": "Failed to process image"
                }
            
            # Берем первое изображение (в случае, если передан список)
            img_info = processed_images[0]
            
            # Генерируем ID изображения
            image_id = custom_id or str(uuid.uuid4())
            
            # Подготавливаем метаданные
            if not metadata:
                metadata = {}
            
            metadata.update({
                "image_id": image_id,
                "content_type": "image",
                "mime_type": img_info.get("mime_type", "image/unknown"),
                "size": img_info.get("size", 0),
                "filename": img_info.get("filename", ""),
                "indexed_at": datetime.now().isoformat()
            })
            
            # Извлекаем текст из изображения, если включено
            if extract_text and self.ocr_enabled:
                try:
                    ocr_text = self._extract_text_from_image(img_info["pil_image"])
                    
                    if ocr_text:
                        metadata["ocr_text"] = ocr_text
                except Exception as e:
                    logger.warning(f"OCR extraction failed for image {image_id}: {e}")
            
            # Получаем мультимодальный провайдер для извлечения эмбеддингов
            multimodal_provider = self._get_multimodal_provider({
                "provider": settings.models.default_provider,
                "model": getattr(settings.models, settings.models.default_provider).model
            })
            
            if not multimodal_provider:
                # Если нет доступного мультимодального провайдера, 
                # и нет OCR-текста, не можем индексировать изображение
                if not metadata.get("ocr_text"):
                    return {
                        "success": False,
                        "error": "No multimodal provider available and no OCR text extracted"
                    }
            else:
                # Получаем описание изображения
                try:
                    image_description = multimodal_provider.describe_image(
                        image=img_info["pil_image"]
                    )
                    
                    if image_description:
                        metadata["image_description"] = image_description
                except Exception as e:
                    logger.warning(f"Image description failed for image {image_id}: {e}")
            
            # Определяем текст для индексации
            if "image_description" in metadata:
                index_text = metadata["image_description"]
                
                if "ocr_text" in metadata:
                    index_text += "\n\n" + metadata["ocr_text"]
            elif "ocr_text" in metadata:
                index_text = metadata["ocr_text"]
            else:
                # Если нет ни описания, ни OCR-текста, используем только метаданные
                index_text = f"Изображение: {metadata.get('filename', 'unnamed')}"
            
            # Сохраняем изображение во временную директорию
            img_path = os.path.join(self.temp_dir, f"{image_id}.jpg")
            img_info["pil_image"].save(img_path, format="JPEG")
            metadata["image_path"] = img_path
            
            # Добавляем в векторное хранилище
            vector_id = self.vector_store.add(
                text=index_text,
                metadata=metadata
            )
            
            return {
                "success": True,
                "image_id": image_id,
                "vector_id": vector_id,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error indexing image: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_similar_images(
        self,
        query_image: Union[str, bytes],
        limit: int = 5,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ищет похожие изображения.
        
        Args:
            query_image: Изображение для поиска
            limit: Максимальное количество результатов
            filter_params: Параметры фильтрации
            
        Returns:
            Список найденных изображений
        """
        try:
            # Подготавливаем данные изображения
            processed_images = self._process_image_data(query_image)
            
            if not processed_images:
                return []
            
            # Берем первое изображение
            img_info = processed_images[0]
            
            # Получаем мультимодальный провайдер
            multimodal_provider = self._get_multimodal_provider({
                "provider": settings.models.default_provider,
                "model": getattr(settings.models, settings.models.default_provider).model
            })
            
            # Если нет мультимодального провайдера, используем OCR
            if not multimodal_provider:
                # Извлекаем текст из изображения
                if self.ocr_enabled:
                    ocr_text = self._extract_text_from_image(img_info["pil_image"])
                    
                    if not ocr_text:
                        return []
                    
                    # Ищем похожие документы по тексту
                    search_params = {
                        "filter_params": {
                            "content_type": "image"
                        }
                    }
                    
                    if filter_params:
                        search_params["filter_params"].update(filter_params)
                    
                    results = self.vector_store.search(
                        query=ocr_text,
                        limit=limit,
                        filter_params=search_params["filter_params"]
                    )
                    
                    # Преобразуем результаты
                    return [{
                        "id": doc.id,
                        "image_id": doc.metadata.get("image_id", "unknown"),
                        "image_path": doc.metadata.get("image_path", ""),
                        "score": doc.metadata.get("score", 0.0),
                        "metadata": doc.metadata
                    } for doc in results]
                else:
                    return []
            
            # Получаем описание изображения
            image_description = multimodal_provider.describe_image(
                image=img_info["pil_image"]
            )
            
            if not image_description:
                return []
            
            # Ищем похожие документы по описанию
            search_params = {
                "filter_params": {
                    "content_type": "image"
                }
            }
            
            if filter_params:
                search_params["filter_params"].update(filter_params)
            
            results = self.vector_store.search(
                query=image_description,
                limit=limit,
                filter_params=search_params["filter_params"]
            )
            
            # Преобразуем результаты
            return [{
                "id": doc.id,
                "image_id": doc.metadata.get("image_id", "unknown"),
                "image_path": doc.metadata.get("image_path", ""),
                "score": doc.metadata.get("score", 0.0),
                "metadata": doc.metadata
            } for doc in results]
        except Exception as e:
            logger.error(f"Error searching similar images: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def detect_text_in_image(
        self,
        image_data: Union[str, bytes]
    ) -> Dict[str, Any]:
        """
        Обнаруживает текст на изображении с помощью OCR.
        
        Args:
            image_data: Данные изображения
            
        Returns:
            Словарь с обнаруженным текстом и метаданными
        """
        try:
            # Проверяем, включен ли OCR
            if not self.ocr_enabled:
                return {
                    "success": False,
                    "error": "OCR is disabled"
                }
            
            # Подготавливаем данные изображения
            processed_images = self._process_image_data(image_data)
            
            if not processed_images:
                return {
                    "success": False,
                    "error": "Failed to process image"
                }
            
            # Берем первое изображение
            img_info = processed_images[0]
            
            # Извлекаем текст
            ocr_text = self._extract_text_from_image(img_info["pil_image"])
            
            # Если текст не найден
            if not ocr_text:
                return {
                    "success": True,
                    "text": "",
                    "text_found": False,
                    "message": "No text found in the image"
                }
            
            # Возвращаем результат
            return {
                "success": True,
                "text": ocr_text,
                "text_found": True,
                "language": self.ocr_language,
                "confidence": 0.0  # В pytesseract сложно получить общую оценку уверенности
            }
        except Exception as e:
            logger.error(f"Error detecting text in image: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def detect_objects_in_image(
        self,
        image_data: Union[str, bytes]
    ) -> Dict[str, Any]:
        """
        Обнаруживает объекты на изображении.
        
        Args:
            image_data: Данные изображения
            
        Returns:
            Словарь с обнаруженными объектами и метаданными
        """
        try:
            # Подготавливаем данные изображения
            processed_images = self._process_image_data(image_data)
            
            if not processed_images:
                return {
                    "success": False,
                    "error": "Failed to process image"
                }
            
            # Берем первое изображение
            img_info = processed_images[0]
            
            # Получаем мультимодальный провайдер
            multimodal_provider = self._get_multimodal_provider({
                "provider": settings.models.default_provider,
                "model": getattr(settings.models, settings.models.default_provider).model
            })
            
            if not multimodal_provider or not hasattr(multimodal_provider, "detect_objects"):
                return {
                    "success": False,
                    "error": "Object detection not supported by the current multimodal provider"
                }
            
            # Обнаруживаем объекты
            objects = multimodal_provider.detect_objects(
                image=img_info["pil_image"]
            )
            
            # Возвращаем результат
            return {
                "success": True,
                "objects": objects,
                "count": len(objects)
            }
        except Exception as e:
            logger.error(f"Error detecting objects in image: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _process_via_ocr(
        self,
        query_text: str,
        processed_images: List[Dict[str, Any]],
        llm_settings: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]],
        query_id: str,
        start_time: datetime
    ) -> QueryResult:
        """
        Обрабатывает мультимодальный запрос через OCR и обычную языковую модель.
        
        Args:
            query_text: Текст запроса
            processed_images: Обработанные изображения
            llm_settings: Настройки языковой модели
            chat_history: История чата
            query_id: ID запроса
            start_time: Время начала обработки
            
        Returns:
            Результат запроса
        """
        # Извлекаем текст из изображений
        ocr_texts = []
        
        for img_info in processed_images:
            try:
                ocr_text = self._extract_text_from_image(img_info["pil_image"])
                
                if ocr_text:
                    ocr_texts.append({
                        "image_id": img_info["id"],
                        "text": ocr_text
                    })
            except Exception as e:
                logger.warning(f"OCR extraction failed for image {img_info['id']}: {e}")
        
        # Если не удалось извлечь текст ни из одного изображения
        if not ocr_texts:
            return QueryResult(
                query=query_text,
                response="Не удалось извлечь текст из предоставленных изображений. Пожалуйста, убедитесь, что изображения содержат текст или используйте модель с поддержкой мультимодальных запросов.",
                sources=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "error": "No text extracted from images",
                    "query_id": query_id
                }
            )
        
        # Формируем контекст на основе извлеченного текста
        context = "Информация, извлеченная из изображений:\n\n"
        
        for i, ocr_item in enumerate(ocr_texts):
            context += f"Изображение {i+1}:\n{ocr_item['text']}\n\n"
        
        # Получаем провайдера языковой модели
        llm_provider = get_llm_provider(
            provider=llm_settings["provider"],
            model=llm_settings["model"]
        )
        
        # Формируем запрос, объединяя исходный запрос и контекст
        prompt = f"{query_text}\n\n{context}"
        
        # Генерируем ответ
        response = llm_provider.generate_response(
            query=prompt,
            context="",
            chat_history=chat_history,
            temperature=llm_settings.get("temperature", 0.7),
            max_tokens=llm_settings.get("max_tokens", 1000),
            top_p=llm_settings.get("top_p", 0.95)
        )
        
        # Формируем результат
        result = QueryResult(
            query=query_text,
            response=response,
            sources=[],
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "query_id": query_id,
                "is_multimodal": True,
                "ocr_only": True,
                "image_count": len(processed_images),
                "images": [
                    {
                        "id": img["id"],
                        "filename": img.get("filename", ""),
                        "size": img.get("size", 0),
                        "format": img.get("mime_type", "unknown")
                    } for img in processed_images
                ],
                "ocr_data": ocr_texts
            }
        )
        
        return result
    
    def _process_image_data(
        self,
        image_data: Union[str, bytes, List[Union[str, bytes]]]
    ) -> List[Dict[str, Any]]:
        """
        Обрабатывает данные изображения и преобразует их в стандартный формат.
        
        Args:
            image_data: Данные изображения (путь, bytes, base64 или список)
            
        Returns:
            Список обработанных изображений
        """
        # Преобразуем одиночное изображение в список
        if not isinstance(image_data, list):
            image_data = [image_data]
        
        processed_images = []
        
        for idx, img in enumerate(image_data):
            try:
                img_info = {
                    "id": str(uuid.uuid4()),
                    "index": idx
                }
                
                # Обрабатываем изображение в зависимости от типа
                if isinstance(img, str):
                    # Проверяем, является ли строка путем к файлу
                    if os.path.exists(img):
                        # Путь к файлу
                        img_info["format"] = "path"
                        img_info["data"] = img
                        img_info["filename"] = os.path.basename(img)
                        
                        # Открываем изображение
                        pil_image = Image.open(img)
                        img_info["pil_image"] = pil_image
                        img_info["size"] = os.path.getsize(img)
                        img_info["mime_type"] = self._get_image_mime_type(pil_image)
                        
                        # Кодируем в base64
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format=pil_image.format or "JPEG")
                        img_buffer.seek(0)
                        img_info["base64"] = base64.b64encode(img_buffer.read()).decode("utf-8")
                    else:
                        # Предполагаем, что это base64
                        try:
                            # Декодируем base64
                            img_bytes = base64.b64decode(img)
                            img_buffer = io.BytesIO(img_bytes)
                            
                            # Открываем изображение
                            pil_image = Image.open(img_buffer)
                            img_info["pil_image"] = pil_image
                            img_info["format"] = "base64"
                            img_info["data"] = img
                            img_info["size"] = len(img_bytes)
                            img_info["mime_type"] = self._get_image_mime_type(pil_image)
                            img_info["base64"] = img
                        except Exception as e:
                            logger.warning(f"Invalid base64 string: {e}")
                            continue
                elif isinstance(img, bytes):
                    # Байты изображения
                    try:
                        img_buffer = io.BytesIO(img)
                        
                        # Открываем изображение
                        pil_image = Image.open(img_buffer)
                        img_info["pil_image"] = pil_image
                        img_info["format"] = "bytes"
                        img_info["data"] = img
                        img_info["size"] = len(img)
                        img_info["mime_type"] = self._get_image_mime_type(pil_image)
                        
                        # Кодируем в base64
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format=pil_image.format or "JPEG")
                        img_buffer.seek(0)
                        img_info["base64"] = base64.b64encode(img_buffer.read()).decode("utf-8")
                    except Exception as e:
                        logger.warning(f"Invalid image bytes: {e}")
                        continue
                elif isinstance(img, PIL.Image.Image):
                    # PIL Image
                    img_info["pil_image"] = img
                    img_info["format"] = "pil"
                    img_info["data"] = img
                    img_info["mime_type"] = self._get_image_mime_type(img)
                    
                    # Кодируем в base64
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format=img.format or "JPEG")
                    img_buffer.seek(0)
                    img_info["base64"] = base64.b64encode(img_buffer.read()).decode("utf-8")
                    img_info["size"] = img_buffer.getbuffer().nbytes
                else:
                    logger.warning(f"Unsupported image format: {type(img)}")
                    continue
                
                # Проверяем размер изображения
                if img_info.get("size", 0) > self.image_size_limit:
                    logger.warning(f"Image size exceeds limit: {img_info.get('size')} > {self.image_size_limit}")
                    continue
                
                # Добавляем обработанное изображение в список
                processed_images.append(img_info)
            except Exception as e:
                logger.warning(f"Error processing image {idx}: {e}")
                continue
        
        return processed_images
    
    def _extract_text_from_image(self, image: PIL.Image.Image) -> str:
        """
        Извлекает текст из изображения с помощью OCR.
        
        Args:
            image: Объект PIL Image
            
        Returns:
            Извлеченный текст
        """
        try:
            # Преобразуем изображение в RGB, если оно в другом режиме
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Извлекаем текст с помощью pytesseract
            ocr_text = pytesseract.image_to_string(
                image,
                lang=self.ocr_language,
                config=f"--dpi {self.ocr_dpi} --oem 1 --psm 3"
            )
            
            # Очищаем текст
            ocr_text = ocr_text.strip()
            
            return ocr_text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            logger.error(traceback.format_exc())
            return ""
    
    def _get_image_mime_type(self, image: PIL.Image.Image) -> str:
        """
        Определяет MIME-тип изображения.
        
        Args:
            image: Объект PIL Image
            
        Returns:
            MIME-тип изображения
        """
        format_map = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "GIF": "image/gif",
            "BMP": "image/bmp",
            "TIFF": "image/tiff",
            "WEBP": "image/webp"
        }
        
        # Получаем формат изображения
        image_format = getattr(image, "format", None)
        
        if image_format and image_format in format_map:
            return format_map[image_format]
        
        return "image/unknown"
    
    def _get_multimodal_provider(self, llm_settings: Dict[str, Any]) -> Any:
        """
        Получает провайдера мультимодальной модели.
        
        Args:
            llm_settings: Настройки языковой модели
            
        Returns:
            Провайдер мультимодальной модели или None
        """
        try:
            # Проверяем, поддерживает ли провайдер мультимодальные запросы
            provider = llm_settings.get("provider", settings.models.default_provider)
            model = llm_settings.get("model", getattr(settings.models, provider).model)
            
            # Получаем мультимодальный провайдер
            return get_multimodal_provider(provider=provider, model=model)
        except Exception as e:
            logger.warning(f"Failed to get multimodal provider: {e}")
            return None