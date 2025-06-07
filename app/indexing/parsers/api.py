import os
import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import requests
import time
import hashlib
from urllib.parse import urlparse
import tempfile
import jsonpath_ng.ext as jsonpath
import traceback

from app.indexing.parsers.base import BaseParser, ParsedDocument
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class APIParser(BaseParser):
    """
    Парсер для данных, получаемых через API.
    Позволяет индексировать данные из REST API endpoints 
    с настраиваемыми парсерами на основе JSON-конфигов.
    """
    
    def __init__(self):
        """Инициализация парсера API-данных."""
        super().__init__()
        self.supported_extensions = [".api", ".apiconfig", ".apijson"]
        self._session = requests.Session()
        self._cache_dir = os.path.join(tempfile.gettempdir(), "smartbot_api_cache")
        
        # Создаем директорию для кеша, если её нет
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает API-конфиг и получает данные из API.
        
        Args:
            file_path: Путь к файлу конфигурации API
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - api_config: Словарь с конфигурацией API (если не передан, читается из file_path)
                - use_cache: Использовать ли кеширование (по умолчанию True)
                - cache_ttl: Время жизни кеша в секундах (по умолчанию 3600)
                - max_retries: Максимальное количество попыток запроса (по умолчанию 3)
                - retry_delay: Задержка между попытками в секундах (по умолчанию 1)
                - timeout: Таймаут запроса в секундах (по умолчанию 30)
                - auth_token: Токен авторизации (перезаписывает значение из конфига)
                - additional_headers: Дополнительные заголовки запроса (добавляются к заголовкам из конфига)
                - additional_params: Дополнительные параметры запроса (добавляются к параметрам из конфига)
                - pagination: Настройки пагинации (перезаписывает значение из конфига)
                - request_rate_limit: Ограничение скорости запросов (запросов в секунду, по умолчанию 0 - без ограничений)
            
        Returns:
            Объект ParsedDocument с содержимым и метаданными
        """
        try:
            # Определяем параметры
            chunk_size = kwargs.get("chunk_size", settings.processing.chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.processing.chunk_overlap)
            api_config = kwargs.get("api_config", None)
            use_cache = kwargs.get("use_cache", True)
            cache_ttl = kwargs.get("cache_ttl", 3600)  # 1 час
            max_retries = kwargs.get("max_retries", 3)
            retry_delay = kwargs.get("retry_delay", 1)
            timeout = kwargs.get("timeout", 30)
            auth_token = kwargs.get("auth_token", None)
            additional_headers = kwargs.get("additional_headers", {})
            additional_params = kwargs.get("additional_params", {})
            pagination = kwargs.get("pagination", None)
            request_rate_limit = kwargs.get("request_rate_limit", 0)
            
            # Если конфигурация не передана, читаем её из файла
            if not api_config:
                with open(file_path, 'r', encoding='utf-8') as f:
                    api_config = json.load(f)
            
            # Получаем метаданные файла и API
            metadata = {}
            if os.path.exists(file_path):
                metadata = self.get_file_metadata(file_path)
            
            # Добавляем базовую информацию о API
            api_metadata = {
                "api_name": api_config.get("api_name", "unknown"),
                "endpoint": api_config.get("endpoint", ""),
                "method": api_config.get("method", "GET"),
                "content_type": "application/json"
            }
            metadata.update(api_metadata)
            
            # Если передан токен авторизации, используем его
            if auth_token:
                # Проверяем, есть ли заголовки в конфиге
                if "headers" not in api_config:
                    api_config["headers"] = {}
                
                # Определяем формат токена в зависимости от заголовков
                if "Authorization" in api_config["headers"]:
                    auth_header = api_config["headers"]["Authorization"]
                    if auth_header.startswith("Bearer "):
                        api_config["headers"]["Authorization"] = f"Bearer {auth_token}"
                    elif auth_header.startswith("Token "):
                        api_config["headers"]["Authorization"] = f"Token {auth_token}"
                    else:
                        api_config["headers"]["Authorization"] = auth_token
                else:
                    # По умолчанию используем Bearer
                    api_config["headers"]["Authorization"] = f"Bearer {auth_token}"
            
            # Добавляем дополнительные заголовки
            if "headers" not in api_config:
                api_config["headers"] = {}
            
            api_config["headers"].update(additional_headers)
            
            # Добавляем дополнительные параметры запроса
            if "params" not in api_config:
                api_config["params"] = {}
            
            api_config["params"].update(additional_params)
            
            # Если передан объект пагинации, используем его
            if pagination:
                api_config["pagination"] = pagination
            
            # Получаем данные из API
            api_response = self._fetch_api_data(
                api_config,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                request_rate_limit=request_rate_limit
            )
            
            # Парсим полученные данные
            content, parsed_items = self._parse_api_response(api_response, api_config)
            
            # Добавляем информацию о количестве полученных элементов
            metadata["item_count"] = len(parsed_items)
            metadata["response_size"] = len(json.dumps(api_response))
            metadata["parsed_items"] = parsed_items
            
            # Разбиваем содержимое на чанки
            chunks = self.chunk_text(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                chunks=chunks
            )
        except Exception as e:
            logger.error(f"Error parsing API data from {file_path}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _fetch_api_data(
        self,
        api_config: Dict[str, Any],
        use_cache: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        retry_delay: int = 1,
        timeout: int = 30,
        request_rate_limit: float = 0
    ) -> Any:
        """
        Получает данные из API с учетом кеширования и пагинации.
        
        Args:
            api_config: Конфигурация API
            use_cache: Использовать ли кеширование
            cache_ttl: Время жизни кеша в секундах
            max_retries: Максимальное количество попыток запроса
            retry_delay: Задержка между попытками в секундах
            timeout: Таймаут запроса в секундах
            request_rate_limit: Ограничение скорости запросов (запросов в секунду)
            
        Returns:
            Полученные данные
        """
        # Проверяем, есть ли обязательные параметры
        if "endpoint" not in api_config:
            raise ValueError("API endpoint not specified in configuration")
        
        # Получаем параметры запроса
        endpoint = api_config["endpoint"]
        method = api_config.get("method", "GET").upper()
        headers = api_config.get("headers", {})
        params = api_config.get("params", {})
        body = api_config.get("body", None)
        auth = api_config.get("auth", None)
        pagination = api_config.get("pagination", None)
        
        # Генерируем ключ кеша
        cache_key = self._generate_cache_key(api_config)
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.json")
        
        # Проверяем, есть ли данные в кеше
        if use_cache and os.path.exists(cache_file):
            # Проверяем время создания кеша
            cache_time = os.path.getmtime(cache_file)
            current_time = time.time()
            
            # Если кеш не устарел, возвращаем данные из него
            if current_time - cache_time < cache_ttl:
                logger.info(f"Using cached data for {endpoint}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # Если нет пагинации, делаем одиночный запрос
        if not pagination:
            response_data = self._make_request(
                method=method,
                url=endpoint,
                headers=headers,
                params=params,
                body=body,
                auth=auth,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout
            )
            
            # Сохраняем данные в кеш
            if use_cache:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=2)
            
            return response_data
        
        # Если есть пагинация, выполняем пагинированные запросы
        paginated_data = self._fetch_paginated_data(
            method=method,
            url=endpoint,
            headers=headers,
            params=params,
            body=body,
            auth=auth,
            pagination=pagination,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            request_rate_limit=request_rate_limit
        )
        
        # Сохраняем данные в кеш
        if use_cache:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(paginated_data, f, ensure_ascii=False, indent=2)
        
        return paginated_data
    
    def _make_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        params: Dict[str, Any] = None,
        body: Any = None,
        auth: Optional[tuple] = None,
        max_retries: int = 3,
        retry_delay: int = 1,
        timeout: int = 30
    ) -> Any:
        """
        Выполняет HTTP-запрос с повторными попытками.
        
        Args:
            method: HTTP-метод (GET, POST, PUT, DELETE)
            url: URL-адрес
            headers: Заголовки запроса
            params: Параметры запроса
            body: Тело запроса
            auth: Аутентификация (tuple username, password)
            max_retries: Максимальное количество попыток
            retry_delay: Задержка между попытками в секундах
            timeout: Таймаут запроса в секундах
            
        Returns:
            Данные ответа
        """
        # Подготавливаем аргументы запроса
        request_kwargs = {
            "url": url,
            "headers": headers or {},
            "params": params or {},
            "timeout": timeout
        }
        
        # Добавляем тело запроса для методов, которые его поддерживают
        if method in ["POST", "PUT", "PATCH"] and body:
            # Если тело - словарь или список, преобразуем его в JSON
            if isinstance(body, (dict, list)):
                request_kwargs["json"] = body
                # Добавляем заголовок Content-Type, если он не указан
                if "Content-Type" not in request_kwargs["headers"]:
                    request_kwargs["headers"]["Content-Type"] = "application/json"
            else:
                request_kwargs["data"] = body
        
        # Добавляем аутентификацию, если она указана
        if auth:
            request_kwargs["auth"] = auth
        
        # Выполняем запрос с повторными попытками
        for attempt in range(max_retries):
            try:
                response = self._session.request(method=method, **request_kwargs)
                
                # Проверяем статус ответа
                response.raise_for_status()
                
                # Если ответ успешный, возвращаем данные
                try:
                    # Пытаемся разобрать JSON
                    return response.json()
                except ValueError:
                    # Если не получилось, возвращаем текст
                    return response.text
            except requests.exceptions.HTTPError as e:
                # Проверяем, стоит ли повторять запрос
                if response.status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    # Экспоненциальная задержка с дрожанием
                    sleep_time = retry_delay * (2 ** attempt) + (time.random() * retry_delay)
                    logger.warning(f"HTTP error {response.status_code}, retrying in {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    continue
                
                # Если это последняя попытка или ошибка не связана с доступностью сервера
                logger.error(f"HTTP error: {e}")
                logger.error(f"Response: {response.text}")
                raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Проблемы с соединением, повторяем запрос
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Connection error, retrying in {sleep_time} seconds: {e}")
                    time.sleep(sleep_time)
                    continue
                
                # Если это последняя попытка
                logger.error(f"Connection error: {e}")
                raise
            except Exception as e:
                # Другие ошибки
                logger.error(f"Request error: {e}")
                raise
    
    def _fetch_paginated_data(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        params: Dict[str, Any] = None,
        body: Any = None,
        auth: Optional[tuple] = None,
        pagination: Dict[str, Any] = None,
        max_retries: int = 3,
        retry_delay: int = 1,
        timeout: int = 30,
        request_rate_limit: float = 0
    ) -> List[Any]:
        """
        Получает данные из API с пагинацией.
        
        Args:
            method: HTTP-метод (GET, POST, PUT, DELETE)
            url: URL-адрес
            headers: Заголовки запроса
            params: Параметры запроса
            body: Тело запроса
            auth: Аутентификация (tuple username, password)
            pagination: Настройки пагинации
            max_retries: Максимальное количество попыток
            retry_delay: Задержка между попытками в секундах
            timeout: Таймаут запроса в секундах
            request_rate_limit: Ограничение скорости запросов (запросов в секунду)
            
        Returns:
            Список с данными всех страниц
        """
        if not pagination:
            raise ValueError("Pagination configuration not provided")
        
        # Получаем настройки пагинации
        pagination_type = pagination.get("type", "page")
        max_pages = pagination.get("max_pages", 10)
        
        # Копируем параметры, чтобы не изменять оригинал
        current_params = params.copy() if params else {}
        current_body = body.copy() if body and isinstance(body, dict) else body
        
        all_data = []
        page = 1
        
        while page <= max_pages:
            # Настраиваем параметры пагинации
            if pagination_type == "page":
                # Пагинация по номеру страницы
                page_param = pagination.get("page_param", "page")
                current_params[page_param] = page
                
                # Если указан параметр размера страницы, добавляем его
                if "page_size_param" in pagination and "page_size" in pagination:
                    current_params[pagination["page_size_param"]] = pagination["page_size"]
            elif pagination_type == "offset":
                # Пагинация по смещению
                offset_param = pagination.get("offset_param", "offset")
                limit_param = pagination.get("limit_param", "limit")
                limit = pagination.get("limit", 100)
                
                current_params[offset_param] = (page - 1) * limit
                current_params[limit_param] = limit
            elif pagination_type == "cursor":
                # Пагинация по курсору
                if page > 1 and "cursor" in pagination:
                    cursor_param = pagination.get("cursor_param", "cursor")
                    current_params[cursor_param] = pagination["cursor"]
            
            # Выполняем запрос
            response = self._make_request(
                method=method,
                url=url,
                headers=headers,
                params=current_params,
                body=current_body,
                auth=auth,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout
            )
            
            # Обрабатываем ответ
            if pagination.get("data_path"):
                # Извлекаем данные по указанному пути
                data_path = pagination["data_path"]
                page_data = self._extract_by_jsonpath(response, data_path)
            else:
                # Если путь не указан, используем весь ответ
                page_data = response
            
            # Добавляем данные страницы к общему результату
            if isinstance(page_data, list):
                all_data.extend(page_data)
            else:
                all_data.append(page_data)
            
            # Проверяем, есть ли следующая страница
            has_next_page = False
            
            if pagination_type == "page" or pagination_type == "offset":
                # Для пагинации по странице или смещению проверяем количество полученных элементов
                if isinstance(page_data, list) and len(page_data) > 0:
                    has_next_page = True
                elif "has_more_path" in pagination:
                    # Если указан путь к флагу наличия следующей страницы
                    has_more_path = pagination["has_more_path"]
                    has_next_page = self._extract_by_jsonpath(response, has_more_path)
                    # Преобразуем в булево значение
                    if not isinstance(has_next_page, bool):
                        has_next_page = bool(has_next_page)
            elif pagination_type == "cursor":
                # Для пагинации по курсору проверяем наличие следующего курсора
                if "next_cursor_path" in pagination:
                    next_cursor_path = pagination["next_cursor_path"]
                    next_cursor = self._extract_by_jsonpath(response, next_cursor_path)
                    
                    if next_cursor:
                        has_next_page = True
                        pagination["cursor"] = next_cursor
            
            # Если нет следующей страницы или достигнут лимит страниц, завершаем
            if not has_next_page or page >= max_pages:
                break
            
            # Увеличиваем номер страницы
            page += 1
            
            # Если задано ограничение скорости запросов, делаем паузу
            if request_rate_limit > 0:
                time.sleep(1.0 / request_rate_limit)
        
        return all_data
    
    def _parse_api_response(
        self,
        api_response: Any,
        api_config: Dict[str, Any]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Парсит ответ API и формирует текстовое содержимое и метаданные.
        
        Args:
            api_response: Ответ API
            api_config: Конфигурация API
            
        Returns:
            Кортеж (текстовое содержимое, список обработанных элементов)
        """
        # Получаем настройки парсера из конфигурации
        response_parser = api_config.get("response_parser", {})
        
        # Получаем путь к данным
        data_path = response_parser.get("data_path", "$")
        
        # Извлекаем данные по указанному пути
        items = self._extract_by_jsonpath(api_response, data_path)
        
        # Если результат не список, оборачиваем его в список
        if not isinstance(items, list):
            items = [items]
        
        # Получаем списки текстовых полей и полей метаданных
        text_fields = response_parser.get("text_fields", [])
        metadata_fields = response_parser.get("metadata_fields", [])
        
        # Если не указаны поля, используем все поля
        if not text_fields and not metadata_fields:
            # Пытаемся определить поля автоматически
            if items and isinstance(items[0], dict):
                all_fields = list(items[0].keys())
                text_fields = [f for f in all_fields if isinstance(items[0].get(f), str)]
                metadata_fields = [f for f in all_fields if f not in text_fields]
        
        # Обрабатываем каждый элемент
        parsed_items = []
        content_parts = []
        
        for i, item in enumerate(items):
            # Если элемент не словарь, преобразуем его
            if not isinstance(item, dict):
                if isinstance(item, (str, int, float, bool)):
                    item = {"value": item}
                else:
                    # Пропускаем сложные объекты, которые не являются словарями
                    continue
            
            # Извлекаем текстовые поля
            item_text_parts = []
            for field in text_fields:
                if field in item:
                    value = item[field]
                    # Форматируем значение
                    if value is not None:
                        item_text_parts.append(f"{field}: {value}")
            
            # Формируем текст элемента
            item_text = "\n".join(item_text_parts)
            
            # Добавляем разделитель между элементами
            if item_text:
                content_parts.append(f"--- Элемент {i+1} ---\n{item_text}")
            
            # Извлекаем метаданные
            item_metadata = {"id": i}
            for field in metadata_fields:
                if field in item:
                    item_metadata[field] = item[field]
            
            # Добавляем метаданные элемента
            parsed_items.append(item_metadata)
        
        # Объединяем текст всех элементов
        content = "\n\n".join(content_parts)
        
        # Добавляем заголовок
        api_name = api_config.get("api_name", "API")
        endpoint = api_config.get("endpoint", "")
        
        header = f"Данные из {api_name}\n"
        header += f"Источник: {endpoint}\n"
        header += f"Количество элементов: {len(parsed_items)}\n\n"
        
        content = header + content
        
        return content, parsed_items
    
    def _extract_by_jsonpath(self, data: Any, path: str) -> Any:
        """
        Извлекает данные по указанному JSONPath.
        
        Args:
            data: Данные для извлечения
            path: JSONPath-выражение
            
        Returns:
            Извлеченные данные
        """
        try:
            # Если путь пустой или $, возвращаем все данные
            if not path or path == "$":
                return data
            
            # Парсим JSONPath-выражение
            jsonpath_expr = jsonpath.parse(path)
            
            # Извлекаем данные
            matches = [match.value for match in jsonpath_expr.find(data)]
            
            # Если нет совпадений, возвращаем пустой список
            if not matches:
                return []
            
            # Если есть только одно совпадение, возвращаем его
            if len(matches) == 1:
                return matches[0]
            
            # Иначе возвращаем список совпадений
            return matches
        except Exception as e:
            logger.error(f"Error extracting data by JSONPath {path}: {e}")
            return None
    
    def _generate_cache_key(self, api_config: Dict[str, Any]) -> str:
        """
        Генерирует ключ кеша на основе конфигурации API.
        
        Args:
            api_config: Конфигурация API
            
        Returns:
            Ключ кеша
        """
        # Создаем копию конфигурации без изменения оригинала
        config_copy = api_config.copy()
        
        # Удаляем поля, которые не влияют на результат запроса
        if "response_parser" in config_copy:
            del config_copy["response_parser"]
        
        # Преобразуем конфигурацию в строку
        config_str = json.dumps(config_copy, sort_keys=True)
        
        # Вычисляем хеш
        hash_obj = hashlib.md5(config_str.encode('utf-8'))
        
        # Добавляем префикс с именем API
        api_name = api_config.get("api_name", "api")
        safe_api_name = re.sub(r'[^\w]', '_', api_name)
        
        return f"{safe_api_name}_{hash_obj.hexdigest()}"


# Регистрируем парсер в фабрике
from app.indexing.parsers.base import ParserFactory

ParserFactory.register_parser(APIParser)