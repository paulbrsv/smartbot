import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict
from datetime import datetime

from app.indexing.parsers.base import BaseParser, ParsedDocument
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StructuredParser(BaseParser):
    """Парсер для структурированных данных (JSON, XML)."""
    
    def __init__(self):
        """Инициализация парсера структурированных данных."""
        super().__init__()
        self.supported_extensions = [
            ".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml"
        ]
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает структурированный файл и возвращает его содержимое и метаданные.
        
        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - encoding: Кодировка файла (по умолчанию автоопределение)
                - format_output: Форматировать ли вывод (по умолчанию True)
                - max_depth: Максимальная глубина обработки (по умолчанию 10)
                - flatten: Преобразовывать ли в плоскую структуру (по умолчанию False)
                - xml_pretty: Форматировать ли XML (по умолчанию True)
                - extract_schema: Извлекать ли схему (по умолчанию True)
                - include_raw: Включать ли исходное содержимое (по умолчанию False)
            
        Returns:
            Объект ParsedDocument с содержимым и метаданными
        """
        try:
            # Получаем метаданные файла
            metadata = self.get_file_metadata(file_path)
            
            # Определяем параметры
            chunk_size = kwargs.get("chunk_size", settings.processing.chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.processing.chunk_overlap)
            encoding = kwargs.get("encoding", self._detect_encoding(file_path))
            format_output = kwargs.get("format_output", True)
            max_depth = kwargs.get("max_depth", 10)
            flatten = kwargs.get("flatten", False)
            xml_pretty = kwargs.get("xml_pretty", True)
            extract_schema = kwargs.get("extract_schema", True)
            include_raw = kwargs.get("include_raw", False)
            
            # Получаем расширение файла
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # Читаем содержимое файла
            with open(file_path, 'r', encoding=encoding) as f:
                raw_content = f.read()
            
            # Обрабатываем файл в зависимости от его типа
            if ext in [".json", ".jsonl", ".ndjson"]:
                # Обрабатываем JSON
                content, structured_metadata = self._process_json(
                    raw_content,
                    file_path=file_path,
                    format_output=format_output,
                    max_depth=max_depth,
                    flatten=flatten,
                    extract_schema=extract_schema,
                    is_jsonl=ext in [".jsonl", ".ndjson"]
                )
            elif ext in [".xml"]:
                # Обрабатываем XML
                content, structured_metadata = self._process_xml(
                    raw_content,
                    file_path=file_path,
                    format_output=format_output,
                    max_depth=max_depth,
                    flatten=flatten,
                    extract_schema=extract_schema,
                    xml_pretty=xml_pretty
                )
            elif ext in [".yaml", ".yml"]:
                # Обрабатываем YAML
                content, structured_metadata = self._process_yaml(
                    raw_content,
                    file_path=file_path,
                    format_output=format_output,
                    max_depth=max_depth,
                    flatten=flatten,
                    extract_schema=extract_schema
                )
            else:
                # Обрабатываем как обычный текст
                content = raw_content
                structured_metadata = {
                    "format": "text",
                    "size": len(raw_content)
                }
            
            # Обновляем метаданные
            metadata.update(structured_metadata)
            
            # Включаем исходное содержимое, если требуется
            if include_raw:
                metadata["raw_content"] = raw_content
            
            # Добавляем информацию о размере содержимого и типе
            metadata["content_length"] = len(content)
            
            # Определяем MIME-тип
            if ext == ".json":
                metadata["content_type"] = "application/json"
            elif ext in [".jsonl", ".ndjson"]:
                metadata["content_type"] = "application/x-ndjson"
            elif ext == ".xml":
                metadata["content_type"] = "application/xml"
            elif ext in [".yaml", ".yml"]:
                metadata["content_type"] = "application/yaml"
            
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
            logger.error(f"Error parsing structured file {file_path}: {e}")
            raise
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Определяет кодировку файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Определенная кодировка или 'utf-8' по умолчанию
        """
        try:
            import chardet
            
            # Считываем часть файла для определения кодировки
            with open(file_path, "rb") as f:
                raw_data = f.read(4096)
            
            # Определяем кодировку
            result = chardet.detect(raw_data)
            confidence = result.get("confidence", 0)
            encoding = result.get("encoding", "utf-8")
            
            # Если уверенность низкая, используем utf-8
            if confidence < 0.7 or not encoding:
                encoding = "utf-8"
            
            # Нормализуем название кодировки
            encoding = encoding.lower().replace("-", "_")
            
            # Для некоторых кодировок используем замены
            encoding_map = {
                "ascii": "utf_8",
                "windows_1251": "cp1251",
                "windows_1252": "cp1252",
                "iso8859_1": "latin1"
            }
            
            return encoding_map.get(encoding, encoding)
        except Exception as e:
            logger.warning(f"Error detecting encoding: {e}, using utf-8")
            return "utf-8"
    
    def _process_json(
        self,
        content: str,
        file_path: str,
        format_output: bool = True,
        max_depth: int = 10,
        flatten: bool = False,
        extract_schema: bool = True,
        is_jsonl: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Обрабатывает JSON-содержимое.
        
        Args:
            content: Содержимое файла
            file_path: Путь к файлу
            format_output: Форматировать ли вывод
            max_depth: Максимальная глубина обработки
            flatten: Преобразовывать ли в плоскую структуру
            extract_schema: Извлекать ли схему
            is_jsonl: Является ли формат JSON Lines
            
        Returns:
            Кортеж (обработанное содержимое, метаданные)
        """
        metadata = {
            "format": "jsonl" if is_jsonl else "json",
            "size": len(content)
        }
        
        try:
            # Обрабатываем JSON Lines
            if is_jsonl:
                # Разбиваем по строкам и обрабатываем каждую строку как отдельный JSON
                lines = content.strip().split("\n")
                valid_lines = []
                objects = []
                
                for i, line in enumerate(lines):
                    try:
                        obj = json.loads(line)
                        objects.append(obj)
                        
                        # Форматируем JSON для вывода
                        if format_output:
                            formatted_json = json.dumps(obj, ensure_ascii=False, indent=2)
                            valid_lines.append(formatted_json)
                        else:
                            valid_lines.append(line)
                    except json.JSONDecodeError:
                        # Пропускаем невалидные строки
                        valid_lines.append(f"# Line {i+1}: Invalid JSON")
                
                # Формируем метаданные
                metadata["line_count"] = len(lines)
                metadata["valid_line_count"] = len(objects)
                metadata["invalid_line_count"] = len(lines) - len(objects)
                
                # Извлекаем схему, если требуется
                if extract_schema and objects:
                    schema = self._extract_json_schema(objects[0], max_depth=max_depth)
                    metadata["schema"] = schema
                    
                    # Проверяем согласованность схемы
                    schemas_consistent = True
                    for obj in objects[1:min(10, len(objects))]:
                        obj_schema = self._extract_json_schema(obj, max_depth=max_depth)
                        if obj_schema != schema:
                            schemas_consistent = False
                            break
                    
                    metadata["schemas_consistent"] = schemas_consistent
                
                # Преобразуем в плоскую структуру, если требуется
                if flatten and objects:
                    flattened_data = []
                    for obj in objects:
                        flat_obj = self._flatten_json(obj)
                        flattened_data.append(flat_obj)
                    
                    # Форматируем плоские данные
                    flat_content = "\n".join([json.dumps(obj, ensure_ascii=False) for obj in flattened_data])
                    metadata["flattened"] = True
                    
                    # Возвращаем плоское содержимое
                    return flat_content, metadata
                
                # Объединяем строки
                processed_content = "\n".join(valid_lines)
                
                return processed_content, metadata
            else:
                # Обрабатываем обычный JSON
                try:
                    obj = json.loads(content)
                    
                    # Извлекаем схему, если требуется
                    if extract_schema:
                        schema = self._extract_json_schema(obj, max_depth=max_depth)
                        metadata["schema"] = schema
                    
                    # Преобразуем в плоскую структуру, если требуется
                    if flatten:
                        flat_obj = self._flatten_json(obj)
                        metadata["flattened"] = True
                        
                        # Форматируем плоские данные
                        if format_output:
                            processed_content = json.dumps(flat_obj, ensure_ascii=False, indent=2)
                        else:
                            processed_content = json.dumps(flat_obj, ensure_ascii=False)
                        
                        return processed_content, metadata
                    
                    # Форматируем JSON для вывода
                    if format_output:
                        processed_content = json.dumps(obj, ensure_ascii=False, indent=2)
                    else:
                        processed_content = content
                    
                    # Анализируем структуру JSON
                    metadata.update(self._analyze_json_structure(obj))
                    
                    return processed_content, metadata
                except json.JSONDecodeError as e:
                    # Если не удалось разобрать JSON, возвращаем исходное содержимое
                    metadata["valid"] = False
                    metadata["error"] = str(e)
                    
                    return content, metadata
        except Exception as e:
            logger.warning(f"Error processing JSON: {e}")
            metadata["error"] = str(e)
            return content, metadata
    
    def _process_xml(
        self,
        content: str,
        file_path: str,
        format_output: bool = True,
        max_depth: int = 10,
        flatten: bool = False,
        extract_schema: bool = True,
        xml_pretty: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Обрабатывает XML-содержимое.
        
        Args:
            content: Содержимое файла
            file_path: Путь к файлу
            format_output: Форматировать ли вывод
            max_depth: Максимальная глубина обработки
            flatten: Преобразовывать ли в плоскую структуру
            extract_schema: Извлекать ли схему
            xml_pretty: Форматировать ли XML
            
        Returns:
            Кортеж (обработанное содержимое, метаданные)
        """
        metadata = {
            "format": "xml",
            "size": len(content)
        }
        
        try:
            # Пытаемся разобрать XML
            try:
                root = ET.fromstring(content)
                metadata["valid"] = True
            except ET.ParseError as e:
                metadata["valid"] = False
                metadata["error"] = str(e)
                return content, metadata
            
            # Анализируем структуру XML
            xml_metadata = self._analyze_xml_structure(root)
            metadata.update(xml_metadata)
            
            # Извлекаем схему, если требуется
            if extract_schema:
                schema = self._extract_xml_schema(root, max_depth=max_depth)
                metadata["schema"] = schema
            
            # Преобразуем в плоскую структуру, если требуется
            if flatten:
                flat_data = self._flatten_xml(root)
                metadata["flattened"] = True
                
                # Преобразуем плоские данные в JSON
                if format_output:
                    processed_content = json.dumps(flat_data, ensure_ascii=False, indent=2)
                else:
                    processed_content = json.dumps(flat_data, ensure_ascii=False)
                
                return processed_content, metadata
            
            # Форматируем XML для вывода
            if format_output and xml_pretty:
                # Используем minidom для форматирования XML
                xmlstr = ET.tostring(root, encoding='unicode')
                reparsed = minidom.parseString(xmlstr)
                processed_content = reparsed.toprettyxml(indent="  ")
                
                # Удаляем лишние пустые строки
                processed_content = "\n".join([line for line in processed_content.split("\n") if line.strip()])
            else:
                processed_content = content
            
            return processed_content, metadata
        except Exception as e:
            logger.warning(f"Error processing XML: {e}")
            metadata["error"] = str(e)
            return content, metadata
    
    def _process_yaml(
        self,
        content: str,
        file_path: str,
        format_output: bool = True,
        max_depth: int = 10,
        flatten: bool = False,
        extract_schema: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Обрабатывает YAML-содержимое.
        
        Args:
            content: Содержимое файла
            file_path: Путь к файлу
            format_output: Форматировать ли вывод
            max_depth: Максимальная глубина обработки
            flatten: Преобразовывать ли в плоскую структуру
            extract_schema: Извлекать ли схему
            
        Returns:
            Кортеж (обработанное содержимое, метаданные)
        """
        metadata = {
            "format": "yaml",
            "size": len(content)
        }
        
        try:
            try:
                import yaml
                # Пытаемся разобрать YAML
                obj = yaml.safe_load(content)
                metadata["valid"] = True
            except ImportError:
                metadata["error"] = "PyYAML library not installed"
                return content, metadata
            except yaml.YAMLError as e:
                metadata["valid"] = False
                metadata["error"] = str(e)
                return content, metadata
            
            # Извлекаем схему, если требуется
            if extract_schema and obj:
                schema = self._extract_json_schema(obj, max_depth=max_depth)
                metadata["schema"] = schema
            
            # Преобразуем в плоскую структуру, если требуется
            if flatten and obj:
                flat_obj = self._flatten_json(obj)
                metadata["flattened"] = True
                
                # Преобразуем плоские данные в JSON или YAML
                if format_output:
                    try:
                        processed_content = yaml.dump(flat_obj, allow_unicode=True, default_flow_style=False)
                    except:
                        processed_content = json.dumps(flat_obj, ensure_ascii=False, indent=2)
                else:
                    processed_content = json.dumps(flat_obj, ensure_ascii=False)
                
                return processed_content, metadata
            
            # Анализируем структуру YAML (так же, как JSON)
            if obj:
                metadata.update(self._analyze_json_structure(obj))
            
            # Форматируем YAML для вывода
            if format_output and obj:
                try:
                    processed_content = yaml.dump(obj, allow_unicode=True, default_flow_style=False)
                except:
                    # Если не удалось отформатировать YAML, преобразуем в JSON
                    processed_content = json.dumps(obj, ensure_ascii=False, indent=2)
            else:
                processed_content = content
            
            return processed_content, metadata
        except Exception as e:
            logger.warning(f"Error processing YAML: {e}")
            metadata["error"] = str(e)
            return content, metadata
    
    def _extract_json_schema(self, obj: Any, prefix: str = "", max_depth: int = 10, current_depth: int = 0) -> Dict[str, Any]:
        """
        Извлекает схему из JSON-объекта.
        
        Args:
            obj: JSON-объект
            prefix: Префикс для ключей (для рекурсивных вызовов)
            max_depth: Максимальная глубина обработки
            current_depth: Текущая глубина (для рекурсивных вызовов)
            
        Returns:
            Словарь со схемой
        """
        if current_depth >= max_depth:
            return {"type": "max_depth_reached"}
        
        # Определяем тип объекта
        if obj is None:
            return {"type": "null"}
        elif isinstance(obj, bool):
            return {"type": "boolean"}
        elif isinstance(obj, int):
            return {"type": "integer"}
        elif isinstance(obj, float):
            return {"type": "number"}
        elif isinstance(obj, str):
            return {"type": "string"}
        elif isinstance(obj, list):
            if not obj:
                return {"type": "array", "items": {"type": "unknown"}}
            
            # Определяем тип элементов массива
            item_types = []
            for i, item in enumerate(obj[:5]):  # Анализируем только первые 5 элементов
                item_schema = self._extract_json_schema(
                    item,
                    prefix=f"{prefix}[{i}]",
                    max_depth=max_depth,
                    current_depth=current_depth + 1
                )
                item_types.append(item_schema)
            
            # Проверяем, все ли элементы имеют одинаковый тип
            if all(t == item_types[0] for t in item_types):
                return {
                    "type": "array",
                    "items": item_types[0]
                }
            else:
                return {
                    "type": "array",
                    "items": {"type": "mixed", "observed_types": item_types}
                }
        elif isinstance(obj, dict):
            properties = {}
            
            for key, value in obj.items():
                properties[key] = self._extract_json_schema(
                    value,
                    prefix=f"{prefix}.{key}" if prefix else key,
                    max_depth=max_depth,
                    current_depth=current_depth + 1
                )
            
            return {
                "type": "object",
                "properties": properties
            }
        else:
            return {"type": "unknown"}
    
    def _analyze_json_structure(self, obj: Any) -> Dict[str, Any]:
        """
        Анализирует структуру JSON-объекта.
        
        Args:
            obj: JSON-объект
            
        Returns:
            Словарь с результатами анализа
        """
        analysis = {}
        
        try:
            # Проверяем тип корневого объекта
            if isinstance(obj, dict):
                analysis["root_type"] = "object"
                analysis["property_count"] = len(obj)
                analysis["top_level_properties"] = list(obj.keys())
            elif isinstance(obj, list):
                analysis["root_type"] = "array"
                analysis["array_length"] = len(obj)
                
                # Определяем тип элементов массива
                if obj:
                    if all(isinstance(item, dict) for item in obj):
                        analysis["array_items_type"] = "object"
                        
                        # Проверяем, имеют ли все объекты одинаковые ключи
                        if len(obj) > 1:
                            first_keys = set(obj[0].keys())
                            all_same_keys = all(set(item.keys()) == first_keys for item in obj[1:])
                            analysis["uniform_objects"] = all_same_keys
                            
                            if all_same_keys and obj[0]:
                                analysis["object_properties"] = list(obj[0].keys())
                    elif all(isinstance(item, list) for item in obj):
                        analysis["array_items_type"] = "array"
                    elif all(isinstance(item, (int, float)) for item in obj):
                        analysis["array_items_type"] = "number"
                    elif all(isinstance(item, str) for item in obj):
                        analysis["array_items_type"] = "string"
                    else:
                        analysis["array_items_type"] = "mixed"
            else:
                analysis["root_type"] = "primitive"
            
            # Рассчитываем глубину вложенности
            depth = self._calculate_json_depth(obj)
            analysis["depth"] = depth
            
            # Подсчитываем количество элементов
            count = self._count_json_elements(obj)
            analysis["element_count"] = count
            
            return analysis
        except Exception as e:
            logger.warning(f"Error analyzing JSON structure: {e}")
            return analysis
    
    def _calculate_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """
        Рассчитывает глубину вложенности JSON-объекта.
        
        Args:
            obj: JSON-объект
            current_depth: Текущая глубина (для рекурсивных вызовов)
            
        Returns:
            Глубина вложенности
        """
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            
            depths = [self._calculate_json_depth(value, current_depth + 1) for value in obj.values()]
            return max(depths) if depths else current_depth
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            
            depths = [self._calculate_json_depth(item, current_depth + 1) for item in obj]
            return max(depths) if depths else current_depth
        else:
            return current_depth
    
    def _count_json_elements(self, obj: Any) -> int:
        """
        Подсчитывает количество элементов в JSON-объекте.
        
        Args:
            obj: JSON-объект
            
        Returns:
            Количество элементов
        """
        if isinstance(obj, dict):
            count = len(obj)
            for value in obj.values():
                count += self._count_json_elements(value)
            return count
        elif isinstance(obj, list):
            count = len(obj)
            for item in obj:
                count += self._count_json_elements(item)
            return count
        else:
            return 1
    
    def _flatten_json(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        """
        Преобразует вложенный JSON-объект в плоскую структуру.
        
        Args:
            obj: JSON-объект
            prefix: Префикс для ключей (для рекурсивных вызовов)
            
        Returns:
            Словарь с плоской структурой
        """
        items = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Формируем новый ключ
                new_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    # Рекурсивно обрабатываем вложенные объекты
                    nested_items = self._flatten_json(value, new_key)
                    items.update(nested_items)
                else:
                    # Добавляем примитивное значение
                    items[new_key] = value
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                # Формируем новый ключ
                new_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                
                if isinstance(value, (dict, list)):
                    # Рекурсивно обрабатываем вложенные объекты
                    nested_items = self._flatten_json(value, new_key)
                    items.update(nested_items)
                else:
                    # Добавляем примитивное значение
                    items[new_key] = value
        else:
            # Примитивное значение
            items[prefix] = obj
        
        return items
    
    def _analyze_xml_structure(self, root) -> Dict[str, Any]:
        """
        Анализирует структуру XML-документа.
        
        Args:
            root: Корневой элемент XML
            
        Returns:
            Словарь с результатами анализа
        """
        analysis = {}
        
        try:
            # Анализируем корневой элемент
            analysis["root_tag"] = root.tag
            
            # Подсчитываем количество элементов
            element_count = 0
            tag_counts = defaultdict(int)
            
            # Функция для обхода дерева XML
            def count_elements(elem):
                nonlocal element_count
                element_count += 1
                tag_counts[elem.tag] += 1
                
                for child in elem:
                    count_elements(child)
            
            # Обходим дерево
            count_elements(root)
            
            analysis["element_count"] = element_count
            analysis["unique_tags"] = len(tag_counts)
            analysis["tag_counts"] = dict(tag_counts)
            
            # Находим самые часто встречающиеся теги
            most_common = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            analysis["most_common_tags"] = {tag: count for tag, count in most_common}
            
            # Рассчитываем глубину вложенности
            depth = self._calculate_xml_depth(root)
            analysis["depth"] = depth
            
            # Проверяем наличие атрибутов
            attributes = []
            def collect_attributes(elem):
                for key, value in elem.attrib.items():
                    attributes.append(f"{elem.tag}[@{key}]")
                
                for child in elem:
                    collect_attributes(child)
            
            collect_attributes(root)
            analysis["has_attributes"] = len(attributes) > 0
            analysis["attribute_count"] = len(attributes)
            
            if attributes:
                analysis["attribute_examples"] = attributes[:5]
            
            return analysis
        except Exception as e:
            logger.warning(f"Error analyzing XML structure: {e}")
            return analysis
    
    def _calculate_xml_depth(self, elem, current_depth: int = 0) -> int:
        """
        Рассчитывает глубину вложенности XML-элемента.
        
        Args:
            elem: XML-элемент
            current_depth: Текущая глубина (для рекурсивных вызовов)
            
        Returns:
            Глубина вложенности
        """
        max_depth = current_depth
        
        for child in elem:
            child_depth = self._calculate_xml_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _extract_xml_schema(self, root, max_depth: int = 10, current_depth: int = 0) -> Dict[str, Any]:
        """
        Извлекает схему из XML-элемента.
        
        Args:
            root: XML-элемент
            max_depth: Максимальная глубина обработки
            current_depth: Текущая глубина (для рекурсивных вызовов)
            
        Returns:
            Словарь со схемой
        """
        if current_depth >= max_depth:
            return {"type": "max_depth_reached"}
        
        schema = {
            "tag": root.tag,
            "type": "element"
        }
        
        # Добавляем атрибуты, если они есть
        if root.attrib:
            schema["attributes"] = {key: {"type": "attribute"} for key in root.attrib.keys()}
        
        # Подсчитываем дочерние элементы по тегам
        children = defaultdict(list)
        for child in root:
            children[child.tag].append(child)
        
        # Если есть дочерние элементы, добавляем их схемы
        if children:
            schema["children"] = {}
            
            for tag, elements in children.items():
                # Проверяем, все ли элементы с этим тегом имеют одинаковую структуру
                if len(elements) == 1:
                    # Единственный элемент
                    schema["children"][tag] = self._extract_xml_schema(
                        elements[0],
                        max_depth=max_depth,
                        current_depth=current_depth + 1
                    )
                else:
                    # Несколько элементов
                    schema["children"][tag] = {
                        "type": "array",
                        "items": self._extract_xml_schema(
                            elements[0],
                            max_depth=max_depth,
                            current_depth=current_depth + 1
                        )
                    }
        
        # Добавляем текстовое содержимое, если оно есть
        if root.text and root.text.strip():
            schema["text"] = {"type": "text"}
        
        return schema
    
    def _flatten_xml(self, root, prefix: str = "") -> Dict[str, Any]:
        """
        Преобразует XML-элемент в плоскую структуру.
        
        Args:
            root: XML-элемент
            prefix: Префикс для ключей (для рекурсивных вызовов)
            
        Returns:
            Словарь с плоской структурой
        """
        items = {}
        
        # Формируем новый ключ
        new_prefix = f"{prefix}/{root.tag}" if prefix else root.tag
        
        # Добавляем атрибуты
        for key, value in root.attrib.items():
            items[f"{new_prefix}[@{key}]"] = value
        
        # Добавляем текстовое содержимое
        if root.text and root.text.strip():
            items[f"{new_prefix}/text()"] = root.text.strip()
        
        # Группируем дочерние элементы по тегам
        children = defaultdict(list)
        for child in root:
            children[child.tag].append(child)
        
        # Обрабатываем дочерние элементы
        for tag, elements in children.items():
            if len(elements) == 1:
                # Единственный элемент
                nested_items = self._flatten_xml(elements[0], new_prefix)
                items.update(nested_items)
            else:
                # Несколько элементов
                for i, elem in enumerate(elements):
                    nested_items = self._flatten_xml(elem, f"{new_prefix}/{tag}[{i}]")
                    items.update(nested_items)
        
        return items


# Регистрируем парсер в фабрике
from app.indexing.parsers.base import ParserFactory

ParserFactory.register_parser(StructuredParser)