import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import csv
import re
from io import StringIO
import json
from datetime import datetime

from app.indexing.parsers.base import BaseParser, ParsedDocument
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TabularParser(BaseParser):
    """Парсер для табличных файлов (CSV, Excel)."""
    
    def __init__(self):
        """Инициализация парсера табличных файлов."""
        super().__init__()
        self.supported_extensions = [
            ".csv", ".tsv", ".xlsx", ".xls", ".ods"
        ]
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает табличный файл и возвращает его содержимое и метаданные.
        
        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - delimiter: Разделитель для CSV (по умолчанию автоопределение)
                - encoding: Кодировка файла (по умолчанию автоопределение)
                - sheet_name: Имя или индекс листа для Excel (по умолчанию None - все листы)
                - header: Индекс строки заголовка (по умолчанию 0)
                - skip_rows: Количество строк для пропуска в начале (по умолчанию 0)
                - max_rows: Максимальное количество строк для обработки (по умолчанию None - все)
                - date_format: Формат дат (по умолчанию None - автоопределение)
                - include_index: Включать ли индекс строк (по умолчанию False)
                - table_format: Формат таблицы ("markdown", "plain", "json") (по умолчанию "markdown")
            
        Returns:
            Объект ParsedDocument с содержимым и метаданными
        """
        try:
            import pandas as pd
            
            # Получаем метаданные файла
            metadata = self.get_file_metadata(file_path)
            
            # Определяем параметры
            chunk_size = kwargs.get("chunk_size", settings.processing.chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.processing.chunk_overlap)
            delimiter = kwargs.get("delimiter", None)
            encoding = kwargs.get("encoding", None)
            sheet_name = kwargs.get("sheet_name", None)
            header = kwargs.get("header", 0)
            skip_rows = kwargs.get("skip_rows", 0)
            max_rows = kwargs.get("max_rows", None)
            date_format = kwargs.get("date_format", None)
            include_index = kwargs.get("include_index", False)
            table_format = kwargs.get("table_format", "markdown")
            
            # Получаем расширение файла
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # Определяем кодировку файла, если не указана
            if not encoding and ext in [".csv", ".tsv"]:
                encoding = self._detect_encoding(file_path)
            
            # Определяем разделитель для CSV, если не указан
            if not delimiter and ext in [".csv", ".tsv"]:
                delimiter = self._detect_delimiter(file_path, encoding)
            
            # Читаем файл в зависимости от формата
            if ext in [".csv", ".tsv"]:
                # Для CSV используем явный разделитель
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    header=header,
                    skiprows=skip_rows,
                    nrows=max_rows,
                    on_bad_lines='warn'
                )
                # Создаем словарь с одним DataFrame для единообразия обработки
                dataframes = {"main": df}
            else:
                # Для Excel читаем все листы или указанный лист
                if sheet_name is not None:
                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                        header=header,
                        skiprows=skip_rows,
                        nrows=max_rows
                    )
                    if isinstance(df, pd.DataFrame):
                        # Если вернулся один DataFrame, создаем словарь с одним листом
                        dataframes = {sheet_name: df}
                    else:
                        # Если вернулся словарь DataFrames, используем его как есть
                        dataframes = df
                else:
                    # Читаем все листы
                    dataframes = pd.read_excel(
                        file_path,
                        sheet_name=None,
                        header=header,
                        skiprows=skip_rows,
                        nrows=max_rows
                    )
            
            # Извлекаем метаданные из табличных данных
            table_metadata = self._extract_table_metadata(dataframes)
            metadata.update(table_metadata)
            
            # Преобразуем таблицы в текстовый формат
            content, format_metadata = self._format_tables(
                dataframes,
                table_format=table_format,
                include_index=include_index
            )
            
            # Обновляем метаданные
            metadata.update(format_metadata)
            
            # Добавляем информацию о размере содержимого и типе
            metadata["content_length"] = len(content)
            
            if ext == ".csv":
                metadata["content_type"] = "text/csv"
            elif ext == ".tsv":
                metadata["content_type"] = "text/tab-separated-values"
            elif ext == ".xlsx":
                metadata["content_type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif ext == ".xls":
                metadata["content_type"] = "application/vnd.ms-excel"
            elif ext == ".ods":
                metadata["content_type"] = "application/vnd.oasis.opendocument.spreadsheet"
            
            # Добавляем информацию о параметрах обработки
            metadata["parsing_params"] = {
                "delimiter": delimiter,
                "encoding": encoding,
                "header": header,
                "skip_rows": skip_rows,
                "max_rows": max_rows,
                "date_format": date_format,
                "table_format": table_format
            }
            
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
        except ImportError:
            logger.error("pandas or openpyxl libraries not installed")
            raise ImportError("pandas and openpyxl are required for parsing tabular files")
        except Exception as e:
            logger.error(f"Error parsing tabular file {file_path}: {e}")
            raise
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Определяет кодировку CSV файла.
        
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
    
    def _detect_delimiter(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Определяет разделитель в CSV файле.
        
        Args:
            file_path: Путь к файлу
            encoding: Кодировка файла
            
        Returns:
            Определенный разделитель или ',' по умолчанию
        """
        try:
            # Список возможных разделителей
            delimiters = [',', ';', '\t', '|']
            
            # Считываем первые несколько строк файла
            sample_lines = []
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                for _ in range(5):  # Читаем первые 5 строк
                    line = f.readline().strip()
                    if line:
                        sample_lines.append(line)
            
            if not sample_lines:
                return ','
            
            # Проверяем каждый разделитель
            results = {}
            for delimiter in delimiters:
                # Подсчитываем количество полей для каждой строки с данным разделителем
                field_counts = []
                for line in sample_lines:
                    reader = csv.reader(StringIO(line), delimiter=delimiter)
                    row = next(reader, None)
                    if row:
                        field_counts.append(len(row))
                
                # Если разделитель хорош, количество полей должно быть одинаковым для всех строк
                if field_counts and all(count == field_counts[0] for count in field_counts):
                    results[delimiter] = field_counts[0]
            
            # Выбираем разделитель с наибольшим количеством полей
            if results:
                best_delimiter = max(results.items(), key=lambda x: x[1])[0]
                return best_delimiter
            
            # Если не удалось определить, используем запятую
            return ','
        except Exception as e:
            logger.warning(f"Error detecting delimiter: {e}, using comma")
            return ','
    
    def _extract_table_metadata(self, dataframes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлекает метаданные из табличных данных.
        
        Args:
            dataframes: Словарь с DataFrame для каждого листа
            
        Returns:
            Словарь с метаданными
        """
        metadata = {
            "sheet_count": len(dataframes),
            "sheets": [],
            "total_rows": 0,
            "total_columns": 0,
            "column_names": {},
            "data_types": {},
            "missing_values": {},
            "has_numeric_data": False,
            "has_datetime_data": False,
            "has_text_data": False
        }
        
        # Счетчики для определения типов данных
        numeric_count = 0
        datetime_count = 0
        text_count = 0
        
        # Обрабатываем каждый лист
        for sheet_name, df in dataframes.items():
            sheet_metadata = {
                "name": sheet_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            }
            
            # Подсчитываем количество пропущенных значений
            missing_values = df.isna().sum().to_dict()
            sheet_metadata["missing_values"] = {str(k): int(v) for k, v in missing_values.items() if v > 0}
            
            # Определяем типы данных
            dtypes = df.dtypes.astype(str).to_dict()
            sheet_metadata["data_types"] = {str(k): v for k, v in dtypes.items()}
            
            # Обновляем счетчики типов данных
            for dtype in dtypes.values():
                if "int" in dtype or "float" in dtype:
                    numeric_count += 1
                elif "datetime" in dtype:
                    datetime_count += 1
                else:
                    text_count += 1
            
            # Добавляем метаданные листа
            metadata["sheets"].append(sheet_metadata)
            
            # Обновляем общие счетчики
            metadata["total_rows"] += len(df)
            metadata["total_columns"] += len(df.columns)
            
            # Сохраняем имена столбцов для каждого листа
            metadata["column_names"][sheet_name] = df.columns.tolist()
            
            # Сохраняем типы данных для каждого листа
            metadata["data_types"][sheet_name] = {str(k): v for k, v in dtypes.items()}
            
            # Сохраняем пропущенные значения для каждого листа
            metadata["missing_values"][sheet_name] = {str(k): int(v) for k, v in missing_values.items() if v > 0}
        
        # Определяем типы данных в таблицах
        metadata["has_numeric_data"] = numeric_count > 0
        metadata["has_datetime_data"] = datetime_count > 0
        metadata["has_text_data"] = text_count > 0
        
        return metadata
    
    def _format_tables(
        self,
        dataframes: Dict[str, Any],
        table_format: str = "markdown",
        include_index: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Преобразует таблицы в текстовый формат.
        
        Args:
            dataframes: Словарь с DataFrame для каждого листа
            table_format: Формат таблицы ("markdown", "plain", "json")
            include_index: Включать ли индекс строк
            
        Returns:
            Кортеж (текстовое представление таблиц, метаданные форматирования)
        """
        text_parts = []
        metadata = {
            "format": table_format,
            "include_index": include_index
        }
        
        # Обрабатываем каждый лист
        for sheet_name, df in dataframes.items():
            # Добавляем заголовок листа
            text_parts.append(f"### Лист: {sheet_name}")
            text_parts.append("")
            
            # Ограничиваем количество отображаемых строк для очень больших таблиц
            max_display_rows = 1000
            if len(df) > max_display_rows:
                df_display = df.head(max_display_rows)
                truncated = True
            else:
                df_display = df
                truncated = False
            
            # Форматируем таблицу в зависимости от выбранного формата
            if table_format == "markdown":
                # Markdown формат
                table_text = df_display.to_markdown(index=include_index)
                text_parts.append(table_text)
            elif table_format == "json":
                # JSON формат
                table_dict = df_display.to_dict(orient="records")
                table_text = json.dumps(table_dict, ensure_ascii=False, indent=2)
                text_parts.append("```json")
                text_parts.append(table_text)
                text_parts.append("```")
            else:
                # Простой текстовый формат
                if include_index:
                    table_text = df_display.to_string()
                else:
                    table_text = df_display.to_string(index=False)
                text_parts.append("```")
                text_parts.append(table_text)
                text_parts.append("```")
            
            # Добавляем информацию о усечении таблицы
            if truncated:
                text_parts.append(f"\n*Показаны только первые {max_display_rows} строк из {len(df)}*")
            
            # Добавляем пустую строку между листами
            text_parts.append("")
        
        # Объединяем текст
        content = "\n".join(text_parts)
        
        return content, metadata
    
    def _detect_language(self, text: str) -> str:
        """
        Определяет язык текста (простая эвристика).
        
        Args:
            text: Текст для анализа
            
        Returns:
            Код языка (en, ru, и т.д.)
        """
        try:
            # Берем небольшой фрагмент текста для анализа
            sample = text[:2000].lower()
            
            # Словарь с частотными словами для разных языков
            language_words = {
                "ru": {"и", "в", "не", "на", "я", "что", "с", "по", "это", "для", "как", "к", "из", "а", "то"},
                "en": {"the", "and", "to", "of", "a", "in", "is", "that", "for", "it", "with", "as", "on", "be", "at"},
                "de": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "für", "ist", "im"},
                "fr": {"le", "la", "et", "les", "des", "en", "un", "une", "du", "dans", "est", "que", "pour", "qui", "sur"}
            }
            
            # Подсчитываем количество совпадений для каждого языка
            matches = {}
            words = re.findall(r"\b\w+\b", sample)
            
            for lang, lang_words in language_words.items():
                matches[lang] = sum(1 for word in words if word in lang_words)
            
            # Определяем язык с наибольшим количеством совпадений
            if not matches:
                return "unknown"
            
            best_lang = max(matches, key=matches.get)
            
            # Если количество совпадений слишком мало, считаем язык неопределенным
            if matches[best_lang] < 3:
                return "unknown"
            
            return best_lang
        except Exception as e:
            logger.warning(f"Error detecting language: {e}")
            return "unknown"


# Регистрируем парсер в фабрике
from app.indexing.parsers.base import ParserFactory

ParserFactory.register_parser(TabularParser)