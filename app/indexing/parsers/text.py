import os
import logging
from typing import Dict, Any, List, Optional, Union
import chardet
import re

from app.indexing.parsers.base import BaseParser, ParsedDocument
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TextParser(BaseParser):
    """Парсер для текстовых файлов."""
    
    def __init__(self):
        """Инициализация парсера текстовых файлов."""
        super().__init__()
        self.supported_extensions = [
            ".txt", ".md", ".markdown", ".rst",
            ".log", ".ini", ".cfg", ".conf",
            ".asc", ".srt", ".vtt", ".css", ".scss", ".less",
            ".html", ".htm"  # резерв, если веб-парсер не используется
        ]
    
    def can_parse(self, file_path: str) -> bool:
        """
        Проверяет, может ли парсер обработать данный файл.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            True, если парсер может обработать файл, иначе False.
        """
        # Проверяем расширение файла
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_extensions
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает текстовый файл и возвращает его содержимое и метаданные.
        
        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - encoding: Кодировка файла (по умолчанию автоопределение)
                - errors: Обработка ошибок при декодировании (по умолчанию 'replace')
                - line_numbers: Включать ли номера строк (по умолчанию False)
                - markdown_formatting: Использовать ли форматирование Markdown (по умолчанию True для .md и .markdown)
            
        Returns:
            Объект ParsedDocument с содержимым и метаданными
        """
        try:
            # Получаем метаданные файла
            metadata = self.get_file_metadata(file_path)
            
            # Определяем параметры
            chunk_size = kwargs.get("chunk_size", settings.processing.chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.processing.chunk_overlap)
            encoding = kwargs.get("encoding", None)
            errors = kwargs.get("errors", "replace")
            line_numbers = kwargs.get("line_numbers", False)
            
            # Для Markdown файлов включаем форматирование по умолчанию
            _, ext = os.path.splitext(file_path)
            is_markdown = ext.lower() in [".md", ".markdown"]
            markdown_formatting = kwargs.get("markdown_formatting", is_markdown)
            
            # Определяем тип содержимого
            content_type = self._get_content_type(ext.lower())
            
            # Определяем кодировку файла, если не указана
            if not encoding:
                encoding = self._detect_encoding(file_path)
            
            # Читаем файл
            with open(file_path, "r", encoding=encoding, errors=errors) as f:
                content = f.read()
            
            # Обрабатываем содержимое в зависимости от типа файла
            processed_content = self._process_content(
                content,
                file_type=ext.lower(),
                markdown_formatting=markdown_formatting,
                line_numbers=line_numbers
            )
            
            # Анализируем текст и дополняем метаданные
            text_metadata = self._analyze_text(processed_content)
            metadata.update(text_metadata)
            
            # Добавляем информацию о типе содержимого и кодировке
            metadata["content_type"] = content_type
            metadata["encoding"] = encoding
            metadata["content_length"] = len(content)
            
            # Разбиваем содержимое на чанки
            chunks = self.chunk_text(
                processed_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            return ParsedDocument(
                content=processed_content,
                metadata=metadata,
                chunks=chunks
            )
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
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
    
    def _get_content_type(self, extension: str) -> str:
        """
        Определяет MIME-тип содержимого по расширению файла.
        
        Args:
            extension: Расширение файла
            
        Returns:
            MIME-тип содержимого
        """
        # Карта расширений и соответствующих им типов содержимого
        content_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".rst": "text/x-rst",
            ".log": "text/plain",
            ".ini": "text/plain",
            ".cfg": "text/plain",
            ".conf": "text/plain",
            ".asc": "text/plain",
            ".srt": "text/plain",
            ".vtt": "text/vtt",
            ".css": "text/css",
            ".scss": "text/x-scss",
            ".less": "text/x-less",
            ".html": "text/html",
            ".htm": "text/html"
        }
        
        return content_types.get(extension, "text/plain")
    
    def _process_content(
        self,
        content: str,
        file_type: str = ".txt",
        markdown_formatting: bool = False,
        line_numbers: bool = False
    ) -> str:
        """
        Обрабатывает содержимое файла в зависимости от его типа.
        
        Args:
            content: Содержимое файла
            file_type: Тип файла (расширение)
            markdown_formatting: Использовать ли форматирование Markdown
            line_numbers: Включать ли номера строк
            
        Returns:
            Обработанное содержимое
        """
        # Нормализуем переносы строк
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        
        # Обрабатываем различные типы файлов
        if file_type in [".md", ".markdown"] and markdown_formatting:
            # Для Markdown можем добавить дополнительную обработку
            # Например, извлечение структуры, заголовков и т.д.
            pass
        elif file_type in [".ini", ".cfg", ".conf"]:
            # Для конфигурационных файлов можем добавить структурирование
            # Например, выделение секций и параметров
            pass
        elif file_type in [".log"]:
            # Для логов можем добавить парсинг дат, уровней логирования и т.д.
            pass
        
        # Добавляем номера строк, если требуется
        if line_numbers:
            lines = content.split("\n")
            content = "\n".join([f"{i+1}: {line}" for i, line in enumerate(lines)])
        
        return content
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Анализирует текст и возвращает метаданные.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с метаданными
        """
        metadata = {}
        
        try:
            # Разбиваем текст на строки и слова
            lines = text.split("\n")
            words = re.findall(r"\b\w+\b", text)
            
            # Подсчитываем статистику
            metadata["line_count"] = len(lines)
            metadata["word_count"] = len(words)
            metadata["char_count"] = len(text)
            
            # Находим самые длинные и короткие строки
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                shortest_line = min(non_empty_lines, key=len)
                longest_line = max(non_empty_lines, key=len)
                
                metadata["shortest_line_length"] = len(shortest_line)
                metadata["longest_line_length"] = len(longest_line)
            
            # Определяем язык текста (простая эвристика)
            metadata["language"] = self._detect_language(text)
            
            # Определяем формат текста (простая эвристика)
            metadata["has_headings"] = any(line.startswith("#") for line in lines)
            metadata["has_lists"] = any(line.strip().startswith(("-", "*", "1.")) for line in lines)
            metadata["has_code_blocks"] = "```" in text
            metadata["has_links"] = bool(re.search(r"\[.*?\]\(.*?\)", text) or re.search(r"https?://\S+", text))
            metadata["has_tables"] = "|" in text and any(line.strip().startswith("|") and line.strip().endswith("|") for line in lines)
            
            return metadata
        except Exception as e:
            logger.warning(f"Error analyzing text: {e}")
            return metadata
    
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
            sample = text[:1000].lower()
            
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

ParserFactory.register_parser(TextParser)