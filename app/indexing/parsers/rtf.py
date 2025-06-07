import os
import logging
from typing import Dict, Any, List, Optional
import re
from datetime import datetime
import io

from app.indexing.parsers.base import BaseParser, ParsedDocument
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RTFParser(BaseParser):
    """Парсер для RTF-файлов (Rich Text Format)."""
    
    def __init__(self):
        """Инициализация парсера RTF-файлов."""
        super().__init__()
        self.supported_extensions = [".rtf"]
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает RTF-файл и возвращает его содержимое и метаданные.
        
        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - extract_images: Извлекать ли изображения (по умолчанию False)
                - extract_formatting: Сохранять ли информацию о форматировании (по умолчанию False)
                - fallback_encoding: Кодировка для использования при сбое (по умолчанию 'utf-8')
            
        Returns:
            Объект ParsedDocument с содержимым и метаданными
        """
        try:
            # Получаем метаданные файла
            metadata = self.get_file_metadata(file_path)
            
            # Определяем параметры
            chunk_size = kwargs.get("chunk_size", settings.processing.chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.processing.chunk_overlap)
            extract_images = kwargs.get("extract_images", False)
            extract_formatting = kwargs.get("extract_formatting", False)
            fallback_encoding = kwargs.get("fallback_encoding", "utf-8")
            
            # Пытаемся использовать striprtf библиотеку
            try:
                from striprtf.striprtf import rtf_to_text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    rtf_text = file.read()
                plain_text = rtf_to_text(rtf_text)
                content = plain_text
                
                # Добавляем информацию о парсере в метаданные
                metadata["parser"] = "striprtf"
            except ImportError:
                logger.warning("striprtf library not available, trying alternative methods")
                # Если библиотека striprtf недоступна, пробуем альтернативный подход
                content = self._extract_text_alternative(file_path, fallback_encoding)
                metadata["parser"] = "alternative"
            except Exception as e:
                logger.warning(f"Error using striprtf: {e}, trying alternative methods")
                content = self._extract_text_alternative(file_path, fallback_encoding)
                metadata["parser"] = "alternative"
            
            # Извлекаем метаданные RTF-файла
            rtf_metadata = self._extract_rtf_metadata(file_path)
            metadata.update(rtf_metadata)
            
            # Если требуется, извлекаем изображения
            if extract_images:
                try:
                    images = self._extract_images(file_path)
                    metadata["images"] = images
                    metadata["image_count"] = len(images)
                except Exception as e:
                    logger.warning(f"Error extracting images from RTF: {e}")
                    metadata["image_extraction_error"] = str(e)
            
            # Если требуется, извлекаем информацию о форматировании
            if extract_formatting:
                try:
                    formatting = self._extract_formatting(file_path)
                    metadata["formatting"] = formatting
                except Exception as e:
                    logger.warning(f"Error extracting formatting from RTF: {e}")
                    metadata["formatting_extraction_error"] = str(e)
            
            # Добавляем информацию о типе содержимого
            metadata["content_type"] = "text/rtf"
            metadata["content_length"] = len(content)
            
            # Анализируем текст и дополняем метаданные
            text_metadata = self._analyze_text(content)
            metadata.update(text_metadata)
            
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
            logger.error(f"Error parsing RTF file {file_path}: {e}")
            raise
    
    def _extract_text_alternative(self, file_path: str, fallback_encoding: str = "utf-8") -> str:
        """
        Альтернативный метод извлечения текста из RTF-файла.
        
        Args:
            file_path: Путь к файлу
            fallback_encoding: Кодировка для использования при сбое
            
        Returns:
            Извлеченный текст
        """
        try:
            # Пробуем использовать pyth
            try:
                from pyth.plugins.rtf15.reader import Rtf15Reader
                from pyth.plugins.plaintext.writer import PlaintextWriter
                
                with open(file_path, "rb") as f:
                    doc = Rtf15Reader.read(f)
                plain_text = PlaintextWriter.write(doc).getvalue()
                return plain_text
            except ImportError:
                logger.warning("pyth library not available, trying direct RTF parsing")
            
            # Если pyth недоступен, пробуем прямую обработку RTF
            with open(file_path, "rb") as f:
                rtf_binary = f.read()
            
            # Убираем RTF-разметку с помощью регулярных выражений
            # Это упрощенный подход, который может не работать со сложными RTF-файлами
            
            # Преобразуем бинарные данные в строку
            rtf_text = rtf_binary.decode(fallback_encoding, errors='replace')
            
            # Удаляем RTF-команды
            text = re.sub(r"\\[a-z0-9]+", " ", rtf_text)
            text = re.sub(r"\\[{}]", "", text)
            text = re.sub(r"{[^{}]*}", "", text)
            
            # Удаляем управляющие символы
            text = re.sub(r"[\x00-\x1F\x7F]", "", text)
            
            # Нормализуем пробелы
            text = re.sub(r"\s+", " ", text)
            
            return text.strip()
        except Exception as e:
            logger.error(f"All RTF extraction methods failed: {e}")
            # В крайнем случае, пытаемся просто прочитать файл как текст
            with open(file_path, "r", encoding=fallback_encoding, errors="replace") as f:
                return f.read()
    
    def _extract_rtf_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Извлекает метаданные из RTF-файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Словарь с метаданными
        """
        metadata = {}
        
        try:
            # Читаем начало файла для извлечения метаданных
            with open(file_path, "rb") as f:
                header = f.read(4096).decode("utf-8", errors="ignore")
            
            # Извлекаем информацию о заголовке
            title_match = re.search(r"\\title\s+([^\\{}]+)", header)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
            
            # Извлекаем информацию об авторе
            author_match = re.search(r"\\author\s+([^\\{}]+)", header)
            if author_match:
                metadata["author"] = author_match.group(1).strip()
            
            # Извлекаем информацию о компании
            company_match = re.search(r"\\company\s+([^\\{}]+)", header)
            if company_match:
                metadata["company"] = company_match.group(1).strip()
            
            # Извлекаем информацию об операторе
            operator_match = re.search(r"\\operator\s+([^\\{}]+)", header)
            if operator_match:
                metadata["operator"] = operator_match.group(1).strip()
            
            # Извлекаем информацию о категории
            category_match = re.search(r"\\category\s+([^\\{}]+)", header)
            if category_match:
                metadata["category"] = category_match.group(1).strip()
            
            # Извлекаем информацию о ключевых словах
            keywords_match = re.search(r"\\keywords\s+([^\\{}]+)", header)
            if keywords_match:
                keywords = keywords_match.group(1).strip()
                metadata["keywords"] = [k.strip() for k in keywords.split(";") if k.strip()]
            
            # Извлекаем информацию о комментариях
            comment_match = re.search(r"\\comment\s+([^\\{}]+)", header)
            if comment_match:
                metadata["comment"] = comment_match.group(1).strip()
            
            # Извлекаем информацию о версии RTF
            version_match = re.search(r"\\rtf(\d+)", header)
            if version_match:
                metadata["rtf_version"] = int(version_match.group(1))
            
            # Извлекаем информацию о кодовой странице
            ansicpg_match = re.search(r"\\ansicpg(\d+)", header)
            if ansicpg_match:
                metadata["ansi_codepage"] = int(ansicpg_match.group(1))
            
            # Извлекаем информацию о языке
            lang_match = re.search(r"\\deflang(\d+)", header)
            if lang_match:
                metadata["language_id"] = int(lang_match.group(1))
            
            return metadata
        except Exception as e:
            logger.warning(f"Error extracting RTF metadata: {e}")
            return metadata
    
    def _extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Извлекает изображения из RTF-файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Список словарей с информацией об изображениях
        """
        images = []
        
        try:
            # Читаем файл для извлечения изображений
            with open(file_path, "rb") as f:
                content = f.read()
            
            # Ищем блоки данных изображений
            # \\pict - начало блока изображения
            # \\pngblip, \\jpegblip, \\emfblip - тип изображения
            
            # Ищем PNG изображения
            png_matches = re.finditer(r"\\pict[^}]*\\pngblip([^}]*)", content.decode("latin1", errors="ignore"))
            for i, match in enumerate(png_matches):
                images.append({
                    "index": i,
                    "type": "png",
                    "format": "PNG",
                    "position": match.start(),
                    "size": len(match.group(1))
                })
            
            # Ищем JPEG изображения
            jpeg_matches = re.finditer(r"\\pict[^}]*\\jpegblip([^}]*)", content.decode("latin1", errors="ignore"))
            for i, match in enumerate(jpeg_matches):
                images.append({
                    "index": i + len(images),
                    "type": "jpeg",
                    "format": "JPEG",
                    "position": match.start(),
                    "size": len(match.group(1))
                })
            
            # Ищем EMF изображения
            emf_matches = re.finditer(r"\\pict[^}]*\\emfblip([^}]*)", content.decode("latin1", errors="ignore"))
            for i, match in enumerate(emf_matches):
                images.append({
                    "index": i + len(images),
                    "type": "emf",
                    "format": "EMF",
                    "position": match.start(),
                    "size": len(match.group(1))
                })
            
            return images
        except Exception as e:
            logger.warning(f"Error extracting images from RTF: {e}")
            return images
    
    def _extract_formatting(self, file_path: str) -> Dict[str, Any]:
        """
        Извлекает информацию о форматировании из RTF-файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Словарь с информацией о форматировании
        """
        formatting = {}
        
        try:
            # Читаем файл для извлечения информации о форматировании
            with open(file_path, "rb") as f:
                content = f.read(8192).decode("utf-8", errors="ignore")
            
            # Извлекаем информацию о шрифтах
            # Таблица шрифтов начинается с \fonttbl и содержит записи вида \f0\fswiss\fcharset0 Arial;
            font_table_match = re.search(r"\\fonttbl\s*{([^}]*)}", content)
            if font_table_match:
                font_table = font_table_match.group(1)
                fonts = []
                
                # Извлекаем записи о шрифтах
                font_matches = re.finditer(r"\\f(\d+)[^;]*\\fcharset(\d+)\s+([^;]+);", font_table)
                for match in font_matches:
                    font_id = int(match.group(1))
                    charset = int(match.group(2))
                    font_name = match.group(3).strip()
                    
                    fonts.append({
                        "id": font_id,
                        "charset": charset,
                        "name": font_name
                    })
                
                if fonts:
                    formatting["fonts"] = fonts
            
            # Извлекаем информацию о цветах
            # Таблица цветов начинается с \colortbl и содержит записи вида \red255\green0\blue0;
            color_table_match = re.search(r"\\colortbl\s*;([^}]*)", content)
            if color_table_match:
                color_table = color_table_match.group(1)
                colors = []
                
                # Извлекаем записи о цветах
                color_matches = re.finditer(r"\\red(\d+)\\green(\d+)\\blue(\d+);", color_table)
                for i, match in enumerate(color_matches):
                    red = int(match.group(1))
                    green = int(match.group(2))
                    blue = int(match.group(3))
                    
                    colors.append({
                        "id": i + 1,  # Цвет 0 - авто, поэтому начинаем с 1
                        "rgb": [red, green, blue],
                        "hex": f"#{red:02x}{green:02x}{blue:02x}"
                    })
                
                if colors:
                    formatting["colors"] = colors
            
            # Извлекаем информацию о настройках страницы
            # Ширина страницы: \paperw
            # Высота страницы: \paperh
            # Ширина полей: \margl, \margr, \margt, \margb
            
            page_width_match = re.search(r"\\paperw(\d+)", content)
            if page_width_match:
                # В RTF размеры указываются в твипах (1/20 пункта, 1/1440 дюйма)
                formatting["page_width_twips"] = int(page_width_match.group(1))
                formatting["page_width_inches"] = formatting["page_width_twips"] / 1440
                formatting["page_width_cm"] = formatting["page_width_inches"] * 2.54
            
            page_height_match = re.search(r"\\paperh(\d+)", content)
            if page_height_match:
                formatting["page_height_twips"] = int(page_height_match.group(1))
                formatting["page_height_inches"] = formatting["page_height_twips"] / 1440
                formatting["page_height_cm"] = formatting["page_height_inches"] * 2.54
            
            # Извлекаем информацию о полях
            margins = {}
            
            left_margin_match = re.search(r"\\margl(\d+)", content)
            if left_margin_match:
                margins["left_twips"] = int(left_margin_match.group(1))
                margins["left_inches"] = margins["left_twips"] / 1440
                margins["left_cm"] = margins["left_inches"] * 2.54
            
            right_margin_match = re.search(r"\\margr(\d+)", content)
            if right_margin_match:
                margins["right_twips"] = int(right_margin_match.group(1))
                margins["right_inches"] = margins["right_twips"] / 1440
                margins["right_cm"] = margins["right_inches"] * 2.54
            
            top_margin_match = re.search(r"\\margt(\d+)", content)
            if top_margin_match:
                margins["top_twips"] = int(top_margin_match.group(1))
                margins["top_inches"] = margins["top_twips"] / 1440
                margins["top_cm"] = margins["top_inches"] * 2.54
            
            bottom_margin_match = re.search(r"\\margb(\d+)", content)
            if bottom_margin_match:
                margins["bottom_twips"] = int(bottom_margin_match.group(1))
                margins["bottom_inches"] = margins["bottom_twips"] / 1440
                margins["bottom_cm"] = margins["bottom_inches"] * 2.54
            
            if margins:
                formatting["margins"] = margins
            
            return formatting
        except Exception as e:
            logger.warning(f"Error extracting formatting from RTF: {e}")
            return formatting
    
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

ParserFactory.register_parser(RTFParser)