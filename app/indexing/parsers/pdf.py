import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import re

from app.indexing.parsers.base import BaseParser, ParsedDocument
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PDFParser(BaseParser):
    """Парсер для PDF-файлов."""
    
    def __init__(self):
        """Инициализация парсера PDF-файлов."""
        super().__init__()
        self.supported_extensions = [".pdf"]
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает PDF-файл и возвращает его содержимое и метаданные.
        
        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - extract_images: Извлекать ли изображения (по умолчанию False)
                - ocr_enabled: Использовать ли OCR, если текст не обнаружен (по умолчанию True)
                - ocr_language: Язык OCR (по умолчанию из настроек)
                - extract_tables: Извлекать ли таблицы (по умолчанию False)
                - password: Пароль для защищенных PDF (по умолчанию None)
                - page_range: Диапазон страниц для обработки (по умолчанию все)
            
        Returns:
            Объект ParsedDocument с содержимым и метаданными
        """
        try:
            import fitz  # PyMuPDF
            
            # Получаем метаданные файла
            metadata = self.extract_metadata(file_path)
            
            # Определяем параметры
            chunk_size = kwargs.get("chunk_size", settings.processing.chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.processing.chunk_overlap)
            extract_images = kwargs.get("extract_images", False)
            ocr_enabled = kwargs.get("ocr_enabled", True)
            ocr_language = kwargs.get("ocr_language", settings.ocr.language)
            extract_tables = kwargs.get("extract_tables", False)
            password = kwargs.get("password", None)
            page_range = kwargs.get("page_range", None)
            
            # Открываем PDF-файл
            if password:
                pdf_document = fitz.open(file_path, password=password)
            else:
                pdf_document = fitz.open(file_path)
            
            # Проверяем, удалось ли открыть документ
            if pdf_document.is_encrypted and not pdf_document.is_open:
                raise ValueError("PDF document is encrypted and could not be opened with the provided password")
            
            # Определяем диапазон страниц
            if page_range:
                start_page, end_page = self._parse_page_range(page_range, pdf_document.page_count)
            else:
                start_page, end_page = 0, pdf_document.page_count - 1
            
            # Извлекаем метаданные PDF
            pdf_metadata = self._extract_pdf_metadata(pdf_document)
            metadata.update(pdf_metadata)
            
            # Извлекаем текст и другие данные из PDF
            content, doc_metadata = self._extract_pdf_content(
                pdf_document,
                start_page=start_page,
                end_page=end_page,
                extract_images=extract_images,
                ocr_enabled=ocr_enabled,
                ocr_language=ocr_language,
                extract_tables=extract_tables
            )
            
            # Обновляем метаданные
            metadata.update(doc_metadata)
            
            # Добавляем информацию о размере содержимого
            metadata["content_length"] = len(content)
            metadata["content_type"] = "application/pdf"
            
            # Разбиваем содержимое на чанки
            # Временно сохраняем текущие значения и устанавливаем новые
            original_chunk_size = self.chunk_size
            original_chunk_overlap = self.chunk_overlap
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            
            # Разбиваем содержимое на чанки
            chunks = self.chunk_text(content)
            
            # Восстанавливаем оригинальные значения
            self.chunk_size = original_chunk_size
            self.chunk_overlap = original_chunk_overlap
            
            # Закрываем PDF-документ
            pdf_document.close()
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                chunks=chunks
            )
        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            raise ImportError(f"Required libraries not installed: {e}")
        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {e}")
            raise
    
    def _parse_page_range(self, page_range: str, total_pages: int) -> Tuple[int, int]:
        """
        Разбирает строку диапазона страниц.
        
        Args:
            page_range: Строка диапазона страниц (например, "1-5" или "3")
            total_pages: Общее количество страниц
            
        Returns:
            Кортеж (начальная страница, конечная страница) с индексами от 0
        """
        try:
            if "-" in page_range:
                start, end = page_range.split("-", 1)
                start_page = max(0, int(start) - 1) if start else 0
                end_page = min(int(end) - 1, total_pages - 1) if end else total_pages - 1
            else:
                page = int(page_range) - 1
                start_page = max(0, page)
                end_page = start_page
            
            # Проверяем, что диапазон правильный
            if start_page > end_page or start_page >= total_pages or end_page < 0:
                logger.warning(f"Invalid page range {page_range} for document with {total_pages} pages")
                return 0, total_pages - 1
            
            return start_page, end_page
        except ValueError:
            logger.warning(f"Invalid page range format: {page_range}")
            return 0, total_pages - 1
    
    def _extract_pdf_metadata(self, pdf_document) -> Dict[str, Any]:
        """
        Извлекает метаданные из PDF-документа.
        
        Args:
            pdf_document: Объект PDF-документа
            
        Returns:
            Словарь с метаданными
        """
        metadata = {}
        
        try:
            # Базовые метаданные
            metadata["page_count"] = pdf_document.page_count
            metadata["form_fields"] = bool(pdf_document.is_form_pdf)
            metadata["is_encrypted"] = bool(pdf_document.is_encrypted)
            metadata["is_repaired"] = bool(pdf_document.is_repaired)
            metadata["needs_repair"] = bool(pdf_document.needs_repair)
            
            # Добавляем стандартные метаданные PDF
            standard_metadata = {
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "creator": None,
                "producer": None,
                "creation_date": None,
                "modification_date": None
            }
            
            # Извлекаем метаданные из документа
            for key, value in pdf_document.metadata.items():
                if key.lower() in standard_metadata:
                    # Нормализуем ключи
                    normalized_key = key.lower()
                    
                    # Преобразуем даты в ISO формат
                    if "date" in normalized_key and value:
                        try:
                            # Попытка преобразовать дату из PDF формата в ISO
                            if isinstance(value, str) and "D:" in value:
                                # Формат PDF: D:YYYYMMDDHHmmSSOHH'mm'
                                # Извлекаем компоненты даты
                                date_str = value.replace("D:", "")
                                
                                # Извлекаем основные компоненты
                                year = date_str[0:4]
                                month = date_str[4:6]
                                day = date_str[6:8]
                                
                                # Если есть время, извлекаем его
                                if len(date_str) >= 14:
                                    hour = date_str[8:10]
                                    minute = date_str[10:12]
                                    second = date_str[12:14]
                                    value = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
                                else:
                                    value = f"{year}-{month}-{day}"
                        except Exception as e:
                            logger.warning(f"Error parsing PDF date {value}: {e}")
                    
                    # Сохраняем нормализованное значение
                    metadata[normalized_key] = value
            
            # Добавляем информацию о версии PDF
            metadata["pdf_version"] = f"{pdf_document.pdf_version:.1f}"
            
            # Извлекаем информацию о шрифтах
            fonts = set()
            for page_idx in range(min(3, pdf_document.page_count)):  # Проверяем только первые 3 страницы
                page = pdf_document[page_idx]
                for font in page.get_fonts():
                    font_name = font[3] if len(font) > 3 else "Unknown"
                    fonts.add(font_name)
            
            metadata["fonts"] = list(fonts)
            
            # Проверяем наличие защиты от копирования
            metadata["allow_copy"] = pdf_document.permissions & 4 != 0
            metadata["allow_print"] = pdf_document.permissions & 4 != 0
            
            return metadata
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
            return metadata
    
    def _extract_pdf_content(
        self,
        pdf_document,
        start_page: int = 0,
        end_page: int = None,
        extract_images: bool = False,
        ocr_enabled: bool = True,
        ocr_language: str = "rus+eng",
        extract_tables: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Извлекает содержимое и дополнительные метаданные из PDF-документа.
        
        Args:
            pdf_document: Объект PDF-документа
            start_page: Начальная страница (индекс от 0)
            end_page: Конечная страница (индекс от 0)
            extract_images: Извлекать ли изображения
            ocr_enabled: Использовать ли OCR, если текст не обнаружен
            ocr_language: Язык OCR
            extract_tables: Извлекать ли таблицы
            
        Returns:
            Кортеж (содержимое, метаданные)
        """
        if end_page is None:
            end_page = pdf_document.page_count - 1
        
        text_parts = []
        metadata = {
            "pages_processed": end_page - start_page + 1,
            "pages_with_text": 0,
            "pages_with_images": 0,
            "total_images": 0,
            "pages_with_tables": 0,
            "total_tables": 0,
            "ocr_used": False
        }
        
        # Обрабатываем каждую страницу
        for page_idx in range(start_page, end_page + 1):
            page = pdf_document[page_idx]
            page_number = page_idx + 1
            
            # Извлекаем текст со страницы
            page_text = page.get_text()
            
            # Если текст не обнаружен и OCR включен, пытаемся использовать OCR
            if not page_text.strip() and ocr_enabled:
                try:
                    page_text = self._extract_text_with_ocr(page, language=ocr_language)
                    metadata["ocr_used"] = True
                except Exception as e:
                    logger.warning(f"Error using OCR on page {page_number}: {e}")
            
            # Если есть текст, увеличиваем счетчик
            if page_text.strip():
                metadata["pages_with_text"] += 1
            
            # Добавляем текст страницы
            text_parts.append(f"--- Страница {page_number} ---\n{page_text}")
            
            # Извлекаем изображения, если требуется
            if extract_images:
                try:
                    image_count = self._extract_images_from_page(page, page_number)
                    if image_count > 0:
                        metadata["pages_with_images"] += 1
                        metadata["total_images"] += image_count
                except Exception as e:
                    logger.warning(f"Error extracting images from page {page_number}: {e}")
            
            # Извлекаем таблицы, если требуется
            if extract_tables:
                try:
                    table_count, tables_text = self._extract_tables_from_page(page, page_number)
                    if table_count > 0:
                        metadata["pages_with_tables"] += 1
                        metadata["total_tables"] += table_count
                        text_parts.append(tables_text)
                except Exception as e:
                    logger.warning(f"Error extracting tables from page {page_number}: {e}")
        
        # Объединяем текст
        content = "\n\n".join(text_parts)
        
        # Анализируем содержимое
        content_metadata = self._analyze_pdf_content(content)
        metadata.update(content_metadata)
        
        return content, metadata
    
    def _extract_text_with_ocr(self, page, language: str = "rus+eng") -> str:
        """
        Извлекает текст из страницы с помощью OCR.
        
        Args:
            page: Страница PDF-документа
            language: Язык OCR
            
        Returns:
            Извлеченный текст
        """
        try:
            import pytesseract
            from PIL import Image
            import numpy as np
            
            # Получаем изображение страницы
            pix = page.get_pixmap(alpha=False)
            
            # Преобразуем в формат, подходящий для Pillow
            mode = "RGB" if pix.n >= 3 else "L"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            
            # Настраиваем Tesseract
            pytesseract.pytesseract.tesseract_cmd = self._get_tesseract_path()
            
            # Опции OCR
            custom_config = f'--oem 3 --psm 6 -l {language}'
            
            # Извлекаем текст с помощью OCR
            text = pytesseract.image_to_string(img, config=custom_config)
            
            return text
        except ImportError:
            logger.error("pytesseract or pillow not installed")
            raise ImportError("pytesseract and pillow are required for OCR")
        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
            raise
    
    def _extract_images_from_page(self, page, page_number: int) -> int:
        """
        Извлекает изображения со страницы PDF.
        
        Args:
            page: Страница PDF-документа
            page_number: Номер страницы
            
        Returns:
            Количество извлеченных изображений
        """
        # Данная функция в основном заготовка для реализации извлечения изображений
        # Полная реализация потребует сохранения изображений и их дальнейшей обработки
        
        # Получаем список изображений на странице
        image_list = page.get_images(full=True)
        
        return len(image_list)
    
    def _extract_tables_from_page(self, page, page_number: int) -> Tuple[int, str]:
        """
        Извлекает таблицы со страницы PDF.
        
        Args:
            page: Страница PDF-документа
            page_number: Номер страницы
            
        Returns:
            Кортеж (количество извлеченных таблиц, текст таблиц)
        """
        try:
            # Для определения таблиц требуется дополнительная обработка
            # В этой заготовке используется упрощенный алгоритм
            
            # Получаем текст страницы
            page_text = page.get_text()
            
            # Ищем таблицы по характерным признакам (наличие нескольких пробелов в строке)
            lines = page_text.split("\n")
            table_lines = []
            in_table = False
            table_count = 0
            
            for line in lines:
                # Проверяем, похожа ли строка на строку таблицы
                # (несколько слов, разделенных множественными пробелами)
                if re.search(r"\S+\s{2,}\S+", line):
                    if not in_table:
                        in_table = True
                        table_count += 1
                        table_lines.append(f"--- Таблица {table_count} на странице {page_number} ---")
                    
                    # Форматируем строку таблицы
                    formatted_line = re.sub(r"\s{2,}", " | ", line)
                    table_lines.append(formatted_line)
                elif in_table and line.strip():
                    # Если мы все еще в таблице и строка не пустая, добавляем ее
                    table_lines.append(line)
                elif in_table:
                    # Если мы в таблице, но строка пустая, завершаем таблицу
                    in_table = False
                    table_lines.append("")
            
            # Объединяем текст таблиц
            tables_text = "\n".join(table_lines)
            
            return table_count, tables_text
        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_number}: {e}")
            return 0, ""
    
    def _analyze_pdf_content(self, content: str) -> Dict[str, Any]:
        """
        Анализирует содержимое PDF и возвращает метаданные.
        
        Args:
            content: Текстовое содержимое PDF
            
        Returns:
            Словарь с метаданными
        """
        metadata = {}
        
        try:
            # Подсчитываем статистику
            words = re.findall(r"\b\w+\b", content)
            metadata["word_count"] = len(words)
            metadata["char_count"] = len(content)
            
            # Определяем язык текста (простая эвристика)
            metadata["language"] = self._detect_language(content)
            
            # Находим возможные заголовки
            headings = []
            lines = content.split("\n")
            
            for i, line in enumerate(lines):
                line = line.strip()
                # Ищем короткие строки с большими буквами, которые могут быть заголовками
                if line and len(line) < 100 and line.isupper():
                    headings.append(line)
                # Ищем строки, которые могут быть заголовками глав или разделов
                elif re.match(r"^(Chapter|Глава|Раздел|Section)\s+\d+", line, re.IGNORECASE):
                    headings.append(line)
            
            metadata["possible_headings"] = headings[:10]  # Ограничиваем количество заголовков
            metadata["heading_count"] = len(headings)
            
            return metadata
        except Exception as e:
            logger.warning(f"Error analyzing PDF content: {e}")
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
    
    def _get_tesseract_path(self) -> str:
        """
        Получает путь к исполняемому файлу Tesseract OCR.
        
        Returns:
            Путь к Tesseract
        """
        # Пытаемся найти tesseract в разных местах
        tesseract_cmd = 'tesseract'
        
        # Типичные пути для разных ОС
        possible_paths = [
            'tesseract',  # в PATH
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
            'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',
        ]
        
        # Проверяем, настроен ли путь в конфигурации
        if hasattr(settings, 'ocr') and hasattr(settings.ocr, 'tesseract_path') and settings.ocr.tesseract_path:
            return settings.ocr.tesseract_path
        
        # Проверяем возможные пути
        for path in possible_paths:
            try:
                # Проверяем, существует ли файл
                if os.path.isfile(path):
                    return path
                
                # Проверяем, доступен ли путь в PATH
                import subprocess
                subprocess.run([path, '--version'], capture_output=True, check=True)
                return path
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
        
        # Возвращаем стандартный путь
        return tesseract_cmd


# Регистрируем парсер в фабрике
from app.indexing.parsers.base import ParserFactory

ParserFactory.register_parser(PDFParser)