import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime
from urllib.parse import urljoin
import json

from app.indexing.parsers.base import BaseParser, ParsedDocument
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class WebParser(BaseParser):
    """Парсер для веб-данных (HTML, Markdown)."""
    
    def __init__(self):
        """Инициализация парсера веб-данных."""
        super().__init__()
        self.supported_extensions = [
            ".html", ".htm", ".xhtml", ".md", ".markdown"
        ]
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает веб-файл и возвращает его содержимое и метаданные.
        
        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - encoding: Кодировка файла (по умолчанию автоопределение)
                - extract_links: Извлекать ли ссылки (по умолчанию True)
                - extract_images: Извлекать ли изображения (по умолчанию True)
                - extract_tables: Извлекать ли таблицы (по умолчанию True)
                - extract_metadata: Извлекать ли метаданные (по умолчанию True)
                - preserve_structure: Сохранять ли структуру документа (по умолчанию True)
                - include_html: Включать ли исходный HTML (по умолчанию False)
                - base_url: Базовый URL для относительных ссылок (по умолчанию None)
                - markdown_extensions: Расширения для Markdown (по умолчанию ["tables", "fenced_code"])
            
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
            extract_links = kwargs.get("extract_links", True)
            extract_images = kwargs.get("extract_images", True)
            extract_tables = kwargs.get("extract_tables", True)
            extract_metadata = kwargs.get("extract_metadata", True)
            preserve_structure = kwargs.get("preserve_structure", True)
            include_html = kwargs.get("include_html", False)
            base_url = kwargs.get("base_url", None)
            markdown_extensions = kwargs.get("markdown_extensions", ["tables", "fenced_code"])
            
            # Получаем расширение файла
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # Читаем содержимое файла
            with open(file_path, 'r', encoding=encoding) as f:
                raw_content = f.read()
            
            # Обрабатываем файл в зависимости от его типа
            if ext in [".html", ".htm", ".xhtml"]:
                # Обрабатываем HTML
                content, html_metadata = self._process_html(
                    raw_content,
                    file_path=file_path,
                    extract_links=extract_links,
                    extract_images=extract_images,
                    extract_tables=extract_tables,
                    extract_metadata=extract_metadata,
                    preserve_structure=preserve_structure,
                    base_url=base_url
                )
                metadata.update(html_metadata)
                
                # Включаем исходный HTML, если требуется
                if include_html:
                    metadata["html"] = raw_content
            elif ext in [".md", ".markdown"]:
                # Обрабатываем Markdown
                content, md_metadata = self._process_markdown(
                    raw_content,
                    file_path=file_path,
                    extract_links=extract_links,
                    extract_images=extract_images,
                    extract_tables=extract_tables,
                    preserve_structure=preserve_structure,
                    markdown_extensions=markdown_extensions,
                    base_url=base_url
                )
                metadata.update(md_metadata)
                
                # Включаем исходный Markdown, если требуется
                if include_html:
                    metadata["markdown"] = raw_content
            else:
                # Обрабатываем как обычный текст
                content = raw_content
                metadata["format"] = "text"
            
            # Добавляем информацию о размере содержимого и типе
            metadata["content_length"] = len(content)
            
            if ext in [".html", ".htm", ".xhtml"]:
                metadata["content_type"] = "text/html"
            elif ext in [".md", ".markdown"]:
                metadata["content_type"] = "text/markdown"
            
            # Разбиваем содержимое на чанки с учетом структуры документа
            if preserve_structure and (metadata.get("headings") or metadata.get("structure")):
                chunks = self._chunk_structured_text(
                    content,
                    metadata.get("headings", []),
                    metadata.get("structure", {}),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
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
            logger.error(f"Error parsing web file {file_path}: {e}")
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
    
    def _process_html(
        self,
        content: str,
        file_path: str,
        extract_links: bool = True,
        extract_images: bool = True,
        extract_tables: bool = True,
        extract_metadata: bool = True,
        preserve_structure: bool = True,
        base_url: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Обрабатывает HTML-содержимое.
        
        Args:
            content: Содержимое файла
            file_path: Путь к файлу
            extract_links: Извлекать ли ссылки
            extract_images: Извлекать ли изображения
            extract_tables: Извлекать ли таблицы
            extract_metadata: Извлекать ли метаданные
            preserve_structure: Сохранять ли структуру документа
            base_url: Базовый URL для относительных ссылок
            
        Returns:
            Кортеж (обработанное содержимое, метаданные)
        """
        try:
            from bs4 import BeautifulSoup
            
            # Создаем метаданные
            metadata = {
                "format": "html",
                "size": len(content)
            }
            
            # Разбираем HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Извлекаем метаданные, если требуется
            if extract_metadata:
                html_metadata = self._extract_html_metadata(soup)
                metadata.update(html_metadata)
            
            # Извлекаем ссылки, если требуется
            if extract_links:
                links = self._extract_html_links(soup, base_url=base_url)
                metadata["links"] = links
                metadata["link_count"] = len(links)
            
            # Извлекаем изображения, если требуется
            if extract_images:
                images = self._extract_html_images(soup, base_url=base_url)
                metadata["images"] = images
                metadata["image_count"] = len(images)
            
            # Извлекаем таблицы, если требуется
            if extract_tables:
                tables = self._extract_html_tables(soup)
                metadata["tables"] = tables
                metadata["table_count"] = len(tables)
            
            # Сохраняем структуру документа, если требуется
            if preserve_structure:
                headings = self._extract_html_headings(soup)
                metadata["headings"] = headings
                metadata["heading_count"] = len(headings)
                
                # Создаем структуру документа
                structure = self._create_document_structure(headings)
                metadata["structure"] = structure
            
            # Извлекаем текст с сохранением структуры
            processed_content = self._extract_html_text(soup, preserve_structure=preserve_structure)
            
            # Определяем язык
            metadata["language"] = self._detect_language(processed_content)
            
            return processed_content, metadata
        except ImportError:
            logger.error("BeautifulSoup4 library not installed")
            raise ImportError("BeautifulSoup4 is required for parsing HTML files")
        except Exception as e:
            logger.warning(f"Error processing HTML: {e}")
            return content, {"format": "html", "error": str(e)}
    
    def _process_markdown(
        self,
        content: str,
        file_path: str,
        extract_links: bool = True,
        extract_images: bool = True,
        extract_tables: bool = True,
        preserve_structure: bool = True,
        markdown_extensions: List[str] = ["tables", "fenced_code"],
        base_url: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Обрабатывает Markdown-содержимое.
        
        Args:
            content: Содержимое файла
            file_path: Путь к файлу
            extract_links: Извлекать ли ссылки
            extract_images: Извлекать ли изображения
            extract_tables: Извлекать ли таблицы
            preserve_structure: Сохранять ли структуру документа
            markdown_extensions: Расширения для Markdown
            base_url: Базовый URL для относительных ссылок
            
        Returns:
            Кортеж (обработанное содержимое, метаданные)
        """
        try:
            # Создаем метаданные
            metadata = {
                "format": "markdown",
                "size": len(content)
            }
            
            # Пытаемся преобразовать Markdown в HTML
            try:
                import markdown
                
                # Преобразуем Markdown в HTML
                html_content = markdown.markdown(content, extensions=markdown_extensions)
                
                # Обрабатываем полученный HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Извлекаем ссылки, если требуется
                if extract_links:
                    links = self._extract_html_links(soup, base_url=base_url)
                    metadata["links"] = links
                    metadata["link_count"] = len(links)
                
                # Извлекаем изображения, если требуется
                if extract_images:
                    images = self._extract_html_images(soup, base_url=base_url)
                    metadata["images"] = images
                    metadata["image_count"] = len(images)
                
                # Извлекаем таблицы, если требуется
                if extract_tables and "tables" in markdown_extensions:
                    tables = self._extract_html_tables(soup)
                    metadata["tables"] = tables
                    metadata["table_count"] = len(tables)
                
                # Сохраняем структуру документа, если требуется
                if preserve_structure:
                    headings = self._extract_html_headings(soup)
                    metadata["headings"] = headings
                    metadata["heading_count"] = len(headings)
                    
                    # Создаем структуру документа
                    structure = self._create_document_structure(headings)
                    metadata["structure"] = structure
                
                # Извлекаем текст
                processed_content = self._extract_html_text(soup, preserve_structure=preserve_structure)
            except ImportError:
                # Если нет библиотеки markdown, обрабатываем текст напрямую
                logger.warning("Python-Markdown library not installed, parsing Markdown manually")
                
                # Извлекаем заголовки
                headings = self._extract_markdown_headings(content)
                metadata["headings"] = headings
                metadata["heading_count"] = len(headings)
                
                # Создаем структуру документа
                structure = self._create_document_structure(headings)
                metadata["structure"] = structure
                
                # Извлекаем ссылки
                if extract_links:
                    links = self._extract_markdown_links(content, base_url=base_url)
                    metadata["links"] = links
                    metadata["link_count"] = len(links)
                
                # Извлекаем изображения
                if extract_images:
                    images = self._extract_markdown_images(content, base_url=base_url)
                    metadata["images"] = images
                    metadata["image_count"] = len(images)
                
                # Используем исходный текст
                processed_content = content
            
            # Определяем язык
            metadata["language"] = self._detect_language(processed_content)
            
            return processed_content, metadata
        except Exception as e:
            logger.warning(f"Error processing Markdown: {e}")
            return content, {"format": "markdown", "error": str(e)}
    
    def _extract_html_metadata(self, soup) -> Dict[str, Any]:
        """
        Извлекает метаданные из HTML-документа.
        
        Args:
            soup: Объект BeautifulSoup
            
        Returns:
            Словарь с метаданными
        """
        metadata = {}
        
        try:
            # Извлекаем заголовок
            title_tag = soup.find('title')
            if title_tag:
                metadata["title"] = title_tag.text.strip()
            
            # Извлекаем мета-теги
            meta_tags = {}
            
            for meta in soup.find_all('meta'):
                # Извлекаем имя и содержимое
                name = meta.get('name', meta.get('property', ''))
                content = meta.get('content', '')
                
                if name and content:
                    meta_tags[name] = content
            
            # Добавляем важные мета-теги
            if meta_tags:
                metadata["meta_tags"] = meta_tags
                
                # Добавляем описание, если есть
                if 'description' in meta_tags:
                    metadata["description"] = meta_tags['description']
                
                # Добавляем ключевые слова, если есть
                if 'keywords' in meta_tags:
                    keywords = meta_tags['keywords'].split(',')
                    metadata["keywords"] = [k.strip() for k in keywords if k.strip()]
                
                # Добавляем автора, если есть
                if 'author' in meta_tags:
                    metadata["author"] = meta_tags['author']
                
                # Добавляем Open Graph метаданные
                og_tags = {k.replace('og:', ''): v for k, v in meta_tags.items() if k.startswith('og:')}
                if og_tags:
                    metadata["open_graph"] = og_tags
                
                # Добавляем Twitter метаданные
                twitter_tags = {k.replace('twitter:', ''): v for k, v in meta_tags.items() if k.startswith('twitter:')}
                if twitter_tags:
                    metadata["twitter"] = twitter_tags
            
            # Извлекаем канонический URL
            canonical = soup.find('link', rel='canonical')
            if canonical and canonical.get('href'):
                metadata["canonical_url"] = canonical['href']
            
            # Извлекаем язык
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                metadata["html_lang"] = html_tag['lang']
            
            return metadata
        except Exception as e:
            logger.warning(f"Error extracting HTML metadata: {e}")
            return metadata
    
    def _extract_html_links(self, soup, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Извлекает ссылки из HTML-документа.
        
        Args:
            soup: Объект BeautifulSoup
            base_url: Базовый URL для относительных ссылок
            
        Returns:
            Список словарей с информацией о ссылках
        """
        links = []
        
        try:
            for a in soup.find_all('a', href=True):
                href = a['href']
                
                # Пропускаем пустые ссылки и якоря
                if not href or href.startswith('#'):
                    continue
                
                # Преобразуем относительные ссылки в абсолютные
                if base_url and not (href.startswith('http://') or href.startswith('https://')):
                    href = urljoin(base_url, href)
                
                # Создаем словарь с информацией о ссылке
                link_info = {
                    "href": href,
                    "text": a.text.strip()
                }
                
                # Добавляем заголовок, если есть
                if a.get('title'):
                    link_info["title"] = a['title']
                
                # Определяем, является ли ссылка внешней
                if base_url and href.startswith(('http://', 'https://')):
                    link_info["is_external"] = not href.startswith(base_url)
                
                links.append(link_info)
            
            return links
        except Exception as e:
            logger.warning(f"Error extracting HTML links: {e}")
            return links
    
    def _extract_html_images(self, soup, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Извлекает изображения из HTML-документа.
        
        Args:
            soup: Объект BeautifulSoup
            base_url: Базовый URL для относительных ссылок
            
        Returns:
            Список словарей с информацией об изображениях
        """
        images = []
        
        try:
            for img in soup.find_all('img'):
                src = img.get('src', '')
                
                # Пропускаем пустые источники
                if not src:
                    continue
                
                # Преобразуем относительные ссылки в абсолютные
                if base_url and not (src.startswith('http://') or src.startswith('https://')):
                    src = urljoin(base_url, src)
                
                # Создаем словарь с информацией об изображении
                image_info = {
                    "src": src
                }
                
                # Добавляем альтернативный текст, если есть
                if img.get('alt'):
                    image_info["alt"] = img['alt']
                
                # Добавляем заголовок, если есть
                if img.get('title'):
                    image_info["title"] = img['title']
                
                # Добавляем размеры, если есть
                if img.get('width'):
                    image_info["width"] = img['width']
                
                if img.get('height'):
                    image_info["height"] = img['height']
                
                images.append(image_info)
            
            return images
        except Exception as e:
            logger.warning(f"Error extracting HTML images: {e}")
            return images
    
    def _extract_html_tables(self, soup) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из HTML-документа.
        
        Args:
            soup: Объект BeautifulSoup
            
        Returns:
            Список словарей с информацией о таблицах
        """
        tables = []
        
        try:
            for i, table in enumerate(soup.find_all('table')):
                # Создаем словарь с информацией о таблице
                table_info = {
                    "index": i,
                    "caption": "",
                    "headers": [],
                    "rows": []
                }
                
                # Извлекаем заголовок таблицы
                caption = table.find('caption')
                if caption:
                    table_info["caption"] = caption.text.strip()
                
                # Извлекаем заголовки
                headers = []
                for th in table.find_all('th'):
                    headers.append(th.text.strip())
                
                table_info["headers"] = headers
                
                # Извлекаем строки
                for tr in table.find_all('tr'):
                    row = []
                    for td in tr.find_all(['td', 'th']):
                        row.append(td.text.strip())
                    
                    if row and not (len(row) == 1 and not row[0]):
                        table_info["rows"].append(row)
                
                # Добавляем таблицу только если в ней есть данные
                if table_info["rows"]:
                    tables.append(table_info)
            
            return tables
        except Exception as e:
            logger.warning(f"Error extracting HTML tables: {e}")
            return tables
    
    def _extract_html_headings(self, soup) -> List[Dict[str, Any]]:
        """
        Извлекает заголовки из HTML-документа.
        
        Args:
            soup: Объект BeautifulSoup
            
        Returns:
            Список словарей с информацией о заголовках
        """
        headings = []
        
        try:
            for i, h in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
                # Определяем уровень заголовка
                level = int(h.name[1])
                
                # Создаем словарь с информацией о заголовке
                heading_info = {
                    "text": h.text.strip(),
                    "level": level,
                    "index": i
                }
                
                # Добавляем id, если есть
                if h.get('id'):
                    heading_info["id"] = h['id']
                
                headings.append(heading_info)
            
            return headings
        except Exception as e:
            logger.warning(f"Error extracting HTML headings: {e}")
            return headings
    
    def _extract_html_text(self, soup, preserve_structure: bool = True) -> str:
        """
        Извлекает текст из HTML-документа с сохранением структуры.
        
        Args:
            soup: Объект BeautifulSoup
            preserve_structure: Сохранять ли структуру документа
            
        Returns:
            Извлеченный текст
        """
        try:
            if not preserve_structure:
                # Просто извлекаем весь текст
                return soup.get_text(' ', strip=True)
            
            # Копируем объект супа для модификации
            soup_copy = BeautifulSoup(str(soup), 'html.parser')
            
            # Заменяем теги заголовков на текст с символами #
            for i in range(1, 7):
                for h in soup_copy.find_all(f'h{i}'):
                    h.replace_with(f"{'#' * i} {h.text.strip()}\n\n")
            
            # Заменяем теги параграфов и дивов на текст с переносами строк
            for tag in soup_copy.find_all(['p', 'div']):
                tag.replace_with(f"{tag.text.strip()}\n\n")
            
            # Заменяем теги списков
            for ul in soup_copy.find_all('ul'):
                items = []
                for li in ul.find_all('li'):
                    items.append(f"* {li.text.strip()}")
                ul.replace_with('\n'.join(items) + '\n\n')
            
            for ol in soup_copy.find_all('ol'):
                items = []
                for i, li in enumerate(ol.find_all('li')):
                    items.append(f"{i+1}. {li.text.strip()}")
                ol.replace_with('\n'.join(items) + '\n\n')
            
            # Заменяем теги таблиц
            for table in soup_copy.find_all('table'):
                rows = []
                
                # Добавляем заголовок таблицы, если есть
                caption = table.find('caption')
                if caption:
                    rows.append(f"Таблица: {caption.text.strip()}")
                
                # Добавляем строки таблицы
                for tr in table.find_all('tr'):
                    cells = []
                    for td in tr.find_all(['td', 'th']):
                        cells.append(td.text.strip())
                    
                    if cells:
                        rows.append(' | '.join(cells))
                
                if rows:
                    table.replace_with('\n'.join(rows) + '\n\n')
            
            # Заменяем теги code
            for code in soup_copy.find_all('code'):
                code.replace_with(f"`{code.text.strip()}`")
            
            # Заменяем теги pre
            for pre in soup_copy.find_all('pre'):
                pre.replace_with(f"```\n{pre.text.strip()}\n```\n\n")
            
            # Заменяем теги blockquote
            for quote in soup_copy.find_all('blockquote'):
                lines = []
                for line in quote.text.strip().split('\n'):
                    lines.append(f"> {line}")
                quote.replace_with('\n'.join(lines) + '\n\n')
            
            # Заменяем теги br на переносы строк
            for br in soup_copy.find_all('br'):
                br.replace_with('\n')
            
            # Получаем текст
            text = soup_copy.get_text(' ', strip=True)
            
            # Нормализуем пробелы и переносы строк
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s+\n', '\n\n', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text.strip()
        except Exception as e:
            logger.warning(f"Error extracting HTML text: {e}")
            return soup.get_text(' ', strip=True)
    
    def _extract_markdown_headings(self, content: str) -> List[Dict[str, Any]]:
        """
        Извлекает заголовки из Markdown-документа.
        
        Args:
            content: Содержимое Markdown-документа
            
        Returns:
            Список словарей с информацией о заголовках
        """
        headings = []
        
        try:
            # Ищем заголовки в формате # Heading
            atx_heading_pattern = r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?$'
            
            # Обрабатываем строки
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # Проверяем ATX-заголовки (# Heading)
                atx_match = re.match(atx_heading_pattern, line.strip())
                if atx_match:
                    level = len(atx_match.group(1))
                    text = atx_match.group(2).strip()
                    
                    headings.append({
                        "text": text,
                        "level": level,
                        "index": i
                    })
                    continue
                
                # Проверяем Setext-заголовки (Heading\n======)
                if i < len(lines) - 1:
                    next_line = lines[i + 1].strip()
                    if re.match(r'^=+$', next_line):
                        # Заголовок уровня 1
                        headings.append({
                            "text": line.strip(),
                            "level": 1,
                            "index": i
                        })
                    elif re.match(r'^-+$', next_line):
                        # Заголовок уровня 2
                        headings.append({
                            "text": line.strip(),
                            "level": 2,
                            "index": i
                        })
            
            return headings
        except Exception as e:
            logger.warning(f"Error extracting Markdown headings: {e}")
            return headings
    
    def _extract_markdown_links(self, content: str, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Извлекает ссылки из Markdown-документа.
        
        Args:
            content: Содержимое Markdown-документа
            base_url: Базовый URL для относительных ссылок
            
        Returns:
            Список словарей с информацией о ссылках
        """
        links = []
        
        try:
            # Ищем ссылки в формате [text](url "title")
            inline_link_pattern = r'\[([^\]]+)\]\(([^)]+?)(?:\s+"([^"]+)")?\)'
            
            # Находим все совпадения
            for match in re.finditer(inline_link_pattern, content):
                text = match.group(1)
                href = match.group(2)
                title = match.group(3) if match.group(3) else None
                
                # Пропускаем пустые ссылки и якоря
                if not href or href.startswith('#'):
                    continue
                
                # Преобразуем относительные ссылки в абсолютные
                if base_url and not (href.startswith('http://') or href.startswith('https://')):
                    href = urljoin(base_url, href)
                
                # Создаем словарь с информацией о ссылке
                link_info = {
                    "href": href,
                    "text": text
                }
                
                # Добавляем заголовок, если есть
                if title:
                    link_info["title"] = title
                
                # Определяем, является ли ссылка внешней
                if base_url and href.startswith(('http://', 'https://')):
                    link_info["is_external"] = not href.startswith(base_url)
                
                links.append(link_info)
            
            # Ищем ссылки в формате [text][ref] и [ref]: url "title"
            reference_link_pattern = r'\[([^\]]+)\]\[([^\]]*)\]'
            reference_def_pattern = r'^\[([^\]]+)\]:\s*([^ ]+)(?:\s+"([^"]+)")?$'
            
            # Находим все определения ссылок
            reference_defs = {}
            for line in content.split('\n'):
                match = re.match(reference_def_pattern, line.strip())
                if match:
                    ref = match.group(1)
                    href = match.group(2)
                    title = match.group(3) if match.group(3) else None
                    reference_defs[ref.lower()] = {"href": href, "title": title}
            
            # Находим все ссылки по ссылкам
            for match in re.finditer(reference_link_pattern, content):
                text = match.group(1)
                ref = match.group(2) or text  # Если ссылка пустая, используем текст
                
                # Проверяем, есть ли определение ссылки
                if ref.lower() in reference_defs:
                    href = reference_defs[ref.lower()]["href"]
                    title = reference_defs[ref.lower()]["title"]
                    
                    # Пропускаем пустые ссылки и якоря
                    if not href or href.startswith('#'):
                        continue
                    
                    # Преобразуем относительные ссылки в абсолютные
                    if base_url and not (href.startswith('http://') or href.startswith('https://')):
                        href = urljoin(base_url, href)
                    
                    # Создаем словарь с информацией о ссылке
                    link_info = {
                        "href": href,
                        "text": text
                    }
                    
                    # Добавляем заголовок, если есть
                    if title:
                        link_info["title"] = title
                    
                    # Определяем, является ли ссылка внешней
                    if base_url and href.startswith(('http://', 'https://')):
                        link_info["is_external"] = not href.startswith(base_url)
                    
                    links.append(link_info)
            
            return links
        except Exception as e:
            logger.warning(f"Error extracting Markdown links: {e}")
            return links
    
    def _extract_markdown_images(self, content: str, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Извлекает изображения из Markdown-документа.
        
        Args:
            content: Содержимое Markdown-документа
            base_url: Базовый URL для относительных ссылок
            
        Returns:
            Список словарей с информацией об изображениях
        """
        images = []
        
        try:
            # Ищем изображения в формате ![alt](url "title")
            inline_image_pattern = r'!\[([^\]]*)\]\(([^)]+?)(?:\s+"([^"]+)")?\)'
            
            # Находим все совпадения
            for match in re.finditer(inline_image_pattern, content):
                alt = match.group(1)
                src = match.group(2)
                title = match.group(3) if match.group(3) else None
                
                # Пропускаем пустые источники
                if not src:
                    continue
                
                # Преобразуем относительные ссылки в абсолютные
                if base_url and not (src.startswith('http://') or src.startswith('https://')):
                    src = urljoin(base_url, src)
                
                # Создаем словарь с информацией об изображении
                image_info = {
                    "src": src,
                    "alt": alt
                }
                
                # Добавляем заголовок, если есть
                if title:
                    image_info["title"] = title
                
                images.append(image_info)
            
            # Ищем изображения в формате ![alt][ref] и [ref]: url "title"
            reference_image_pattern = r'!\[([^\]]*)\]\[([^\]]*)\]'
            reference_def_pattern = r'^\[([^\]]+)\]:\s*([^ ]+)(?:\s+"([^"]+)")?$'
            
            # Находим все определения ссылок
            reference_defs = {}
            for line in content.split('\n'):
                match = re.match(reference_def_pattern, line.strip())
                if match:
                    ref = match.group(1)
                    href = match.group(2)
                    title = match.group(3) if match.group(3) else None
                    reference_defs[ref.lower()] = {"href": href, "title": title}
            
            # Находим все изображения по ссылкам
            for match in re.finditer(reference_image_pattern, content):
                alt = match.group(1)
                ref = match.group(2) or alt  # Если ссылка пустая, используем текст
                
                # Проверяем, есть ли определение ссылки
                if ref.lower() in reference_defs:
                    src = reference_defs[ref.lower()]["href"]
                    title = reference_defs[ref.lower()]["title"]
                    
                    # Пропускаем пустые источники
                    if not src:
                        continue
                    
                    # Преобразуем относительные ссылки в абсолютные
                    if base_url and not (src.startswith('http://') or src.startswith('https://')):
                        src = urljoin(base_url, src)
                    
                    # Создаем словарь с информацией об изображении
                    image_info = {
                        "src": src,
                        "alt": alt
                    }
                    
                    # Добавляем заголовок, если есть
                    if title:
                        image_info["title"] = title
                    
                    images.append(image_info)
            
            return images
        except Exception as e:
            logger.warning(f"Error extracting Markdown images: {e}")
            return images
    
    def _create_document_structure(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Создает структуру документа на основе заголовков.
        
        Args:
            headings: Список словарей с информацией о заголовках
            
        Returns:
            Словарь со структурой документа
        """
        structure = {
            "title": None,
            "sections": []
        }
        
        try:
            if not headings:
                return structure
            
            # Находим заголовок документа (H1)
            h1_headings = [h for h in headings if h["level"] == 1]
            if h1_headings:
                structure["title"] = h1_headings[0]["text"]
            
            # Создаем иерархию разделов
            current_sections = structure["sections"]
            current_level = 0
            section_stack = []
            
            for heading in headings:
                level = heading["level"]
                
                # Переходим на уровень выше
                while current_level >= level and section_stack:
                    current_sections = section_stack.pop()
                    current_level -= 1
                
                # Создаем новый раздел
                section = {
                    "title": heading["text"],
                    "level": level,
                    "index": heading.get("index"),
                    "sections": []
                }
                
                # Добавляем раздел
                current_sections.append(section)
                
                # Переходим на уровень ниже
                section_stack.append(current_sections)
                current_sections = section["sections"]
                current_level = level
            
            return structure
        except Exception as e:
            logger.warning(f"Error creating document structure: {e}")
            return structure
    
    def _chunk_structured_text(
        self,
        content: str,
        headings: List[Dict[str, Any]],
        structure: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        Разбивает текст на чанки с учетом структуры документа.
        
        Args:
            content: Текст документа
            headings: Список словарей с информацией о заголовках
            structure: Словарь со структурой документа
            chunk_size: Максимальный размер чанка
            chunk_overlap: Размер перекрытия между чанками
            
        Returns:
            Список чанков
        """
        try:
            # Если нет заголовков, используем обычное разбиение
            if not headings:
                return self.chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Разбиваем текст на разделы по заголовкам
            lines = content.split('\n')
            sections = []
            
            # Сортируем заголовки по индексу
            sorted_headings = sorted(headings, key=lambda h: h.get("index", 0))
            
            for i, heading in enumerate(sorted_headings):
                # Определяем начало раздела
                start_index = heading.get("index", 0)
                
                # Определяем конец раздела
                if i < len(sorted_headings) - 1:
                    end_index = sorted_headings[i + 1].get("index", len(lines)) - 1
                else:
                    end_index = len(lines) - 1
                
                # Создаем раздел
                section_lines = lines[start_index:end_index + 1]
                section_text = '\n'.join(section_lines)
                
                # Добавляем информацию о разделе
                section = {
                    "title": heading["text"],
                    "level": heading["level"],
                    "text": section_text
                }
                
                sections.append(section)
            
            # Разбиваем разделы на чанки
            chunks = []
            
            # Если есть только один раздел
            if len(sections) == 1:
                return self.chunk_text(sections[0]["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            current_chunk = ""
            current_section_titles = []
            
            for section in sections:
                # Добавляем заголовок раздела
                section_header = f"{'#' * section['level']} {section['title']}\n\n"
                section_text = section["text"]
                
                # Если текущий чанк + новый раздел меньше максимального размера,
                # добавляем весь раздел
                if len(current_chunk) + len(section_text) <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n"
                    
                    current_chunk += section_text
                    current_section_titles.append(section["title"])
                else:
                    # Если раздел слишком большой, разбиваем его на части
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    # Разбиваем большой раздел
                    section_chunks = self.chunk_text(
                        section_text,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Добавляем заголовок к каждому чанку
                    for i, section_chunk in enumerate(section_chunks):
                        if i == 0:
                            chunks.append(section_chunk)
                        else:
                            chunks.append(f"{section_header}{section_chunk}")
                    
                    current_section_titles = []
                
                # Если текущий чанк достаточно большой, добавляем его в список
                if len(current_chunk) >= chunk_size - chunk_overlap:
                    chunks.append(current_chunk)
                    
                    # Начинаем новый чанк с перекрытием
                    words = current_chunk.split()
                    overlap_words = words[-min(len(words), chunk_overlap // 5):]
                    current_chunk = ' '.join(overlap_words)
                    
                    current_section_titles = []
            
            # Добавляем последний чанк
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
        except Exception as e:
            logger.warning(f"Error chunking structured text: {e}")
            return self.chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
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

ParserFactory.register_parser(WebParser)