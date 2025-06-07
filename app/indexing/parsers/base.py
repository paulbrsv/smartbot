import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from app.core.config import config
from app.core.models import Document, DocumentChunk, ChunkMetadata

logger = logging.getLogger("smartbot.parsers")


class BaseParser(ABC):
    """
    Абстрактный базовый класс для всех парсеров документов.
    Определяет общий интерфейс для парсеров различных типов файлов.
    """

    def __init__(self):
        self.processing_config = config.settings.processing
        self.chunk_size = self.processing_config.chunk_size
        self.chunk_overlap = self.processing_config.chunk_overlap

    def can_parse(self, file_path: str) -> bool:
        """
        Проверяет, может ли парсер обработать данный файл.
        По умолчанию проверяет, входит ли расширение файла в список поддерживаемых расширений.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            True, если парсер может обработать файл, иначе False.
        """
        if hasattr(self, 'supported_extensions'):
            _, ext = os.path.splitext(file_path)
            return ext.lower() in self.supported_extensions
        return False

    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> 'ParsedDocument':
        """
        Парсит файл и извлекает его содержимое.
        
        Args:
            file_path: Путь к файлу.
            **kwargs: Дополнительные параметры для парсинга.
            
        Returns:
            Объект ParsedDocument с содержимым и метаданными.
        """
        pass

    def create_document(self, file_path: str) -> Document:
        """
        Создает объект Document на основе файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Объект Document.
        """
        path = Path(file_path)
        filename = path.name
        file_size = path.stat().st_size
        file_type = path.suffix.lower().lstrip('.')
        
        return Document(
            filename=filename,
            filepath=str(path),
            filetype=file_type,
            size_bytes=file_size,
            metadata={
                "created_at": path.stat().st_ctime,
                "modified_at": path.stat().st_mtime
            }
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Разбивает текст на чанки заданного размера с перекрытием.
        
        Args:
            text: Текст для разбивки.
            
        Returns:
            Список чанков текста.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Если это не последний чанк, то ищем конец предложения или абзаца
            if end < text_length:
                # Поиск конца предложения или абзаца
                for delimiter in ['\n\n', '\n', '. ', '! ', '? ']:
                    last_delimiter = text.rfind(delimiter, start, end)
                    if last_delimiter > start:
                        end = last_delimiter + len(delimiter)
                        break
            
            # Добавляем чанк
            chunks.append(text[start:end].strip())
            
            # Вычисляем следующую стартовую позицию с учетом перекрытия
            start = end - self.chunk_overlap
            
            # Корректируем стартовую позицию, если она слишком близка к концу
            if start + self.chunk_size >= text_length and start < text_length:
                start = max(text_length - self.chunk_size, 0)
            
        return chunks

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Извлекает базовые метаданные из файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Словарь с метаданными.
        """
        path = Path(file_path)
        
        metadata = {
            "filename": path.name,
            "filepath": str(path),
            "filetype": path.suffix.lower().lstrip('.'),
            "size_bytes": path.stat().st_size,
            "created_at": path.stat().st_ctime,
            "modified_at": path.stat().st_mtime
        }
        
        return metadata

    def create_chunks(self, document: Document, texts: List[str], 
                      metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Создает объекты DocumentChunk на основе текстовых фрагментов.
        
        Args:
            document: Объект Document.
            texts: Список текстовых фрагментов.
            metadata: Дополнительные метаданные для чанков.
            
        Returns:
            Список объектов DocumentChunk.
        """
        chunks = []
        
        for i, text in enumerate(texts):
            # Создаем метаданные чанка
            chunk_metadata = ChunkMetadata(
                document_id=str(document.id),
                document_name=document.filename,
                document_type=document.filetype,
                chunk_order=i
            )
            
            # Добавляем дополнительные метаданные, если они есть
            if metadata:
                for key, value in metadata.items():
                    if hasattr(chunk_metadata, key):
                        setattr(chunk_metadata, key, value)
            
            # Создаем чанк
            chunk = DocumentChunk(
                document_id=str(document.id),
                chunk_text=text,
                chunk_order=i,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
        
        return chunks


class ParserFactory:
    """
    Фабрика для создания и получения подходящего парсера для файла.
    """
    
    def __init__(self):
        self.parsers = []
    
    @staticmethod
    def register_parser(parser_class):
        """
        Регистрирует парсер в фабрике.
        
        Args:
            parser_class: Класс парсера.
        """
        try:
            # Создаем экземпляр парсера
            parser = parser_class()
            # Добавляем в глобальный экземпляр фабрики
            parser_factory.parsers.append(parser)
            logger.info(f"Parser {parser_class.__name__} successfully registered")
        except Exception as e:
            logger.error(f"Error registering parser {parser_class.__name__}: {e}")
    
    def get_parser(self, file_path: str) -> Optional[BaseParser]:
        """
        Получает подходящий парсер для файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Подходящий парсер или None, если такого нет.
        """
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        
        return None
    
    @staticmethod
    def get_all_parsers() -> List[BaseParser]:
        """
        Получает список всех доступных парсеров.
        
        Returns:
            Список экземпляров всех парсеров.
        """
        import importlib
        import os
        import inspect
        import sys
        from pathlib import Path
        
        parsers = []
        parsers_dir = Path(__file__).parent
        
        # Получаем список всех Python-файлов в директории парсеров
        parser_files = [f for f in os.listdir(parsers_dir)
                       if f.endswith('.py') and f != 'base.py' and f != '__init__.py']
        
        # Импортируем каждый модуль и ищем в нем классы, наследующиеся от BaseParser
        for file in parser_files:
            module_name = file[:-3]  # Удаляем расширение .py
            try:
                # Импортируем модуль
                module_path = f"app.indexing.parsers.{module_name}"
                module = importlib.import_module(module_path)
                
                # Ищем классы парсеров в модуле
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, BaseParser) and
                        obj != BaseParser):
                        try:
                            # Создаем экземпляр парсера и добавляем его в список
                            parser = obj()
                            parsers.append(parser)
                            logger.info(f"Loaded parser: {name} from {module_path}")
                        except Exception as e:
                            logger.error(f"Error instantiating parser {name}: {e}")
            except Exception as e:
                logger.error(f"Error importing parser module {module_name}: {e}")
        
        logger.info(f"Total parsers loaded: {len(parsers)}")
        return parsers


class ParsedDocument:
    """
    Класс, представляющий результат парсинга документа.
    Содержит извлеченное содержимое, чанки и метаданные.
    """
    
    def __init__(
        self,
        content: str = "",
        chunks: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Инициализирует результат парсинга документа.
        
        Args:
            content: Полное содержимое документа в текстовом виде.
            chunks: Список текстовых фрагментов документа.
            metadata: Метаданные документа.
        """
        self.content = content
        self.chunks = chunks or []
        self.metadata = metadata or {}


# Создаем экземпляр фабрики парсеров
parser_factory = ParserFactory()