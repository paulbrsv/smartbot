import os
import logging
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import zipfile
import rarfile
import pathlib

from app.indexing.parsers.base import BaseParser, ParsedDocument, ParserFactory
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ArchiveParser(BaseParser):
    """Парсер для архивных файлов (ZIP, RAR)."""
    
    def __init__(self):
        """Инициализация парсера архивных файлов."""
        super().__init__()
        self.supported_extensions = [".zip", ".rar"]
        
        # Проверяем наличие UnRAR
        try:
            rarfile.UNRAR_TOOL  # Проверяем, задана ли переменная
            self._rar_supported = True
        except (AttributeError, ImportError):
            # Пытаемся установить путь к unrar для разных ОС
            if os.name == 'nt':  # Windows
                rarfile.UNRAR_TOOL = "unrar.exe"
            else:  # Linux/Mac
                rarfile.UNRAR_TOOL = "unrar"
            
            # Проверяем, работает ли теперь
            try:
                rarfile.check_rarfile_program()
                self._rar_supported = True
            except rarfile.RarCannotExec:
                logger.warning("UnRAR executable not found. RAR archives will not be supported.")
                self._rar_supported = False
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает архивный файл, извлекает его содержимое и обрабатывает файлы внутри архива.
        
        Args:
            file_path: Путь к файлу архива
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - max_files: Максимальное количество файлов для обработки (по умолчанию 100)
                - max_size_mb: Максимальный размер распакованных данных в МБ (по умолчанию 500)
                - nested_archives: Обрабатывать ли вложенные архивы (по умолчанию True)
                - max_depth: Максимальная глубина вложенных архивов (по умолчанию 3)
                - include_extensions: Список расширений файлов для обработки (по умолчанию None - все поддерживаемые)
                - exclude_extensions: Список расширений файлов для исключения (по умолчанию [".exe", ".dll", ".bin"])
                - password: Пароль для защищенных архивов (по умолчанию None)
                - extract_only: Только извлечь список файлов без обработки (по умолчанию False)
            
        Returns:
            Объект ParsedDocument с содержимым и метаданными
        """
        try:
            # Получаем метаданные файла
            metadata = self.get_file_metadata(file_path)
            
            # Определяем параметры
            chunk_size = kwargs.get("chunk_size", settings.processing.chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.processing.chunk_overlap)
            max_files = kwargs.get("max_files", 100)
            max_size_mb = kwargs.get("max_size_mb", 500)
            nested_archives = kwargs.get("nested_archives", True)
            max_depth = kwargs.get("max_depth", 3)
            include_extensions = kwargs.get("include_extensions", None)
            exclude_extensions = kwargs.get("exclude_extensions", [".exe", ".dll", ".bin"])
            password = kwargs.get("password", None)
            extract_only = kwargs.get("extract_only", False)
            
            # Проверяем, поддерживается ли формат архива
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == ".rar" and not self._rar_supported:
                raise ValueError("RAR archives are not supported. UnRAR executable not found.")
            
            # Создаем временную директорию для распаковки архива
            temp_dir = tempfile.mkdtemp(prefix="smartbot_archive_")
            
            try:
                # Определяем тип архива и извлекаем файлы
                if ext == ".zip":
                    archive_info = self._process_zip_archive(
                        file_path, 
                        temp_dir,
                        max_files=max_files,
                        max_size_mb=max_size_mb,
                        password=password
                    )
                elif ext == ".rar":
                    archive_info = self._process_rar_archive(
                        file_path, 
                        temp_dir,
                        max_files=max_files,
                        max_size_mb=max_size_mb,
                        password=password
                    )
                else:
                    raise ValueError(f"Unsupported archive format: {ext}")
                
                # Добавляем информацию об архиве в метаданные
                metadata.update(archive_info)
                
                if extract_only:
                    # Если нужно только извлечь список файлов, возвращаем результаты
                    return ParsedDocument(
                        content=f"Archive: {os.path.basename(file_path)}\nFiles: {len(archive_info.get('files', []))}\n",
                        metadata=metadata,
                        chunks=[]
                    )
                
                # Обрабатываем извлеченные файлы
                content, files_metadata = self._process_extracted_files(
                    temp_dir,
                    depth=0,
                    max_depth=max_depth,
                    nested_archives=nested_archives,
                    include_extensions=include_extensions,
                    exclude_extensions=exclude_extensions,
                    processed_archives=set(),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Добавляем информацию об обработанных файлах в метаданные
                metadata["processed_files"] = files_metadata
                metadata["processed_file_count"] = len(files_metadata)
                
                # Разбиваем содержимое на чанки
                chunks = self.chunk_text(
                    content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Добавляем информацию о размере содержимого
                metadata["content_length"] = len(content)
                metadata["content_type"] = "application/x-archive-content"
                
                return ParsedDocument(
                    content=content,
                    metadata=metadata,
                    chunks=chunks
                )
            finally:
                # Удаляем временную директорию
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Error removing temporary directory {temp_dir}: {e}")
        except Exception as e:
            logger.error(f"Error parsing archive file {file_path}: {e}")
            raise
    
    def _process_zip_archive(
        self,
        file_path: str,
        extract_dir: str,
        max_files: int = 100,
        max_size_mb: int = 500,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Обрабатывает ZIP-архив.
        
        Args:
            file_path: Путь к файлу архива
            extract_dir: Директория для распаковки
            max_files: Максимальное количество файлов для обработки
            max_size_mb: Максимальный размер распакованных данных в МБ
            password: Пароль для защищенных архивов
            
        Returns:
            Словарь с информацией об архиве
        """
        try:
            # Открываем архив
            pwd = password.encode('utf-8') if password else None
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Получаем информацию о файлах в архиве
                file_info_list = zip_ref.infolist()
                
                # Сортируем файлы по размеру (от меньших к большим)
                file_info_list.sort(key=lambda x: x.file_size)
                
                # Ограничиваем количество файлов
                if max_files > 0 and len(file_info_list) > max_files:
                    file_info_list = file_info_list[:max_files]
                
                # Подсчитываем общий размер файлов
                total_size = sum(info.file_size for info in file_info_list)
                max_size_bytes = max_size_mb * 1024 * 1024
                
                if total_size > max_size_bytes:
                    # Если общий размер превышает максимальный, ограничиваем список файлов
                    current_size = 0
                    limited_file_list = []
                    
                    for info in file_info_list:
                        if current_size + info.file_size <= max_size_bytes:
                            limited_file_list.append(info)
                            current_size += info.file_size
                        else:
                            break
                    
                    file_info_list = limited_file_list
                
                # Собираем информацию о файлах
                files = []
                directories = []
                
                for info in file_info_list:
                    # Формируем путь для извлечения
                    # В Windows имена файлов в архивах могут использовать разные разделители
                    filename = info.filename.replace('\\', '/')
                    
                    # Проверяем, является ли это директорией
                    if filename.endswith('/'):
                        directories.append({
                            "name": filename,
                            "size": 0,
                            "date_time": datetime(*info.date_time).isoformat(),
                            "crc": info.CRC
                        })
                        continue
                    
                    # Получаем информацию о файле
                    _, ext = os.path.splitext(filename)
                    
                    file_metadata = {
                        "name": filename,
                        "size": info.file_size,
                        "compressed_size": info.compress_size,
                        "date_time": datetime(*info.date_time).isoformat(),
                        "crc": info.CRC,
                        "extension": ext.lower(),
                        "compression_ratio": info.compress_size / info.file_size if info.file_size > 0 else 1.0
                    }
                    
                    files.append(file_metadata)
                    
                    # Извлекаем файл
                    try:
                        zip_ref.extract(info, path=extract_dir, pwd=pwd)
                    except zipfile.BadZipFile as e:
                        logger.warning(f"Error extracting {info.filename}: {e}")
                        file_metadata["extraction_error"] = str(e)
                    except Exception as e:
                        logger.warning(f"Unexpected error extracting {info.filename}: {e}")
                        file_metadata["extraction_error"] = str(e)
                
                # Формируем метаданные архива
                archive_info = {
                    "archive_type": "zip",
                    "total_files": len(zip_ref.namelist()),
                    "extracted_files": len(files),
                    "directories": len(directories),
                    "total_size_bytes": total_size,
                    "total_size_mb": total_size / (1024 * 1024),
                    "files": files,
                    "password_protected": zip_ref.getinfo(zip_ref.namelist()[0]).flag_bits & 0x1,
                    "comment": zip_ref.comment.decode('utf-8', errors='replace') if zip_ref.comment else None
                }
                
                return archive_info
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file {file_path}: {e}")
            raise ValueError(f"Invalid ZIP file: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing ZIP file {file_path}: {e}")
            raise
    
    def _process_rar_archive(
        self,
        file_path: str,
        extract_dir: str,
        max_files: int = 100,
        max_size_mb: int = 500,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Обрабатывает RAR-архив.
        
        Args:
            file_path: Путь к файлу архива
            extract_dir: Директория для распаковки
            max_files: Максимальное количество файлов для обработки
            max_size_mb: Максимальный размер распакованных данных в МБ
            password: Пароль для защищенных архивов
            
        Returns:
            Словарь с информацией об архиве
        """
        try:
            # Открываем архив
            with rarfile.RarFile(file_path, 'r', pwd=password) as rar_ref:
                # Получаем информацию о файлах в архиве
                file_info_list = rar_ref.infolist()
                
                # Сортируем файлы по размеру (от меньших к большим)
                file_info_list.sort(key=lambda x: x.file_size)
                
                # Ограничиваем количество файлов
                if max_files > 0 and len(file_info_list) > max_files:
                    file_info_list = file_info_list[:max_files]
                
                # Подсчитываем общий размер файлов
                total_size = sum(info.file_size for info in file_info_list)
                max_size_bytes = max_size_mb * 1024 * 1024
                
                if total_size > max_size_bytes:
                    # Если общий размер превышает максимальный, ограничиваем список файлов
                    current_size = 0
                    limited_file_list = []
                    
                    for info in file_info_list:
                        if current_size + info.file_size <= max_size_bytes:
                            limited_file_list.append(info)
                            current_size += info.file_size
                        else:
                            break
                    
                    file_info_list = limited_file_list
                
                # Собираем информацию о файлах
                files = []
                directories = []
                
                for info in file_info_list:
                    # Формируем путь для извлечения
                    # В Windows имена файлов в архивах могут использовать разные разделители
                    filename = info.filename.replace('\\', '/')
                    
                    # Проверяем, является ли это директорией
                    if info.isdir():
                        directories.append({
                            "name": filename,
                            "size": 0,
                            "date_time": datetime.fromtimestamp(info.date_time).isoformat(),
                            "crc": info.CRC
                        })
                        continue
                    
                    # Получаем информацию о файле
                    _, ext = os.path.splitext(filename)
                    
                    file_metadata = {
                        "name": filename,
                        "size": info.file_size,
                        "compressed_size": info.compress_size,
                        "date_time": datetime.fromtimestamp(info.date_time).isoformat(),
                        "crc": info.CRC,
                        "extension": ext.lower(),
                        "compression_ratio": info.compress_size / info.file_size if info.file_size > 0 else 1.0
                    }
                    
                    files.append(file_metadata)
                    
                    # Извлекаем файл
                    try:
                        rar_ref.extract(info, path=extract_dir)
                    except rarfile.BadRarFile as e:
                        logger.warning(f"Error extracting {info.filename}: {e}")
                        file_metadata["extraction_error"] = str(e)
                    except Exception as e:
                        logger.warning(f"Unexpected error extracting {info.filename}: {e}")
                        file_metadata["extraction_error"] = str(e)
                
                # Формируем метаданные архива
                archive_info = {
                    "archive_type": "rar",
                    "total_files": len(rar_ref.namelist()),
                    "extracted_files": len(files),
                    "directories": len(directories),
                    "total_size_bytes": total_size,
                    "total_size_mb": total_size / (1024 * 1024),
                    "files": files,
                    "password_protected": rar_ref.needs_password(),
                    "rar_version": rar_ref.rarversion,
                    "volume": rar_ref.is_multivolume()
                }
                
                # Добавляем информацию о комментарии, если он есть
                if rar_ref.comment:
                    archive_info["comment"] = rar_ref.comment.decode('utf-8', errors='replace')
                
                return archive_info
        except rarfile.BadRarFile as e:
            logger.error(f"Invalid RAR file {file_path}: {e}")
            raise ValueError(f"Invalid RAR file: {str(e)}")
        except rarfile.RarCannotExec:
            logger.error("UnRAR executable not found or not working properly")
            raise ValueError("UnRAR executable not found or not working properly")
        except Exception as e:
            logger.error(f"Error processing RAR file {file_path}: {e}")
            raise
    
    def _process_extracted_files(
        self,
        extract_dir: str,
        depth: int = 0,
        max_depth: int = 3,
        nested_archives: bool = True,
        include_extensions: Optional[List[str]] = None,
        exclude_extensions: Optional[List[str]] = None,
        processed_archives: Optional[Set[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Обрабатывает извлеченные из архива файлы.
        
        Args:
            extract_dir: Директория с извлеченными файлами
            depth: Текущая глубина вложенности
            max_depth: Максимальная глубина вложенности
            nested_archives: Обрабатывать ли вложенные архивы
            include_extensions: Список расширений файлов для обработки
            exclude_extensions: Список расширений файлов для исключения
            processed_archives: Множество уже обработанных архивов (для избежания циклических ссылок)
            chunk_size: Размер чанка
            chunk_overlap: Размер перекрытия
            
        Returns:
            Кортеж (объединенное содержимое, метаданные обработанных файлов)
        """
        if processed_archives is None:
            processed_archives = set()
        
        content = []
        files_metadata = []
        
        # Получаем все файлы в директории
        file_paths = []
        
        for root, _, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        
        # Проверяем расширения файлов
        for file_path in file_paths:
            rel_path = os.path.relpath(file_path, extract_dir)
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # Проверяем, нужно ли обрабатывать этот файл
            if include_extensions and ext not in include_extensions:
                continue
            
            if exclude_extensions and ext in exclude_extensions:
                continue
            
            # Проверяем, является ли файл архивом
            is_archive = ext in [".zip", ".rar"]
            
            # Обрабатываем архивы, если это разрешено и не превышена максимальная глубина
            if is_archive and nested_archives and depth < max_depth:
                # Проверяем, не обрабатывали ли мы уже этот архив
                file_hash = self._get_file_hash(file_path)
                
                if file_hash in processed_archives:
                    logger.warning(f"Skipping already processed archive: {rel_path}")
                    continue
                
                # Добавляем архив в множество обработанных
                processed_archives.add(file_hash)
                
                # Создаем временную директорию для распаковки вложенного архива
                nested_temp_dir = tempfile.mkdtemp(prefix="smartbot_nested_archive_")
                
                try:
                    # Обрабатываем вложенный архив
                    if ext == ".zip":
                        archive_info = self._process_zip_archive(
                            file_path, 
                            nested_temp_dir,
                            max_files=100,
                            max_size_mb=100  # Ограничиваем размер вложенных архивов
                        )
                    elif ext == ".rar":
                        archive_info = self._process_rar_archive(
                            file_path, 
                            nested_temp_dir,
                            max_files=100,
                            max_size_mb=100  # Ограничиваем размер вложенных архивов
                        )
                    else:
                        continue
                    
                    # Обрабатываем извлеченные файлы из вложенного архива
                    nested_content, nested_files_metadata = self._process_extracted_files(
                        nested_temp_dir,
                        depth=depth + 1,
                        max_depth=max_depth,
                        nested_archives=nested_archives,
                        include_extensions=include_extensions,
                        exclude_extensions=exclude_extensions,
                        processed_archives=processed_archives,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Добавляем информацию о вложенном архиве
                    file_content = f"NESTED ARCHIVE: {rel_path}\n"
                    file_content += f"Type: {archive_info.get('archive_type', 'unknown')}\n"
                    file_content += f"Total files: {archive_info.get('total_files', 0)}\n"
                    file_content += f"Extracted files: {archive_info.get('extracted_files', 0)}\n"
                    file_content += f"Total size: {archive_info.get('total_size_mb', 0):.2f} MB\n"
                    
                    if archive_info.get('comment'):
                        file_content += f"Comment: {archive_info.get('comment')}\n"
                    
                    file_content += "\n"
                    file_content += nested_content
                    
                    content.append(file_content)
                    
                    # Добавляем метаданные о вложенном архиве
                    archive_metadata = {
                        "path": rel_path,
                        "type": "nested_archive",
                        "extension": ext,
                        "archive_info": archive_info,
                        "processed_files": nested_files_metadata,
                        "nested_depth": depth + 1
                    }
                    
                    files_metadata.append(archive_metadata)
                finally:
                    # Удаляем временную директорию
                    try:
                        shutil.rmtree(nested_temp_dir)
                    except Exception as e:
                        logger.warning(f"Error removing temporary directory {nested_temp_dir}: {e}")
            else:
                # Обрабатываем обычный файл
                # Проверяем, поддерживается ли формат файла
                # Для этого используем фабрику парсеров
                try:
                    # Пытаемся найти подходящий парсер
                    parser = ParserFactory.get_parser_for_file(file_path)
                    
                    if parser:
                        # Парсим файл
                        parsed_doc = parser.parse(
                            file_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        # Добавляем содержимое
                        file_content = f"FILE: {rel_path}\n"
                        
                        # Добавляем метаданные
                        if "title" in parsed_doc.metadata:
                            file_content += f"Title: {parsed_doc.metadata.get('title')}\n"
                        
                        if "author" in parsed_doc.metadata:
                            file_content += f"Author: {parsed_doc.metadata.get('author')}\n"
                        
                        if "content_type" in parsed_doc.metadata:
                            file_content += f"Content type: {parsed_doc.metadata.get('content_type')}\n"
                        
                        file_content += "\n"
                        file_content += parsed_doc.content
                        file_content += "\n\n" + "=" * 80 + "\n\n"
                        
                        content.append(file_content)
                        
                        # Добавляем метаданные о файле
                        file_metadata = {
                            "path": rel_path,
                            "type": "file",
                            "extension": ext,
                            "parser": parser.__class__.__name__,
                            "metadata": parsed_doc.metadata,
                            "chunk_count": len(parsed_doc.chunks) if parsed_doc.chunks else 0
                        }
                        
                        files_metadata.append(file_metadata)
                    else:
                        # Если парсер не найден, добавляем информацию о файле
                        logger.warning(f"No parser found for file: {rel_path}")
                        
                        # Пытаемся определить, является ли файл текстовым
                        try:
                            # Определяем тип файла
                            import magic
                            mime_type = magic.from_file(file_path, mime=True)
                            
                            # Если файл текстовый, читаем его содержимое
                            if mime_type and mime_type.startswith('text/'):
                                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                    file_text = f.read(1024 * 100)  # Ограничиваем размер
                                
                                file_content = f"FILE: {rel_path} (text)\n"
                                file_content += f"MIME type: {mime_type}\n\n"
                                file_content += file_text
                                file_content += "\n\n" + "=" * 80 + "\n\n"
                                
                                content.append(file_content)
                            else:
                                # Для бинарных файлов просто добавляем информацию
                                file_content = f"FILE: {rel_path} (binary)\n"
                                file_content += f"MIME type: {mime_type}\n"
                                file_content += f"Size: {os.path.getsize(file_path)} bytes\n\n"
                                
                                content.append(file_content)
                            
                            # Добавляем метаданные о файле
                            file_metadata = {
                                "path": rel_path,
                                "type": "unsupported_file",
                                "extension": ext,
                                "mime_type": mime_type,
                                "size": os.path.getsize(file_path)
                            }
                            
                            files_metadata.append(file_metadata)
                        except ImportError:
                            # Если нет библиотеки magic, просто добавляем информацию о файле
                            file_content = f"FILE: {rel_path} (unsupported)\n"
                            file_content += f"Size: {os.path.getsize(file_path)} bytes\n\n"
                            
                            content.append(file_content)
                            
                            # Добавляем метаданные о файле
                            file_metadata = {
                                "path": rel_path,
                                "type": "unsupported_file",
                                "extension": ext,
                                "size": os.path.getsize(file_path)
                            }
                            
                            files_metadata.append(file_metadata)
                except Exception as e:
                    logger.warning(f"Error processing file {rel_path}: {e}")
                    
                    # Добавляем информацию об ошибке
                    file_content = f"FILE: {rel_path} (error)\n"
                    file_content += f"Error: {str(e)}\n\n"
                    
                    content.append(file_content)
                    
                    # Добавляем метаданные о файле
                    file_metadata = {
                        "path": rel_path,
                        "type": "error",
                        "extension": ext,
                        "error": str(e)
                    }
                    
                    files_metadata.append(file_metadata)
        
        # Объединяем содержимое
        combined_content = "\n".join(content)
        
        return combined_content, files_metadata
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Вычисляет хеш файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Хеш файла
        """
        import hashlib
        
        # Получаем размер файла
        file_size = os.path.getsize(file_path)
        
        # Если файл большой, считаем хеш только части
        if file_size > 1024 * 1024:  # 1MB
            with open(file_path, 'rb') as f:
                # Считываем начало и конец файла
                head = f.read(1024 * 512)  # 512KB
                f.seek(max(0, file_size - 1024 * 512))
                tail = f.read(1024 * 512)  # 512KB
                
                # Вычисляем хеш
                sha256 = hashlib.sha256()
                sha256.update(head)
                sha256.update(tail)
                
                return sha256.hexdigest()
        else:
            # Для небольших файлов считаем хеш всего содержимого
            with open(file_path, 'rb') as f:
                data = f.read()
                
                # Вычисляем хеш
                sha256 = hashlib.sha256()
                sha256.update(data)
                
                return sha256.hexdigest()


# Регистрируем парсер в фабрике
from app.indexing.parsers.base import ParserFactory

ParserFactory.register_parser(ArchiveParser)