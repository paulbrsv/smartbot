import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime
import json
import base64
from io import BytesIO

from app.indexing.parsers.base import BaseParser, ParsedDocument
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ImageParser(BaseParser):
    """Парсер для файлов изображений с OCR."""
    
    def __init__(self):
        """Инициализация парсера изображений."""
        super().__init__()
        self.supported_extensions = [
            ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"
        ]
    
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Разбирает файл изображения и возвращает извлеченный текст и метаданные.
        
        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры
                - chunk_size: Размер чанка (по умолчанию из настроек)
                - chunk_overlap: Размер перекрытия (по умолчанию из настроек)
                - ocr_enabled: Использовать ли OCR (по умолчанию True)
                - ocr_language: Язык OCR (по умолчанию из настроек)
                - ocr_dpi: DPI для OCR (по умолчанию из настроек)
                - extract_metadata: Извлекать ли метаданные EXIF (по умолчанию True)
                - include_image_data: Включать ли данные изображения в base64 (по умолчанию False)
                - image_resize: Размер для изменения размера изображения перед OCR (по умолчанию None)
                - preprocess_image: Применять ли предобработку изображения для улучшения OCR (по умолчанию True)
            
        Returns:
            Объект ParsedDocument с извлеченным текстом и метаданными
        """
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import pytesseract
            
            # Получаем метаданные файла
            metadata = self.get_file_metadata(file_path)
            
            # Определяем параметры
            chunk_size = kwargs.get("chunk_size", settings.processing.chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.processing.chunk_overlap)
            ocr_enabled = kwargs.get("ocr_enabled", True)
            ocr_language = kwargs.get("ocr_language", settings.ocr.language)
            ocr_dpi = kwargs.get("ocr_dpi", settings.ocr.dpi)
            extract_metadata = kwargs.get("extract_metadata", True)
            include_image_data = kwargs.get("include_image_data", False)
            image_resize = kwargs.get("image_resize", None)
            preprocess_image = kwargs.get("preprocess_image", True)
            
            # Открываем изображение
            image = Image.open(file_path)
            
            # Добавляем базовые метаданные изображения
            image_metadata = self._extract_image_metadata(image, file_path, extract_metadata)
            metadata.update(image_metadata)
            
            # Извлекаем текст с помощью OCR, если включено
            content = ""
            if ocr_enabled:
                # Предобработка изображения для улучшения OCR
                if preprocess_image:
                    image = self._preprocess_image(image)
                
                # Изменяем размер изображения, если указано
                if image_resize:
                    width, height = self._parse_image_resize(image_resize, image.size)
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                
                # Выполняем OCR
                content = self._perform_ocr(image, ocr_language, ocr_dpi)
                
                # Добавляем метаданные OCR
                metadata["ocr_enabled"] = True
                metadata["ocr_language"] = ocr_language
                metadata["ocr_text_length"] = len(content)
                metadata["ocr_text_empty"] = len(content.strip()) == 0
            else:
                metadata["ocr_enabled"] = False
            
            # Включаем данные изображения в base64, если требуется
            if include_image_data:
                buffered = BytesIO()
                image.save(buffered, format=image.format or "JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                metadata["image_data_base64"] = img_str
            
            # Добавляем информацию о размере содержимого и типе
            metadata["content_length"] = len(content)
            metadata["content_type"] = f"image/{metadata.get('format', 'jpeg').lower()}"
            
            # Разбиваем содержимое на чанки
            chunks = self.chunk_text(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ) if content else []
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                chunks=chunks
            )
        except ImportError:
            logger.error("Required libraries (PIL, pytesseract) not installed")
            raise ImportError("PIL and pytesseract are required for parsing images")
        except Exception as e:
            logger.error(f"Error parsing image file {file_path}: {e}")
            raise
    
    def _extract_image_metadata(
        self,
        image,
        file_path: str,
        extract_exif: bool = True
    ) -> Dict[str, Any]:
        """
        Извлекает метаданные из изображения.
        
        Args:
            image: Объект PIL Image
            file_path: Путь к файлу
            extract_exif: Извлекать ли метаданные EXIF
            
        Returns:
            Словарь с метаданными
        """
        metadata = {}
        
        try:
            # Базовые метаданные изображения
            metadata["width"] = image.width
            metadata["height"] = image.height
            metadata["format"] = image.format
            metadata["mode"] = image.mode
            metadata["is_animated"] = getattr(image, "is_animated", False)
            metadata["n_frames"] = getattr(image, "n_frames", 1)
            
            # Извлекаем EXIF метаданные, если включено
            if extract_exif:
                exif_metadata = self._extract_exif_metadata(image, file_path)
                if exif_metadata:
                    metadata["exif"] = exif_metadata
            
            # Анализируем изображение
            image_analysis = self._analyze_image(image)
            metadata.update(image_analysis)
            
            return metadata
        except Exception as e:
            logger.warning(f"Error extracting image metadata: {e}")
            return metadata
    
    def _extract_exif_metadata(self, image, file_path: str) -> Dict[str, Any]:
        """
        Извлекает метаданные EXIF из изображения.
        
        Args:
            image: Объект PIL Image
            file_path: Путь к файлу
            
        Returns:
            Словарь с метаданными EXIF
        """
        exif_metadata = {}
        
        try:
            # Пытаемся получить EXIF данные из изображения
            exif_data = image._getexif()
            
            # Если EXIF данные не найдены, пробуем использовать библиотеку exifread
            if not exif_data:
                try:
                    import exifread
                    with open(file_path, 'rb') as f:
                        tags = exifread.process_file(f, details=False)
                        if tags:
                            exif_data = {str(k): str(v) for k, v in tags.items()}
                except ImportError:
                    logger.debug("exifread library not installed")
                except Exception as e:
                    logger.debug(f"Error using exifread: {e}")
            
            # Обрабатываем EXIF данные, если они найдены
            if exif_data:
                # EXIF теги и их человекочитаемые названия
                exif_tags = {
                    271: "make",
                    272: "model",
                    274: "orientation",
                    306: "datetime",
                    36867: "datetime_original",
                    36868: "datetime_digitized",
                    37377: "shutter_speed",
                    37378: "aperture",
                    37379: "brightness",
                    37380: "exposure_compensation",
                    37381: "max_aperture",
                    37383: "metering_mode",
                    37384: "light_source",
                    37385: "flash",
                    37386: "focal_length",
                    37521: "rating",
                    41728: "file_source",
                    41729: "scene_type",
                    41986: "exposure_mode",
                    41987: "white_balance",
                    41988: "digital_zoom_ratio",
                    41989: "focal_length_35mm",
                    41990: "scene_capture_type",
                    41991: "gain_control",
                    41992: "contrast",
                    41993: "saturation",
                    41994: "sharpness",
                    42034: "lens_info",
                    42035: "lens_make",
                    42036: "lens_model",
                    0x8825: "gps_info",
                }
                
                # Преобразуем EXIF данные в словарь
                for tag, value in exif_data.items():
                    if isinstance(tag, int) and tag in exif_tags:
                        tag_name = exif_tags[tag]
                        exif_metadata[tag_name] = str(value)
                    elif isinstance(tag, str):
                        # Для данных из exifread
                        tag_name = tag.replace(" ", "_").lower()
                        exif_metadata[tag_name] = str(value)
                
                # Обработка GPS данных, если они есть
                if 'gps_info' in exif_metadata and isinstance(exif_data.get(0x8825), dict):
                    gps_data = exif_data[0x8825]
                    gps_metadata = {}
                    
                    # GPS теги
                    gps_tags = {
                        1: "gps_latitude_ref",
                        2: "gps_latitude",
                        3: "gps_longitude_ref",
                        4: "gps_longitude",
                        5: "gps_altitude_ref",
                        6: "gps_altitude",
                        7: "gps_timestamp",
                        29: "gps_date"
                    }
                    
                    # Преобразуем GPS данные
                    for tag, value in gps_data.items():
                        if tag in gps_tags:
                            tag_name = gps_tags[tag]
                            gps_metadata[tag_name] = str(value)
                    
                    # Добавляем GPS данные
                    if gps_metadata:
                        exif_metadata["gps"] = gps_metadata
            
            return exif_metadata
        except Exception as e:
            logger.warning(f"Error extracting EXIF metadata: {e}")
            return exif_metadata
    
    def _analyze_image(self, image) -> Dict[str, Any]:
        """
        Анализирует изображение для извлечения дополнительных метаданных.
        
        Args:
            image: Объект PIL Image
            
        Returns:
            Словарь с результатами анализа
        """
        analysis = {}
        
        try:
            # Преобразуем изображение в RGB, если необходимо
            if image.mode not in ["RGB", "L"]:
                try:
                    img_rgb = image.convert("RGB")
                except:
                    img_rgb = image
            else:
                img_rgb = image
            
            # Получаем гистограмму изображения
            hist = img_rgb.histogram()
            
            # Определяем, является ли изображение черно-белым
            if image.mode == "L" or len(set(image.getdata())) <= 2:
                analysis["is_bw"] = True
            else:
                analysis["is_bw"] = False
            
            # Определяем, содержит ли изображение текст (эвристика)
            text_probability = self._estimate_text_probability(img_rgb)
            analysis["text_probability"] = text_probability
            analysis["likely_contains_text"] = text_probability > 0.5
            
            # Определяем, является ли изображение сканом документа (эвристика)
            scan_probability = self._estimate_scan_probability(img_rgb)
            analysis["scan_probability"] = scan_probability
            analysis["likely_document_scan"] = scan_probability > 0.7
            
            return analysis
        except Exception as e:
            logger.warning(f"Error analyzing image: {e}")
            return analysis
    
    def _estimate_text_probability(self, image) -> float:
        """
        Оценивает вероятность наличия текста на изображении (эвристика).
        
        Args:
            image: Объект PIL Image
            
        Returns:
            Вероятность наличия текста (от 0 до 1)
        """
        try:
            # Преобразуем в оттенки серого
            gray_image = image.convert("L")
            
            # Применяем фильтр для выделения краев
            edges = gray_image.filter(ImageFilter.FIND_EDGES)
            
            # Получаем пиксели
            pixels = list(edges.getdata())
            
            # Подсчитываем количество краевых пикселей
            edge_pixels = sum(1 for p in pixels if p > 50)
            total_pixels = len(pixels)
            
            # Вычисляем долю краевых пикселей
            edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
            
            # Оцениваем вероятность наличия текста
            # Изображения с текстом обычно имеют много краев
            text_probability = min(1.0, edge_ratio * 5)
            
            return text_probability
        except Exception as e:
            logger.warning(f"Error estimating text probability: {e}")
            return 0.5
    
    def _estimate_scan_probability(self, image) -> float:
        """
        Оценивает вероятность того, что изображение является сканом документа (эвристика).
        
        Args:
            image: Объект PIL Image
            
        Returns:
            Вероятность того, что изображение является сканом (от 0 до 1)
        """
        try:
            # Преобразуем в оттенки серого
            gray_image = image.convert("L")
            
            # Получаем пиксели
            pixels = list(gray_image.getdata())
            
            # Подсчитываем количество белых и почти белых пикселей
            white_pixels = sum(1 for p in pixels if p > 200)
            total_pixels = len(pixels)
            
            # Вычисляем долю белых пикселей
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
            
            # Вычисляем соотношение сторон
            aspect_ratio = image.width / image.height if image.height > 0 else 0
            
            # Документы обычно имеют много белого пространства и стандартные соотношения сторон
            scan_probability = white_ratio * 0.7
            
            # Корректируем вероятность в зависимости от соотношения сторон
            # Стандартные соотношения для документов: A4 (1.414), Letter (1.294)
            if 1.2 < aspect_ratio < 1.5 or 0.66 < aspect_ratio < 0.85:
                scan_probability += 0.3
            
            return min(1.0, scan_probability)
        except Exception as e:
            logger.warning(f"Error estimating scan probability: {e}")
            return 0.5
    
    def _preprocess_image(self, image):
        """
        Предобрабатывает изображение для улучшения OCR.
        
        Args:
            image: Объект PIL Image
            
        Returns:
            Предобработанное изображение
        """
        try:
            # Преобразуем в оттенки серого
            gray_image = image.convert("L")
            
            # Увеличиваем контраст
            enhancer = ImageEnhance.Contrast(gray_image)
            contrast_image = enhancer.enhance(2.0)
            
            # Увеличиваем резкость
            sharpened = contrast_image.filter(ImageFilter.SHARPEN)
            
            # Для определенных типов изображений применяем бинаризацию
            if self._estimate_scan_probability(image) > 0.7:
                # Применяем пороговую обработку для документов
                threshold = 150
                binarized = sharpened.point(lambda p: 255 if p > threshold else 0)
                return binarized
            else:
                return sharpened
        except Exception as e:
            logger.warning(f"Error preprocessing image: {e}")
            return image
    
    def _perform_ocr(self, image, language: str = "rus+eng", dpi: int = 300) -> str:
        """
        Выполняет OCR для извлечения текста из изображения.
        
        Args:
            image: Объект PIL Image
            language: Язык OCR
            dpi: DPI для OCR
            
        Returns:
            Извлеченный текст
        """
        try:
            # Настраиваем путь к Tesseract
            pytesseract.pytesseract.tesseract_cmd = self._get_tesseract_path()
            
            # Опции OCR
            custom_config = f'--oem 3 --psm 6 -l {language} --dpi {dpi}'
            
            # Выполняем OCR
            text = pytesseract.image_to_string(image, config=custom_config)
            
            return text
        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
            raise
    
    def _parse_image_resize(self, resize_spec: str, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Разбирает спецификацию изменения размера изображения.
        
        Args:
            resize_spec: Спецификация размера ("widthxheight" или "percentage%")
            original_size: Исходный размер изображения
            
        Returns:
            Кортеж (ширина, высота)
        """
        orig_width, orig_height = original_size
        
        try:
            # Проверяем, является ли спецификация процентной
            if isinstance(resize_spec, str) and "%" in resize_spec:
                percentage = float(resize_spec.strip("%")) / 100.0
                width = int(orig_width * percentage)
                height = int(orig_height * percentage)
                return width, height
            
            # Проверяем, является ли спецификация конкретным размером
            if isinstance(resize_spec, str) and "x" in resize_spec:
                parts = resize_spec.lower().split("x")
                if len(parts) == 2:
                    width = int(parts[0]) if parts[0] else orig_width
                    height = int(parts[1]) if parts[1] else orig_height
                    return width, height
            
            # Если спецификация является числом, интерпретируем как максимальную ширину
            if isinstance(resize_spec, (int, float)) or resize_spec.isdigit():
                max_width = int(float(resize_spec))
                ratio = max_width / orig_width
                width = max_width
                height = int(orig_height * ratio)
                return width, height
            
            # По умолчанию возвращаем исходный размер
            return orig_width, orig_height
        except Exception as e:
            logger.warning(f"Error parsing resize specification: {e}")
            return orig_width, orig_height
    
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

ParserFactory.register_parser(ImageParser)