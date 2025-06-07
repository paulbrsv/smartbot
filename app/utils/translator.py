"""
Модуль для перевода текста.
Поддерживает различные провайдеры перевода.
"""

import logging
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
import os
import json

# Опциональные зависимости для разных провайдеров
try:
    from googletrans import Translator as GoogleTranslator
    GOOGLE_TRANS_AVAILABLE = True
except ImportError:
    GOOGLE_TRANS_AVAILABLE = False

try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseTranslator(ABC):
    """Базовый класс для переводчиков"""
    
    def __init__(self):
        self.name = "base"
        self.supported_languages = []
    
    @abstractmethod
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> str:
        """
        Перевести текст на целевой язык.
        
        Args:
            text: Текст для перевода
            target_language: Целевой язык (код ISO 639-1)
            source_language: Исходный язык (если None, определяется автоматически)
            
        Returns:
            Переведенный текст
        """
        pass
    
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """
        Определить язык текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Код языка (ISO 639-1)
        """
        pass
    
    def is_language_supported(self, language_code: str) -> bool:
        """Проверить, поддерживается ли язык"""
        return language_code.lower() in self.supported_languages
    
    def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None
    ) -> List[str]:
        """
        Перевести несколько текстов.
        
        Args:
            texts: Список текстов для перевода
            target_language: Целевой язык
            source_language: Исходный язык
            
        Returns:
            Список переведенных текстов
        """
        results = []
        for text in texts:
            try:
                translated = self.translate(text, target_language, source_language)
                results.append(translated)
            except Exception as e:
                logger.error(f"Error translating text: {e}")
                results.append(text)  # Возвращаем оригинал при ошибке
        return results


class GoogleTranslateProvider(BaseTranslator):
    """Провайдер Google Translate"""
    
    def __init__(self):
        super().__init__()
        self.name = "google"
        
        if not GOOGLE_TRANS_AVAILABLE:
            raise ImportError("googletrans library is not installed")
        
        self.translator = GoogleTranslator()
        
        # Поддерживаемые языки Google Translate
        self.supported_languages = [
            'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs',
            'bg', 'ca', 'ceb', 'ny', 'zh-cn', 'zh-tw', 'co', 'hr', 'cs',
            'da', 'nl', 'en', 'eo', 'et', 'tl', 'fi', 'fr', 'fy', 'gl',
            'ka', 'de', 'el', 'gu', 'ht', 'ha', 'haw', 'iw', 'hi', 'hmn',
            'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jw', 'kn', 'kk',
            'km', 'ko', 'ku', 'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk',
            'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 'my', 'ne', 'no',
            'ps', 'fa', 'pl', 'pt', 'pa', 'ro', 'ru', 'sm', 'gd', 'sr',
            'st', 'sn', 'sd', 'si', 'sk', 'sl', 'so', 'es', 'su', 'sw',
            'sv', 'tg', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'uz', 'vi',
            'cy', 'xh', 'yi', 'yo', 'zu'
        ]
    
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> str:
        """Перевести текст через Google Translate"""
        try:
            # Проверяем поддержку языка
            if not self.is_language_supported(target_language):
                raise ValueError(f"Language {target_language} is not supported")
            
            # Переводим
            if source_language:
                result = self.translator.translate(
                    text,
                    src=source_language,
                    dest=target_language
                )
            else:
                result = self.translator.translate(
                    text,
                    dest=target_language
                )
            
            return result.text
        except Exception as e:
            logger.error(f"Google Translate error: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Определить язык текста"""
        try:
            detection = self.translator.detect(text)
            return detection.lang
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "unknown"


class DeepLProvider(BaseTranslator):
    """Провайдер DeepL"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.name = "deepl"
        
        if not DEEPL_AVAILABLE:
            raise ImportError("deepl library is not installed")
        
        # Получаем API ключ
        self.api_key = api_key or os.getenv("DEEPL_API_KEY")
        if not self.api_key:
            raise ValueError("DeepL API key is required")
        
        self.translator = deepl.Translator(self.api_key)
        
        # Поддерживаемые языки DeepL
        self.supported_languages = [
            'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr',
            'hu', 'id', 'it', 'ja', 'ko', 'lt', 'lv', 'nb', 'nl', 'pl',
            'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'tr', 'uk', 'zh'
        ]
    
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> str:
        """Перевести текст через DeepL"""
        try:
            # Проверяем поддержку языка
            if not self.is_language_supported(target_language):
                raise ValueError(f"Language {target_language} is not supported")
            
            # Переводим
            result = self.translator.translate_text(
                text,
                target_lang=target_language.upper(),
                source_lang=source_language.upper() if source_language else None
            )
            
            return result.text
        except Exception as e:
            logger.error(f"DeepL error: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Определить язык текста"""
        try:
            # DeepL не имеет отдельной функции определения языка
            # Используем перевод с автоопределением
            result = self.translator.translate_text(text, target_lang="EN-US")
            return result.detected_source_lang.lower()
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "unknown"


class LocalTranslator(BaseTranslator):
    """Локальный переводчик (заглушка для будущей реализации)"""
    
    def __init__(self):
        super().__init__()
        self.name = "local"
        
        # Пока поддерживаем только английский и русский
        self.supported_languages = ['en', 'ru']
        
        # Простой словарь для демонстрации
        self.dictionary = {
            "hello": "привет",
            "world": "мир",
            "computer": "компьютер",
            "file": "файл",
            "document": "документ",
            "search": "поиск",
            "question": "вопрос",
            "answer": "ответ"
        }
    
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> str:
        """Простой перевод по словарю"""
        if target_language == 'ru' and (source_language == 'en' or source_language is None):
            # Переводим слова из словаря
            words = text.lower().split()
            translated_words = []
            
            for word in words:
                if word in self.dictionary:
                    translated_words.append(self.dictionary[word])
                else:
                    translated_words.append(word)
            
            return ' '.join(translated_words)
        else:
            # Для других языков возвращаем оригинал
            return text
    
    def detect_language(self, text: str) -> str:
        """Простое определение языка"""
        # Проверяем наличие кириллицы
        if any('а' <= char <= 'я' or 'А' <= char <= 'Я' for char in text):
            return 'ru'
        else:
            return 'en'


class TranslatorFactory:
    """Фабрика для создания переводчиков"""
    
    @staticmethod
    def create_translator(
        provider: str = "google",
        api_key: Optional[str] = None
    ) -> BaseTranslator:
        """
        Создать переводчик указанного типа.
        
        Args:
            provider: Тип провайдера (google, deepl, local)
            api_key: API ключ (если требуется)
            
        Returns:
            Экземпляр переводчика
        """
        provider = provider.lower()
        
        if provider == "google":
            if not GOOGLE_TRANS_AVAILABLE:
                raise ImportError("Google Translate is not available. Install googletrans-new")
            return GoogleTranslateProvider()
        
        elif provider == "deepl":
            if not DEEPL_AVAILABLE:
                raise ImportError("DeepL is not available. Install deepl")
            return DeepLProvider(api_key)
        
        elif provider == "local":
            return LocalTranslator()
        
        else:
            raise ValueError(f"Unknown translator provider: {provider}")


class TranslationService:
    """Сервис для работы с переводами в приложении"""
    
    def __init__(self):
        """Инициализация сервиса перевода"""
        self.enabled = settings.translation.enabled if hasattr(settings, 'translation') else False
        self.provider = settings.translation.provider if hasattr(settings, 'translation') else "google"
        self.target_language = settings.translation.target_language if hasattr(settings, 'translation') else "ru"
        
        self.translator = None
        if self.enabled:
            try:
                self.translator = TranslatorFactory.create_translator(self.provider)
                logger.info(f"Translation service initialized with {self.provider} provider")
            except Exception as e:
                logger.error(f"Failed to initialize translation service: {e}")
                self.enabled = False
    
    def translate_if_needed(
        self,
        text: str,
        force: bool = False
    ) -> str:
        """
        Перевести текст, если включен перевод.
        
        Args:
            text: Текст для перевода
            force: Принудительный перевод
            
        Returns:
            Переведенный текст или оригинал
        """
        if not self.enabled and not force:
            return text
        
        if not self.translator:
            return text
        
        try:
            # Определяем язык текста
            detected_lang = self.translator.detect_language(text)
            
            # Если язык уже целевой, не переводим
            if detected_lang == self.target_language:
                return text
            
            # Переводим
            translated = self.translator.translate(
                text,
                self.target_language,
                detected_lang
            )
            
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def translate_document(
        self,
        document: Dict[str, Any],
        fields: List[str] = ["content", "title", "description"]
    ) -> Dict[str, Any]:
        """
        Перевести поля документа.
        
        Args:
            document: Документ для перевода
            fields: Список полей для перевода
            
        Returns:
            Документ с переведенными полями
        """
        if not self.enabled or not self.translator:
            return document
        
        translated_doc = document.copy()
        
        for field in fields:
            if field in translated_doc and isinstance(translated_doc[field], str):
                try:
                    translated_doc[field] = self.translate_if_needed(
                        translated_doc[field]
                    )
                except Exception as e:
                    logger.error(f"Error translating field {field}: {e}")
        
        return translated_doc
    
    def get_supported_languages(self) -> List[str]:
        """Получить список поддерживаемых языков"""
        if self.translator:
            return self.translator.supported_languages
        return []
    
    def is_available(self) -> bool:
        """Проверить доступность сервиса перевода"""
        return self.enabled and self.translator is not None


# Глобальный экземпляр сервиса перевода
translation_service = TranslationService()


def translate_text(text: str, target_language: Optional[str] = None) -> str:
    """
    Удобная функция для перевода текста.
    
    Args:
        text: Текст для перевода
        target_language: Целевой язык (если не указан, используется из настроек)
        
    Returns:
        Переведенный текст
    """
    if target_language:
        # Создаем временный переводчик для конкретного языка
        try:
            translator = TranslatorFactory.create_translator(
                translation_service.provider
            )
            return translator.translate(text, target_language)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    else:
        # Используем глобальный сервис
        return translation_service.translate_if_needed(text)