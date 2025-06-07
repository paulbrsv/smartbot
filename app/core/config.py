import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic_settings import BaseSettings
from pydantic import validator

from app.core.models import (
    Settings, ModelsConfig, VectorStoreConfig, DatabaseConfig, 
    ProcessingConfig, OCRConfig, TranslationConfig, InterfaceConfig,
    ModelProvider, ModelConfig
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("smartbot.log"),
    ],
)

logger = logging.getLogger("smartbot")


class AppConfig:
    """
    Основной класс конфигурации приложения.
    Загружает настройки из YAML файла и переменных окружения.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._settings = None
        self.load_config()

    def load_config(self) -> None:
        """
        Загружает конфигурацию из YAML файла.
        """
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Конфигурационный файл {self.config_path} не найден. Используются настройки по умолчанию.")
                self._settings = self._create_default_settings()
                return

            with open(config_file, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                
            # Обработка переменных окружения в конфиге
            self._process_env_variables(yaml_config)
            
            # Преобразование в Pydantic модель
            self._settings = Settings(**yaml_config)
            logger.info(f"Конфигурация успешно загружена из {self.config_path}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            logger.info("Используются настройки по умолчанию")
            self._settings = self._create_default_settings()

    def _process_env_variables(self, config: Dict[str, Any]) -> None:
        """
        Обрабатывает переменные окружения в конфигурации.
        Заменяет строки формата ${ENV_VAR} на значения из переменных окружения.
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    self._process_env_variables(value)
                elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        config[key] = env_value
                    else:
                        logger.warning(f"Переменная окружения {env_var} не найдена")
        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, (dict, list)):
                    self._process_env_variables(item)
                elif isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                    env_var = item[2:-1]
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        config[i] = env_value
                    else:
                        logger.warning(f"Переменная окружения {env_var} не найдена")

    def _create_default_settings(self) -> Settings:
        """
        Создает настройки по умолчанию.
        """
        ollama_config = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="llama3.1:8b",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_p=0.95,
            max_tokens=2000,
        )

        openai_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            temperature=0.7,
            top_p=0.95,
            max_tokens=2000,
        )

        models_config = ModelsConfig(
            default_provider=ModelProvider.OLLAMA,
            ollama=ollama_config,
            openai=openai_config,
        )

        vector_store_config = VectorStoreConfig(
            type="chromadb",
            path="./data/vector_store",
            collection_name="documents",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
        )

        database_config = DatabaseConfig(
            type="sqlite",
            path="./data/app.db",
        )

        processing_config = ProcessingConfig(
            max_file_size_mb=500,
            chunk_size=1000,
            chunk_overlap=200,
            supported_extensions=[".txt", ".pdf", ".docx", ".json", ".csv", ".xml", ".html", ".md", ".rtf"],
        )

        ocr_config = OCRConfig(
            enabled=True,
            language="rus+eng",
            dpi=300,
        )

        translation_config = TranslationConfig(
            enabled=False,
            provider="google",
            target_language="ru",
        )

        interface_config = InterfaceConfig(
            theme="dark",
            language="ru",
            max_history=100,
        )

        return Settings(
            models=models_config,
            vector_store=vector_store_config,
            database=database_config,
            processing=processing_config,
            ocr=ocr_config,
            translation=translation_config,
            interface=interface_config,
        )

    @property
    def settings(self) -> Settings:
        """
        Возвращает текущие настройки.
        """
        return self._settings

    def save_config(self) -> None:
        """
        Сохраняет текущую конфигурацию в YAML файл.
        """
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Преобразование в словарь
            config_dict = self._settings.dict()
            
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Конфигурация успешно сохранена в {self.config_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {str(e)}")

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Обновляет текущие настройки и сохраняет их в файл.
        """
        try:
            # Обновление текущих настроек
            current_dict = self._settings.dict()
            self._update_dict_recursively(current_dict, new_settings)
            
            # Применение обновленных настроек
            self._settings = Settings(**current_dict)
            
            # Сохранение в файл
            self.save_config()
            
            logger.info("Настройки успешно обновлены")
        except Exception as e:
            logger.error(f"Ошибка обновления настроек: {str(e)}")

    def _update_dict_recursively(self, current: Dict[str, Any], new: Dict[str, Any]) -> None:
        """
        Рекурсивно обновляет словарь настроек.
        """
        for key, value in new.items():
            if key in current and isinstance(current[key], dict) and isinstance(value, dict):
                self._update_dict_recursively(current[key], value)
            else:
                current[key] = value


# Создаем экземпляр конфигурации
config = AppConfig()

def get_settings() -> Settings:
    """
    Функция для получения текущих настроек приложения.
    
    Returns:
        Settings: Объект с настройками приложения
    """
    return config.settings