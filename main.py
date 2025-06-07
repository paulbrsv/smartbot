"""
Главный файл приложения SmartBot.
Точка входа для запуска системы.
"""

import os
import sys
import logging
import argparse
import threading
import webbrowser
import time
from pathlib import Path

# Добавляем корневую директорию в путь Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import get_settings
from app.core.database import init_database
from app.api.main import run_api
import streamlit.web.cli as stcli

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получаем настройки
settings = get_settings()


def ensure_directories():
    """Создание необходимых директорий"""
    directories = [
        Path.home() / ".smartbot",
        Path.home() / ".smartbot" / "data",
        Path.home() / ".smartbot" / "uploads",
        Path.home() / ".smartbot" / "temp",
        Path.home() / ".smartbot" / "temp" / "images",
        Path.home() / ".smartbot" / "logs",
        Path("data"),
        Path("data") / "vector_store",
        Path("logs")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def run_streamlit():
    """Запуск Streamlit UI"""
    logger.info("Starting Streamlit UI...")
    
    # Путь к файлу Streamlit приложения
    streamlit_file = os.path.join(
        os.path.dirname(__file__),
        "app",
        "ui",
        "streamlit_app.py"
    )
    
    # Запускаем Streamlit
    sys.argv = [
        "streamlit",
        "run",
        streamlit_file,
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    sys.exit(stcli.main())


def run_api_server():
    """Запуск API сервера в отдельном потоке"""
    logger.info("Starting API server...")
    run_api()


def check_dependencies():
    """Проверка зависимостей"""
    required_modules = [
        "fastapi",
        "streamlit",
        "langchain",
        "chromadb",
        "ollama",
        "torch",
        "transformers",
        "sentence_transformers",
        "pytesseract",
        "PIL"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {', '.join(missing_modules)}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False
    
    # Проверяем Tesseract OCR
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        logger.warning("Tesseract OCR not found. OCR features will be disabled.")
        logger.warning("Install Tesseract: https://github.com/tesseract-ocr/tesseract")
    
    return True


def check_ollama():
    """Проверка доступности Ollama"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                logger.info(f"Ollama is running with {len(models)} models")
                return True
            else:
                logger.warning("Ollama is running but no models are installed")
                logger.warning("Install a model: ollama pull llama3.1:8b")
                return True
        else:
            logger.warning("Ollama is not responding properly")
            return False
    except Exception:
        logger.warning("Ollama is not running. Local models will not be available.")
        logger.warning("Start Ollama: ollama serve")
        return False


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="SmartBot RAG Assistant")
    parser.add_argument(
        "--mode",
        choices=["all", "api", "ui"],
        default="all",
        help="Run mode: all (API + UI), api (only API), ui (only UI)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8501,
        help="UI server port (default: 8501)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # ASCII арт логотип
    logo = """
    ╔═══════════════════════════════════════╗
    ║      SmartBot RAG Assistant v1.0      ║
    ║   Universal Chatbot with RAG Search   ║
    ╚═══════════════════════════════════════╝
    """
    print(logo)
    
    # Проверка зависимостей
    logger.info("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Создание директорий
    logger.info("Creating necessary directories...")
    ensure_directories()
    
    # Инициализация базы данных
    logger.info("Initializing database...")
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    
    # Проверка Ollama
    check_ollama()
    
    # Запуск компонентов в зависимости от режима
    if args.mode == "all":
        # Запускаем API в отдельном потоке
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        
        # Ждем запуска API
        logger.info("Waiting for API to start...")
        time.sleep(3)
        
        # Открываем браузер
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.ui_port}")
        
        # Запускаем UI в главном потоке
        run_streamlit()
        
    elif args.mode == "api":
        # Только API
        logger.info("Running in API-only mode")
        run_api_server()
        
    elif args.mode == "ui":
        # Только UI
        logger.info("Running in UI-only mode")
        logger.warning("Make sure API is running separately!")
        
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.ui_port}")
        
        run_streamlit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down SmartBot...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)