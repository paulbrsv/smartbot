"""
Скрипт для сборки SmartBot в исполняемый файл.
Использует PyInstaller для создания портативной версии.
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path

# Определяем платформу
PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM == "windows"
IS_MACOS = PLATFORM == "darwin"
IS_LINUX = PLATFORM == "linux"

# Пути
BASE_DIR = Path(__file__).parent
BUILD_DIR = BASE_DIR / "build"
DIST_DIR = BASE_DIR / "dist"
RELEASE_DIR = BASE_DIR / "release"


def clean_build():
    """Очистка предыдущих сборок"""
    print("Cleaning previous builds...")
    
    for directory in [BUILD_DIR, DIST_DIR, RELEASE_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
            print(f"Removed {directory}")


def create_spec_file():
    """Создание spec файла для PyInstaller"""
    print("Creating PyInstaller spec file...")
    
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Пути
BASE_DIR = Path(r'{BASE_DIR}')

# Анализ
a = Analysis(
    [str(BASE_DIR / 'main.py')],
    pathex=[str(BASE_DIR)],
    binaries=[],
    datas=[
        (str(BASE_DIR / 'config'), 'config'),
        (str(BASE_DIR / 'app'), 'app'),
    ],
    hiddenimports=[
        'streamlit',
        'streamlit.web.cli',
        'streamlit.runtime.scriptrunner.magic_funcs',
        'fastapi',
        'uvicorn',
        'langchain',
        'langchain_community',
        'chromadb',
        'chromadb.api',
        'chromadb.config',
        'ollama',
        'sentence_transformers',
        'transformers',
        'torch',
        'torchvision',
        'PIL',
        'pytesseract',
        'pandas',
        'openpyxl',
        'PyMuPDF',
        'fitz',
        'docx',
        'bs4',
        'lxml',
        'sqlalchemy',
        'pydantic',
        'yaml',
        'tqdm',
        'aiohttp',
        'requests',
        'numpy',
        'sklearn',
        'jsonpath_ng',
        'tenacity',
        'multipart',
        'starlette',
        'anyio',
        'sniffio',
        'httpcore',
        'httpx',
        'click',
        'watchdog',
        'altair',
        'plotly',
        'matplotlib',
        'seaborn',
        'bokeh',
        'pyarrow',
        'validators',
        'toml',
        'gitpython',
        'pympler',
        'tornado',
        'pytz',
        'tzlocal',
        'protobuf',
        'grpcio',
        'packaging',
        'filelock',
        'huggingface_hub',
        'safetensors',
        'tokenizers',
        'accelerate',
        'datasets',
        'evaluate',
        'scipy',
        'scikit-learn',
        'joblib',
        'threadpoolctl',
        'regex',
        'ftfy',
        'wcwidth',
        'sympy',
        'mpmath',
        'networkx',
        'tifffile',
        'imageio',
        'lazy_loader',
        'contourpy',
        'cycler',
        'fonttools',
        'kiwisolver',
        'pyparsing',
        'python-dateutil',
        'pyproject_hooks',
        'build',
        'installer',
        'wheel',
        'setuptools',
        'certifi',
        'charset-normalizer',
        'idna',
        'urllib3',
        'cryptography',
        'cffi',
        'pycparser',
        'six',
        'attrs',
        'jsonschema',
        'pyrsistent',
        'importlib-metadata',
        'zipp',
        'typing-extensions',
        'mypy-extensions',
        'pathspec',
        'platformdirs',
        'tomli',
        'exceptiongroup',
        'iniconfig',
        'pluggy',
        'colorama',
        'tqdm',
        'rich',
        'typer',
        'shellingham',
        'questionary',
        'prompt-toolkit',
        'pygments',
        'markdown-it-py',
        'mdurl',
        'linkify-it-py',
        'mdit-py-plugins',
        'textual',
        'httptools',
        'websockets',
        'watchfiles',
        'python-multipart',
        'email-validator',
        'dnspython',
        'orjson',
        'ujson',
        'itsdangerous',
        'python-jose',
        'passlib',
        'bcrypt',
        'python-dotenv',
        'environs',
        'marshmallow',
        'apispec',
        'webargs',
        'flask',
        'werkzeug',
        'jinja2',
        'markupsafe',
        'blinker',
        'cachetools',
        'google-auth',
        'google-auth-oauthlib',
        'google-auth-httplib2',
        'google-api-core',
        'google-api-python-client',
        'googleapis-common-protos',
        'rsa',
        'pyasn1',
        'pyasn1-modules',
        'oauthlib',
        'requests-oauthlib',
        'httplib2',
        'uritemplate',
        'google-cloud-core',
        'google-cloud-storage',
        'google-resumable-media',
        'google-crc32c',
        'proto-plus',
        'grpc-google-iam-v1',
        'grpcio-status',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'test',
        'tests',
        'testing',
        'pytest',
        'nose',
        'ipython',
        'jupyter',
        'notebook',
        'ipykernel',
        'ipywidgets',
        'nbformat',
        'nbconvert',
        'sphinx',
        'docutils',
        'alabaster',
        'babel',
        'imagesize',
        'snowballstemmer',
        'sphinx-rtd-theme',
        'recommonmark',
        'numpydoc',
        'scipy.weave',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Сборка PYZ
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# Создание EXE
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SmartBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# Создание директории с приложением
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SmartBot',
)
"""
    
    spec_file = BASE_DIR / "SmartBot.spec"
    with open(spec_file, "w", encoding="utf-8") as f:
        f.write(spec_content)
    
    print(f"Created {spec_file}")
    return spec_file


def build_executable(spec_file):
    """Сборка исполняемого файла"""
    print("Building executable with PyInstaller...")
    
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file)
    ]
    
    result = subprocess.run(cmd, cwd=BASE_DIR)
    
    if result.returncode != 0:
        print("ERROR: PyInstaller build failed!")
        sys.exit(1)
    
    print("Build completed successfully!")


def create_release():
    """Создание релизной версии"""
    print("Creating release package...")
    
    # Определяем имя релиза
    if IS_WINDOWS:
        release_name = "SmartBot_v1.0_Windows"
        exe_name = "SmartBot.exe"
    elif IS_MACOS:
        release_name = "SmartBot_v1.0_macOS"
        exe_name = "SmartBot"
    else:
        release_name = "SmartBot_v1.0_Linux"
        exe_name = "SmartBot"
    
    # Создаем директорию релиза
    release_path = RELEASE_DIR / release_name
    release_path.mkdir(parents=True, exist_ok=True)
    
    # Копируем исполняемый файл
    dist_exe = DIST_DIR / "SmartBot" / exe_name
    if dist_exe.exists():
        shutil.copy2(dist_exe, release_path / exe_name)
    else:
        print(f"ERROR: Executable not found at {dist_exe}")
        sys.exit(1)
    
    # Копируем конфигурационные файлы
    config_dir = release_path / "config"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(BASE_DIR / "config" / "config.yaml", config_dir / "config.yaml")
    
    # Создаем пустые директории
    for dir_name in ["data", "logs", "models"]:
        (release_path / dir_name).mkdir(exist_ok=True)
    
    # Создаем README
    readme_content = f"""# SmartBot RAG Assistant v1.0

## Универсальный чат-бот с индексацией и поиском данных

### Системные требования:
- Операционная система: {platform.system()} {platform.release()}
- RAM: минимум 4GB (рекомендуется 8GB)
- Свободное место на диске: минимум 2GB

### Установка и запуск:

1. Распакуйте архив в любую папку
2. Запустите {exe_name}
3. Откройте браузер и перейдите по адресу http://localhost:8501

### Первоначальная настройка:

1. Установите Ollama для работы с локальными моделями:
   - Скачайте с https://ollama.ai
   - Установите модель: `ollama pull llama3.1:8b`

2. Для работы OCR установите Tesseract:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

3. Настройте API ключи в config/config.yaml для внешних моделей

### Структура папок:

```
{release_name}/
├── {exe_name}          # Исполняемый файл
├── config/             # Конфигурационные файлы
│   └── config.yaml     # Основная конфигурация
├── data/               # Данные приложения
├── logs/               # Логи
└── models/             # Локальные модели (опционально)
```

### Использование:

1. **Загрузка документов**: Используйте вкладку "Загрузка файлов" для добавления документов
2. **Задавайте вопросы**: Введите вопрос в поле чата
3. **Настройки**: Выберите модель и параметры в боковой панели

### Поддерживаемые форматы файлов:

- Документы: PDF, DOCX, DOC, TXT, RTF
- Данные: JSON, XML, CSV, XLSX
- Изображения: JPG, PNG, TIFF (с OCR)
- Веб: HTML, Markdown
- Архивы: ZIP, RAR

### Решение проблем:

- Если приложение не запускается, проверьте наличие свободных портов 8000 и 8501
- Логи находятся в папке logs/
- При проблемах с OCR убедитесь, что Tesseract установлен корректно

### Поддержка:

GitHub: https://github.com/smartbot/smartbot
Email: support@smartbot.ai

---
© 2025 SmartBot Team. All rights reserved.
"""
    
    with open(release_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Создаем скрипт запуска для Linux/macOS
    if not IS_WINDOWS:
        run_script = f"""#!/bin/bash
cd "$(dirname "$0")"
./{exe_name}
"""
        run_file = release_path / "run.sh"
        with open(run_file, "w") as f:
            f.write(run_script)
        os.chmod(run_file, 0o755)
    
    # Создаем архив
    print(f"Creating archive {release_name}.zip...")
    shutil.make_archive(
        RELEASE_DIR / release_name,
        'zip',
        RELEASE_DIR,
        release_name
    )
    
    print(f"Release created: {RELEASE_DIR / release_name}.zip")
    
    # Показываем размер
    size_mb = (RELEASE_DIR / f"{release_name}.zip").stat().st_size / (1024 * 1024)
    print(f"Release size: {size_mb:.1f} MB")


def main():
    """Основная функция сборки"""
    print("=" * 50)
    print("SmartBot Build Script")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("=" * 50)
    
    # Проверяем наличие PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("ERROR: PyInstaller not found!")
        print("Install it with: pip install pyinstaller")
        sys.exit(1)
    
    # Очистка
    clean_build()
    
    # Создание spec файла
    spec_file = create_spec_file()
    
    # Сборка
    build_executable(spec_file)
    
    # Создание релиза
    create_release()
    
    print("\n" + "=" * 50)
    print("BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()