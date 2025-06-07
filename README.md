# SmartBot RAG Assistant

Универсальный чат-бот с индексацией и поиском данных на основе RAG (Retrieval Augmented Generation).

## 🚀 Возможности

- **📄 Индексация документов**: Поддержка множества форматов (PDF, DOCX, TXT, JSON, CSV, изображения и др.)
- **🔍 Семантический поиск**: Гибридный поиск с векторными представлениями
- **🤖 Множество LLM**: Поддержка Ollama, OpenAI, Anthropic, Google
- **🖼️ Мультимодальность**: Обработка изображений с OCR
- **💬 Контекстные диалоги**: Сохранение истории и контекста
- **🌐 Веб-интерфейс**: Удобный UI на Streamlit
- **🔧 REST API**: Полноценный API для интеграции
- **🌍 Переводчик**: Встроенная поддержка перевода

## 📋 Требования

- Python 3.11+
- 4GB RAM (рекомендуется 8GB)
- 2GB свободного места на диске
- Tesseract OCR (для обработки изображений)
- Ollama (для локальных моделей)

## 🛠️ Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/smartbot.git
cd smartbot
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Установка Tesseract OCR

**Windows:**
- Скачайте установщик с [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Установите и добавьте в PATH

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-rus
```

### 5. Установка Ollama (опционально)

Для работы с локальными моделями:

```bash
# Скачайте с https://ollama.ai
# Установите модель:
ollama pull llama3.1:8b
```

## ⚙️ Конфигурация

Отредактируйте `config/config.yaml`:

```yaml
models:
  default_provider: "ollama"  # или "openai", "anthropic", "google"
  
  openai:
    api_key: "${OPENAI_API_KEY}"  # Или укажите напрямую
    
  # ... другие настройки
```

### Переменные окружения

Создайте файл `.env`:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
DEEPL_API_KEY=your_deepl_key  # Опционально
```

## 🚀 Запуск

### Обычный запуск (API + UI)

```bash
python main.py
```

Откроется браузер с интерфейсом на http://localhost:8501

### Только API

```bash
python main.py --mode api
```

API будет доступен на http://localhost:8000

### Только UI

```bash
python main.py --mode ui
```

### Дополнительные параметры

```bash
python main.py --help

Опции:
  --mode {all,api,ui}  Режим запуска
  --no-browser         Не открывать браузер автоматически
  --api-port PORT      Порт для API (по умолчанию: 8000)
  --ui-port PORT       Порт для UI (по умолчанию: 8501)
  --debug              Режим отладки
```

## 📖 Использование

### Веб-интерфейс

1. **Загрузка документов**: Перейдите на вкладку "📤 Загрузка файлов"
2. **Задавайте вопросы**: Введите вопрос в поле чата
3. **Настройки**: Выберите модель и параметры в боковой панели

### API

#### Загрузка файлов

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "files=@document.pdf" \
  -F "files=@data.csv"
```

#### Отправка запроса

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Что такое машинное обучение?",
    "llm_settings": {
      "provider": "ollama",
      "model": "llama3.1:8b"
    }
  }'
```

#### Получение документов

```bash
curl http://localhost:8000/api/documents
```

## 📦 Сборка портативной версии

```bash
python build.py
```

Создаст архив в папке `release/` с исполняемым файлом.

## 🏗️ Архитектура

```
SmartBot/
├── app/
│   ├── api/          # REST API
│   ├── core/         # Ядро системы
│   ├── indexing/     # Индексация документов
│   ├── models/       # Работа с LLM
│   ├── rag/          # RAG engine
│   ├── ui/           # Веб-интерфейс
│   └── utils/        # Утилиты
├── config/           # Конфигурация
├── data/             # Данные (создается автоматически)
├── logs/             # Логи
└── main.py           # Точка входа
```

## 🔧 API Endpoints

- `POST /api/upload` - Загрузка файлов
- `GET /api/documents` - Список документов
- `POST /api/chat` - Отправка запроса
- `GET /api/history` - История диалогов
- `POST /api/reindex` - Переиндексация
- `GET /api/settings` - Получение настроек
- `PUT /api/settings` - Обновление настроек
- `DELETE /api/documents/{id}` - Удаление документа
- `POST /api/feedback` - Обратная связь
- `GET /api/status` - Статус системы

## 🐛 Решение проблем

### Ollama не запускается

```bash
# Проверьте, запущен ли Ollama
curl http://localhost:11434/api/tags

# Запустите Ollama
ollama serve
```

### OCR не работает

Убедитесь, что Tesseract установлен:

```bash
tesseract --version
```

### Ошибки с зависимостями

```bash
# Переустановите зависимости
pip install --upgrade -r requirements.txt
```

## 📝 Лицензия

MIT License

## 🤝 Вклад в проект

Приветствуются pull requests. Для больших изменений сначала откройте issue.

## 📧 Контакты

- GitHub: [https://github.com/yourusername/smartbot](https://github.com/yourusername/smartbot)
- Email: support@smartbot.ai

---

© 2025 SmartBot Team. Все права защищены.