"""
Веб-интерфейс SmartBot на Streamlit.
Предоставляет удобный UI для взаимодействия с системой.
"""

import streamlit as st
import requests
import json
import os
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image
import io

# Конфигурация страницы
st.set_page_config(
    page_title="SmartBot RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL API
API_URL = os.getenv("SMARTBOT_API_URL", "http://localhost:8000")

# Инициализация состояния сессии
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_api_status():
    """Проверка статуса API"""
    try:
        response = requests.get(f"{API_URL}/api/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def upload_files(files):
    """Загрузка файлов через API"""
    try:
        files_data = []
        for file in files:
            files_data.append(
                ("files", (file.name, file.getvalue(), file.type))
            )
        
        response = requests.post(
            f"{API_URL}/api/upload",
            files=files_data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def send_message(query: str, images: Optional[List[Any]] = None):
    """Отправка сообщения в чат"""
    try:
        # Подготавливаем данные запроса
        request_data = {
            "query": query,
            "chat_history": st.session_state.chat_history[-10:],  # Последние 10 сообщений
            "llm_settings": {
                "provider": st.session_state.get("provider", "ollama"),
                "model": st.session_state.get("model", "llama3.1:8b"),
                "temperature": st.session_state.get("temperature", 0.7)
            }
        }
        
        # Если есть изображения, конвертируем их в base64
        if images:
            image_data = []
            for img in images:
                if isinstance(img, bytes):
                    image_data.append(base64.b64encode(img).decode())
                elif hasattr(img, "read"):
                    image_data.append(base64.b64encode(img.read()).decode())
            request_data["image_data"] = image_data
        
        # Отправляем запрос
        response = requests.post(
            f"{API_URL}/api/chat",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Chat request failed: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def get_documents():
    """Получение списка документов"""
    try:
        response = requests.get(f"{API_URL}/api/documents")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


def delete_document(doc_id: int):
    """Удаление документа"""
    try:
        response = requests.delete(f"{API_URL}/api/documents/{doc_id}")
        return response.status_code == 200
    except:
        return False


def submit_feedback(query_id: str, rating: int, comment: str = ""):
    """Отправка обратной связи"""
    try:
        response = requests.post(
            f"{API_URL}/api/feedback",
            json={
                "query_id": query_id,
                "rating": rating,
                "comment": comment
            }
        )
        return response.status_code == 200
    except:
        return False


# Основной интерфейс
def main():
    # Заголовок
    st.title("🤖 SmartBot RAG Assistant")
    st.markdown("Универсальный чат-бот с индексацией и поиском данных")
    
    # Проверка статуса API
    api_status = get_api_status()
    if not api_status:
        st.error("⚠️ API недоступен. Убедитесь, что сервер запущен.")
        return
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Выбор модели
        st.subheader("Языковая модель")
        available_models = api_status.get("available_models", {})
        
        provider = st.selectbox(
            "Провайдер",
            options=list(available_models.keys()),
            index=0 if available_models else None
        )
        
        if provider and provider in available_models:
            model = st.selectbox(
                "Модель",
                options=available_models[provider],
                index=0 if available_models[provider] else None
            )
            st.session_state.provider = provider
            st.session_state.model = model
        
        # Параметры генерации
        st.subheader("Параметры генерации")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1
        )
        st.session_state.temperature = temperature
        
        # Статистика
        st.subheader("📊 Статистика")
        stats = api_status.get("statistics", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Документов", stats.get("total_documents", 0))
            st.metric("Запросов", stats.get("total_queries", 0))
        with col2:
            st.metric("Без ответа", stats.get("unanswered_queries", 0))
            st.metric("Векторная БД", api_status.get("vector_store", {}).get("status", "unknown"))
        
        # Управление документами
        st.subheader("📁 Документы")
        if st.button("🔄 Обновить список"):
            st.rerun()
        
        documents = get_documents()
        if documents:
            st.write(f"Всего документов: {len(documents)}")
            
            # Показываем последние 5 документов
            for doc in documents[:5]:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    st.write(f"**Тип:** {doc['filetype']}")
                    st.write(f"**Размер:** {doc['size_bytes'] / 1024:.1f} KB")
                    st.write(f"**Индексирован:** {doc['indexed_at']}")
                    if st.button(f"🗑️ Удалить", key=f"del_{doc['id']}"):
                        if delete_document(doc['id']):
                            st.success("Документ удален")
                            st.rerun()
    
    # Основная область
    tabs = st.tabs(["💬 Чат", "📤 Загрузка файлов", "📜 История", "ℹ️ О системе"])
    
    # Вкладка чата
    with tabs[0]:
        # Область сообщений
        messages_container = st.container()
        
        # Отображаем историю сообщений
        with messages_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Если есть источники, показываем их
                    if message.get("sources"):
                        with st.expander("📚 Источники"):
                            for source in message["sources"]:
                                st.write(f"- {source.get('title', 'Документ')}")
                    
                    # Если есть метаданные с query_id, показываем кнопки обратной связи
                    if message.get("metadata", {}).get("query_id"):
                        query_id = message["metadata"]["query_id"]
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            if st.button("👍", key=f"like_{query_id}"):
                                submit_feedback(query_id, 5)
                                st.success("Спасибо за отзыв!")
                        with col2:
                            if st.button("👎", key=f"dislike_{query_id}"):
                                submit_feedback(query_id, 1)
                                st.success("Спасибо за отзыв!")
        
        # Форма ввода
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "Введите ваш вопрос:",
                    height=100,
                    placeholder="Например: Что такое машинное обучение?"
                )
            
            with col2:
                uploaded_images = st.file_uploader(
                    "Изображения",
                    type=["png", "jpg", "jpeg", "gif", "bmp"],
                    accept_multiple_files=True,
                    label_visibility="collapsed"
                )
                
                submit_button = st.form_submit_button(
                    "📤 Отправить",
                    use_container_width=True
                )
            
            if submit_button and user_input:
                # Добавляем сообщение пользователя
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Показываем индикатор загрузки
                with st.spinner("🤔 Обрабатываю запрос..."):
                    # Отправляем запрос
                    response = send_message(user_input, uploaded_images)
                    
                    if "error" in response:
                        st.error(f"Ошибка: {response['error']}")
                    else:
                        # Добавляем ответ ассистента
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["response"],
                            "sources": response.get("sources", []),
                            "metadata": response.get("metadata", {})
                        })
                        
                        # Добавляем в историю чата
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_input
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response["response"]
                        })
                
                st.rerun()
    
    # Вкладка загрузки файлов
    with tabs[1]:
        st.header("📤 Загрузка файлов")
        st.markdown("""
        Загрузите файлы для индексации. Поддерживаемые форматы:
        - Документы: PDF, DOCX, DOC, TXT, RTF
        - Данные: JSON, XML, CSV, XLSX
        - Изображения: JPG, PNG, TIFF (с OCR)
        - Веб: HTML, Markdown
        - Архивы: ZIP, RAR
        """)
        
        uploaded_files = st.file_uploader(
            "Выберите файлы",
            accept_multiple_files=True,
            type=["pdf", "docx", "doc", "txt", "rtf", "json", "xml", 
                  "csv", "xlsx", "jpg", "jpeg", "png", "tiff", "html", 
                  "md", "zip", "rar"]
        )
        
        if st.button("📤 Загрузить файлы", disabled=not uploaded_files):
            with st.spinner("Загрузка файлов..."):
                result = upload_files(uploaded_files)
                
                if "error" in result:
                    st.error(f"Ошибка: {result['error']}")
                else:
                    st.success("Файлы успешно загружены!")
                    
                    # Показываем результаты
                    for file_result in result.get("files", []):
                        if file_result["status"] == "queued":
                            st.info(f"✅ {file_result['filename']} - добавлен в очередь индексации")
                        else:
                            st.error(f"❌ {file_result['filename']} - {file_result['message']}")
    
    # Вкладка истории
    with tabs[2]:
        st.header("📜 История запросов")
        
        # Получаем историю
        try:
            response = requests.get(f"{API_URL}/api/history")
            if response.status_code == 200:
                history = response.json()
                
                if history:
                    # Преобразуем в DataFrame для удобного отображения
                    df = pd.DataFrame(history)
                    
                    # Фильтры
                    col1, col2 = st.columns(2)
                    with col1:
                        search_query = st.text_input("🔍 Поиск по запросам")
                    with col2:
                        show_rated = st.checkbox("Только с оценками", value=False)
                    
                    # Применяем фильтры
                    if search_query:
                        df = df[df["query"].str.contains(search_query, case=False)]
                    if show_rated:
                        df = df[df["rating"].notna()]
                    
                    # Отображаем историю
                    for _, row in df.iterrows():
                        with st.expander(f"❓ {row['query'][:100]}...", expanded=False):
                            st.write(f"**Ответ:** {row['response'][:500]}...")
                            st.write(f"**Дата:** {row['created_at']}")
                            if row.get("rating"):
                                st.write(f"**Оценка:** {'⭐' * int(row['rating'])}")
                else:
                    st.info("История пуста")
            else:
                st.error("Не удалось загрузить историю")
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
    
    # Вкладка информации
    with tabs[3]:
        st.header("ℹ️ О системе")
        st.markdown("""
        ### SmartBot RAG Assistant
        
        **Версия:** 1.0.0
        
        **Возможности:**
        - 🔍 Семантический поиск по документам
        - 🤖 Генерация ответов с использованием RAG
        - 📄 Поддержка множества форматов файлов
        - 🖼️ Обработка изображений с OCR
        - 💬 Контекстные диалоги
        - 🌐 Поддержка различных языковых моделей
        
        **Горячие клавиши:**
        - `Ctrl+Enter` - отправить сообщение
        - `Esc` - очистить поле ввода
        
        **Полезные команды в чате:**
        - `/help` - справка по командам
        - `/status` - статус системы
        - `/clear` - очистить историю
        - `/settings` - показать настройки
        """)
        
        # Кнопка очистки истории чата
        if st.button("🗑️ Очистить историю чата"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("История очищена")
            st.rerun()


if __name__ == "__main__":
    main()