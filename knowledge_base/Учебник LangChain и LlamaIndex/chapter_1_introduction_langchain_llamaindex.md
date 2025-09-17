# Глава 1: Введение в LangChain и LlamaIndex

Добро пожаловать в первую главу учебника, где мы начнем с основ LangChain и LlamaIndex. Эта глава познакомит вас с ключевыми концепциями этих фреймворков, их ролью в создании современных AI-систем и востребованностью на рынке труда в России в 2025 году. Материал структурирован для интеграции в RAG-базу знаний чат-бота с ролью Виртуального Ментора, а примеры и упражнения помогут вам закрепить знания, как специалистов Data Science.

## 1.1 Что такое LangChain?

LangChain — это фреймворк для разработки приложений, использующих большие языковые модели (LLM), с акцентом на интеграцию внешних данных и инструментов. Он позволяет создавать цепочки (Chains), агентов (Agents) и использовать контекстную память (Memory) для обработки сложных задач. Навыки работы с LangChain высоко ценятся в задачах автоматизации, таких как чат-боты и аналитика текстов.

**Пример**: Создание простой цепочки в LangChain для генерации ответа на вопрос с использованием модели OpenAI.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(input_variables=["question"], template="Ответь на вопрос: {question}")
chain = prompt | llm
response = chain.invoke({"question": "Что такое RAG (Retrieval-Augmented Generation)?"})
print(response.content)
```

## 1.2 Что такое LlamaIndex?

LlamaIndex — это инструмент для эффективной индексации и поиска данных, оптимизированный для работы с LLM через RAG. Он предоставляет Data Connectors и Query Engines для обработки больших объемов текстов, таких как PDF или API-данные. LlamaIndex востребован в задачах, требующих быстрого поиска и анализа неструктурированных данных.

**Пример**: Индексация текстового документа и выполнение запроса с LlamaIndex.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Что такое искусственный интеллект?")
print(response)
```

## 1.3 Роль RAG в современных AI-системах

Retrieval-Augmented Generation (RAG) сочетает поиск информации с генерацией текста, позволяя LLM использовать внешние данные для повышения точности ответов. LangChain и LlamaIndex упрощают создание RAG-систем, где LlamaIndex отвечает за индексацию, а LangChain — за логику обработки. RAG востребован в бизнесе для создания чат-ботов, аналитики и автоматизации.

**Пример**: Настройка RAG с LangChain и FAISS для ответа на вопросы по корпоративным документам.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# Настройка API-ключа OpenAI
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Укажите ваш ключ

# Создаем векторную базу
docs = ["Компания X основана в 2020 году.", "Продукт Y — это AI-платформа."]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever()

# Настраиваем RAG
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(input_variables=["context", "question"], template="Контекст: {context}\nВопрос: {question}\nОтвет:")
chain = {"context": retriever, "question": lambda x: x} | prompt | llm
response = chain.invoke("Когда была основана компания X?")
print(response.content)
```

## 1.4 Актуальность на рынке труда в России (2025)

Навыки работы с LangChain и LlamaIndex высоко востребованы в России, особенно в финтехе, e-commerce и маркетинге, где компании автоматизируют клиентскую поддержку и анализ данных. Вакансии для Middle+ специалистов требуют умения разрабатывать RAG-системы и интегрировать их с API. Этот учебник поможет вам освоить эти навыки и подготовиться к собеседованиям.

## 1.5 Области применения в Data Science

LangChain и LlamaIndex применяются для создания чат-ботов, анализа текстов, автоматизации отчетности и персонализированных рекомендаций. Например, в маркетинге они помогают анализировать отзывы клиентов, а в финтехе — обрабатывать финансовые документы. Эти инструменты позволяют Data Scientist’ам решать сложные задачи, минимизируя галлюцинации LLM.

## Упражнения

1. **Создание базовой цепочки в LangChain**: Настройте цепочку с использованием модели OpenAI для ответа на вопрос "Что такое машинное обучение?" с кастомным промптом. Выведите ответ в консоль.
2. **Индексация текста в LlamaIndex**: Создайте папку `data` с текстовым файлом, содержащим описание компании (2-3 предложения). Проиндексируйте его и выполните запрос "Чем занимается компания?".
3. **Простая RAG-система**: Используя LangChain и FAISS, создайте RAG-систему для ответа на вопрос "Какой продукт выпускает компания?" на основе двух документов: "Продукт A — это чат-бот" и "Продукт B — аналитическая платформа".

## Рекомендации по выполнению

- Используйте Python 3.8+ и создайте виртуальное окружение для изоляции зависимостей:
  ```bash
  python -m venv venv
  source venv/bin/activate  # На Windows: venv\Scripts\activate
  pip install -U pip
  pip install langchain==0.3.2 langchain-openai langchain-community llama-index openai faiss-cpu
  ```
- Для упражнений настройте API-ключ OpenAI через переменную окружения.
  ```python
  import os
  os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
  ```
- Для упражнения 2 создайте текстовый файл в папке `data` перед индексацией.