# Решения упражнений к главе 2

## Упражнение 1: Создание RAG с LangChain

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

docs = ["Чат-боты — это программы для общения с клиентами.", "Чат-боты используют ИИ для автоматизации."]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(input_variables=["context", "question"], template="Контекст: {context}\nВопрос: {question}\nОтвет:")
chain = {"context": retriever, "question": lambda x: x} | prompt | llm
response = chain.invoke("Что такое чат-бот?")
print(response.content)
```

**Объяснение**: Документы индексируются в FAISS с использованием эмбеддингов OpenAI. Ретривер извлекает релевантные документы, которые передаются в промпт LLM. Ответ описывает чат-боты как программы для общения и автоматизации.

## Упражнение 2: RAG с LlamaIndex

1. Создайте папку `data` и файл `products.txt` с текстом:
   ```
   Продукт A — чат-бот для автоматизации клиентской поддержки. Он отвечает на вопросы в реальном времени. Продукт B — аналитическая платформа для обработки больших данных. Она помогает выявлять тренды и прогнозировать спрос.
   ```

2. Код:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Какой функционал у продуктов?")
print(response)
```

**Объяснение**: `SimpleDirectoryReader` загружает текстовый файл, который индексируется с помощью `VectorStoreIndex`. Запрос возвращает описание функционала продуктов A и B, основанное на содержимом файла.

## Упражнение 3: Сравнение ретриверов

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

docs = ["ИИ улучшает эффективность.", "ИИ снижает затраты."]
embeddings = OpenAIEmbeddings()
prompt = PromptTemplate(input_variables=["context", "question"], template="Контекст: {context}\nВопрос: {question}\nОтвет:")
llm = ChatOpenAI(model="gpt-3.5-turbo")

# FAISS Retriever 1
vectorstore_faiss1 = FAISS.from_texts(docs, embeddings)
retriever_faiss1 = vectorstore_faiss1.as_retriever()
chain_faiss1 = {"context": retriever_faiss1, "question": lambda x: x} | prompt | llm
response_faiss1 = chain_faiss1.invoke("Как ИИ влияет на бизнес?")
print("FAISS Retriever 1:", response_faiss1.content)

# FAISS Retriever 2
vectorstore_faiss2 = FAISS.from_texts(docs, embeddings)
retriever_faiss2 = vectorstore_faiss2.as_retriever()
chain_faiss2 = {"context": retriever_faiss2, "question": lambda x: x} | prompt | llm
response_faiss2 = chain_faiss2.invoke("Как ИИ влияет на бизнес?")
print("FAISS Retriever 2:", response_faiss2.content)
```

**Объяснение**: Документы индексируются в двух экземплярах FAISS, затем для каждого создается RAG-пайплайн. Ответы сравниваются: оба ретривера должны вернуть схожие результаты, описывающие влияние ИИ на эффективность и затраты. Различия могут быть связаны с внутренней оптимизацией ретриверов.

## Упражнение 4: Гибридный поиск

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

docs = ["Чат-боты используют ИИ для автоматизации общения.", "ИИ помогает анализировать большие данные и выявлять тренды."]
embeddings = OpenAIEmbeddings()
prompt = PromptTemplate(input_variables=["context", "question"], template="Контекст: {context}\nВопрос: {question}\nОтвет:")
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Векторный поиск с FAISS
vectorstore = FAISS.from_texts(docs, embeddings)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Ключевое слово поиск с BM25
bm25_retriever = BM25Retriever.from_texts(docs)
bm25_retriever.k = 2

# Гибридный ретривер
ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5])

# RAG-пайплайн
chain = {"context": ensemble_retriever, "question": lambda x: x} | prompt | llm
response = chain.invoke("Как ИИ используется в чат-ботах и аналитике?")
print("Гибридный поиск:", response.content)
```

**Объяснение**: Документы индексируются в FAISS для векторного поиска и в BM25 для поиска по ключевым словам. `EnsembleRetriever` объединяет оба метода с равными весами (0.5), чтобы комбинировать семантическую и текстовую релевантность. Запрос возвращает ответ, описывающий использование ИИ в чат-ботах для автоматизации общения и в аналитике для обработки данных и выявления трендов.