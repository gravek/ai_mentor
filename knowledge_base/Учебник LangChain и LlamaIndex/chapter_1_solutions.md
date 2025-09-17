# Решения упражнений к главе 1

## Упражнение 1: Создание базовой цепочки в LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(input_variables=["question"], template="Ответь кратко и понятно: {question}")
chain = prompt | llm
response = chain.invoke({"question": "Что такое машинное обучение?"})
print(response.content)
```

**Объяснение**: Создается промпт с переменной `question`, который передается в модель OpenAI. Цепочка объединяет промпт и модель, а метод `invoke` выполняет запрос. Ответ будет содержать краткое описание машинного обучения.

## Упражнение 2: Индексация текста в LlamaIndex

1. Создайте папку `data` и файл `company.txt` с текстом:  
   ```
   Компания TechAI разрабатывает решения на базе искусственного интеллекта. Основной продукт — чат-боты для клиентской поддержки. Компания основана в 2023 году.
   ```

2. Код:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Чем занимается компания?")
print(response)
```

**Объяснение**: `SimpleDirectoryReader` загружает текстовый файл из папки `data`. `VectorStoreIndex` создает векторный индекс, а `query_engine` выполняет запрос, возвращая информацию о деятельности компании.

## Упражнение 3: Простая RAG-система

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

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

**Объяснение**: Документы индексируются в FAISS с использованием эмбеддингов OpenAI. Retriever извлекает релевантные документы, которые передаются в промпт вместе с вопросом. Модель возвращает ответ, основанный на контексте, например: "Компания X была основана в 2020 году".