# Решения упражнений к главе 6

## Упражнение 1: Ансамблевый ретривер

```python
import csv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Загрузка отзывов из CSV
docs = []
with open('reviews.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        docs.append(row['review_text'])

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(docs, embeddings)
vector_retriever = vectorstore.as_retriever()
bm25_retriever = BM25Retriever.from_texts(docs)
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
results = ensemble_retriever.invoke("Каковы отзывы о платформе?")
print(results)

# Вариант с весами [0.3, 0.7]
ensemble_retriever_variant = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.3, 0.7])
results_variant = ensemble_retriever_variant.invoke("Каковы отзывы о платформе?")
print("Вариант:", results_variant)
```

**Объяснение**: Отзывы загружаются из CSV, индексируются в BM25 и FAISS. EnsembleRetriever комбинирует результаты, возвращая релевантные отзывы о платформе. Вариант меняет веса для большего акцента на векторный поиск.

## Упражнение 2: Настройка эмбеддингов

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Загрузка текста
with open('knowledge_base.txt', 'r', encoding='utf-8') as f:
    text = f.read()
docs = text.split('\n')  # Разделение на строки

openai_embeddings = OpenAIEmbeddings()
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore_openai = FAISS.from_texts(docs, openai_embeddings)
vectorstore_hf = FAISS.from_texts(docs, hf_embeddings)
retriever_openai = vectorstore_openai.as_retriever()
retriever_hf = vectorstore_hf.as_retriever()
results_openai = retriever_openai.invoke("Как AI повышает эффективность?")
results_hf = retriever_hf.invoke("Как AI повышает эффективность?")
print("OpenAI:", results_openai)
print("HF:", results_hf)
```

**Объяснение**: Текст из knowledge_base.txt индексируется с двумя эмбеддингами. Запрос возвращает релевантные фрагменты, такие как "AI повышает эффективность бизнеса на 20-30%". Сравните релевантность: OpenAI может быть точнее для семантики.

## Упражнение 3: Минимизация галлюцинаций

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Загрузка из PDF (предполагаем текст извлечен)
docs = ["TechTrend Innovations предлагает два основных продукта: AI-Analytics и ChatBot Pro."]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever()
context_docs = retriever.invoke("Какие продукты предлагает компания?")
context = " ".join([doc.page_content for doc in context_docs])

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(input_variables=["context", "question"], template="Отвечай только по контексту: {context}. Нет данных? Скажи 'Нет данных'. Вопрос: {question}")
chain = prompt | llm
response = chain.invoke({"context": context, "question": "Какие продукты предлагает компания?"})
print(response.content)

# Вариант с ссылками
prompt_variant = PromptTemplate(input_variables=["context", "question"], template="Отвечай по контексту: {context}. Добавь источники. Вопрос: {question}")
chain_variant = prompt_variant | llm
response_variant = chain_variant.invoke({"context": context, "question": "Какие продукты предлагает компания?"})
print("Вариант:", response_variant.content)
```

**Объяснение**: Контекст из document.pdf (упрощен) используется в промпте. Ответ: "AI-Analytics и ChatBot Pro". Вариант добавляет источники для верификации.