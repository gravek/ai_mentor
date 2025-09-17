# Глава 4: Продвинутые техники работы с LlamaIndex

В этой главе вы углубитесь в продвинутые возможности LlamaIndex, которые позволяют эффективно индексировать и обрабатывать большие объемы данных для RAG-систем. Мы рассмотрим Data Connectors, Retrievers, Query Engines и оптимизацию работы с текстами, такими как PDF, CSV и API-данные.

## 4.1 Data Connectors: загрузка данных в LlamaIndex

Data Connectors в LlamaIndex позволяют загружать данные из различных типов файлов, таких как файлы (PDF, CSV, TXT, JPG, MP3), базы данных или API. Они упрощают интеграцию неструктурированных данных, автоматически преобразуя их в формат, пригодный для индексации. Это особенно полезно для задач, где нужно обрабатывать большие объемы текстов, например, корпоративные документы или клиентские отзывы.

**Пример**: Загрузка PDF-документа с помощью `SimpleDirectoryReader`.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Предполагается, что в папке data есть файл document.pdf
documents = SimpleDirectoryReader(input_files=["data/document.pdf"]).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Что описано в документе?")
print(response)
```

## 4.2 Retrievers: настройка поиска данных

Retrievers в LlamaIndex отвечают за извлечение релевантных данных из индекса на основе запроса пользователя. LlamaIndex поддерживает различные типы ретриверов, такие как `VectorIndexRetriever` (поиск по векторным эмбеддингам) и `SummaryIndexRetriever` (поиск по обобщенным данным). Выбор ретривера зависит от задачи: векторный поиск подходит для точных запросов, а обобщающий — для анализа больших текстов.

**Пример**: Использование `VectorIndexRetriever` для поиска по индексу.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
results = retriever.retrieve("Какие продукты выпускает компания?")
for result in results:
    print(result.node.text)
```

## 4.3 Query Engines: обработка запросов

Query Engines в LlamaIndex объединяют ретривер и LLM для генерации ответов на основе извлеченных данных. Они поддерживают кастомизацию промптов через `PromptTemplate`, что позволяет задавать формат и стиль ответов, адаптируя их под конкретные задачи, например, юридическую аналитику или клиентскую поддержку. `PromptTemplate` определяет, как объединить контекст из индекса и запрос пользователя, чтобы LLM выдал точный и релевантный ответ. Это критично для управления тоном и содержанием ответа, особенно в корпоративных приложениях.

**Пример**: Настройка `QueryEngine` с кастомным промптом.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts import PromptTemplate

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
prompt = PromptTemplate("Контекст: {context_str}\nВопрос: {query_str}\nОтветь кратко:")
query_engine = index.as_query_engine(text_qa_template=prompt)
response = query_engine.query("Какие технологии использует компания?")
print(response)
```

## 4.4 Оптимизация работы с большими данными

LlamaIndex позволяет оптимизировать индексацию больших объемов данных через чанкинг (разделение текста на части) и выбор подходящих эмбеддингов. Семантическое разделение текста на чанки, например, по предложениям или абзацам, улучшает релевантность поиска, сохраняя целостность идей. Это снижает вычислительные затраты и ускоряет обработку запросов, что критично для корпоративных приложений, таких как индексация каталогов в e-commerce.

**Пример**: Индексация текстового файла с семантическим чанкингом.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

documents = SimpleDirectoryReader("data").load_data()
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100, paragraph_separator="\n\n")
nodes = node_parser.get_nodes_from_documents(documents)
for node in nodes:
    node.metadata["doc_title"] = node.source_doc.metadata.get("file_name", "Unknown")
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine()
response = query_engine.query("Что содержится в документе?")
print(response)
```

### Практические рекомендации
- **Тип документа**: Для PDF с таблицами используйте `chunk_size=256`, для текстов — `chunk_size=512` или выше.
- **Производительность**: Меньшие чанки увеличивают точность поиска, но замедляют индексацию. Большие чанки ускоряют индексацию, но могут снизить точность.
- **Тестирование**: Экспериментируйте с `chunk_size` и `chunk_overlap` на тестовых данных для оптимального баланса.

## 4.5 Практическое применение в корпоративных задачах

LlamaIndex широко используется для обработки больших объемов неструктурированных данных, таких как отчеты, контракты или клиентские запросы. В России в 2025 году навыки работы с LlamaIndex востребованы в финтехе и маркетинге, где требуется быстрый поиск и анализ текстов. Например, LlamaIndex помогает создавать системы для автоматического ответа на запросы клиентов на основе баз знаний.

## Упражнения

1. **Многоисточниковая интеграция данных**: Создайте папку `data` с файлами `products.txt` (описание трех продуктов, 4-5 предложений на каждый) и `reviews.csv` (5 отзывов с колонками: `product_id`, `review_text`, `rating`). Используйте `SimpleDirectoryReader` для загрузки и проиндексируйте с помощью `VectorStoreIndex`. Выполните запрос: "Что говорят клиенты о продукте X?".
2. **Гибридный поиск с кастомным ретривером**: Проиндексируйте два документа: "AI оптимизирует логистику цепочек поставок." и "Машинное обучение улучшает прогнозирование спроса." Настройте гибридный ретривер, комбинирующий `VectorIndexRetriever` и `KeywordTableRetriever` с `similarity_top_k=2`. Выполните запрос: "Как AI улучшает логистику и прогнозирование?" и выведите тексты найденных документов.
3. **Оптимизированная база знаний для поддержки клиентов**: Создайте файл `knowledge_base.txt` с описанием услуг компании (6-8 предложений). Используйте `SentenceSplitter` (chunk_size=256, chunk_overlap=50) для индексации. Настройте `PromptTemplate` для ответа в вежливом тоне: "Уважаемый клиент, на основе наших услуг: {context_str}\nВаш вопрос: {query_str}\nНаш ответ:". Выполните запрос: "Какие AI-услуги вы предоставляете?".

## Рекомендации по выполнению

- Понадобятся библиотеки `llama-index`, `llama-index-core`.
- Настройте API-ключ для LLM (например, OpenAI) через переменную окружения.
- Для упражнений 1 и 3 используте текстовый файл и CSV в папке `data` перед выполнением.