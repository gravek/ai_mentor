# Решения обновленных упражнений к главе 4

## Упражнение 1: Многоисточниковая интеграция данных

1. Создайте папку `data` с файлами:
   - `products.txt`:
     ```
     Product A: Чат-бот для автоматизации клиентской поддержки. Использует NLP для обработки запросов. Поддерживает интеграцию с CRM-системами. Обеспечивает круглосуточную поддержку клиентов. Улучшает удовлетворенность клиентов на 30%.
     Product B: Аналитическая платформа для больших данных. Обрабатывает терабайты данных в реальном времени. Предоставляет визуализации и прогнозы. Подходит для ритейла и финансов. Повышает точность принятия решений.
     Product C: Система рекомендаций на базе ИИ. Анализирует поведение пользователей. Формирует персонализированные предложения. Применяется в e-commerce и медиа. Увеличивает продажи на 15-20%.
     ```
   - `reviews.csv`:
     ```
     product_id,review_text,rating
     A,Чат-бот быстро отвечает, но иногда не понимает сложных запросов.,4
     A,Отличная интеграция с нашей CRM, поддержка стала эффективнее.,5
     B,Платформа мощная, но интерфейс требует доработки.,3
     B,Аналитика в реальном времени спасает наш бизнес!,5
     C,Рекомендации точные, продажи выросли.,5
     ```

2. Код:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data", required_exts=[".txt", ".csv"]).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Что говорят клиенты о продукте A?")
print(response)
```

**Объяснение**: `SimpleDirectoryReader` загружает файлы `.txt` и `.csv` из папки `data`. `VectorStoreIndex` создает индекс на основе содержимого. Запрос "Что говорят клиенты о продукте A?" извлекает отзывы из `reviews.csv` и описание из `products.txt`, возвращая, например: "Клиенты отмечают, что чат-бот Product A быстро отвечает и хорошо интегрируется с CRM, но иногда не справляется со сложными запросами. Оценки: 4 и 5."

## Упражнение 2: Гибридный поиск с кастомным ретривером

1. Создайте папку `data` с файлом `ai_solutions.txt`:
   ```
   AI оптимизирует логистику цепочек поставок. Снижает затраты на транспортировку. Улучшает маршрутизацию доставки. Повышает эффективность на 25%.
   Машинное обучение улучшает прогнозирование спроса. Анализирует исторические данные. Учитывает сезонные тренды. Снижает избыточные запасы на складах.
   ```

2. Код:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, KeywordTableIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle

documents = SimpleDirectoryReader("data").load_data()
vector_index = VectorStoreIndex.from_documents(documents)
keyword_index = KeywordTableIndex.from_documents(documents)

vector_retriever = vector_index.as_retriever(retriever_mode="embedding", similarity_top_k=2)
keyword_retriever = keyword_index.as_retriever(retriever_mode="default", similarity_top_k=2)

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, keyword_retriever):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever

    def _retrieve(self, query_bundle: QueryBundle):
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        keyword_nodes = self.keyword_retriever.retrieve(query_bundle)
        all_nodes = vector_nodes + keyword_nodes
        # Удаляем дубликаты, если нужно
        unique_nodes = {node.node_id: node for node in all_nodes}.values()
        return list(unique_nodes)[:2]

hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever)
results = hybrid_retriever.retrieve("Как AI улучшает логистику и прогнозирование?")

for node in results:
    print(f"Текст узла:\n\n{node.text}\n\nМетаданные узла:\n{node.metadata}\n\n{'-'*50}\n")
```

**Объяснение**: Гибридный ретривер объединяет результаты `VectorIndexRetriever` (семантический поиск по эмбеддингам) и `KeywordTableRetriever` (поиск по ключевым словам) с `similarity_top_k=2`. Запрос возвращает до двух уникальных документов. Ожидаемый результат: "AI оптимизирует логистику цепочек поставок... Повышает эффективность на 25%." и "Машинное обучение улучшает прогнозирование спроса... Снижает избыточные запасы на складах."

## Упражнение 3: Оптимизированная база знаний для поддержки клиентов

1. Создайте файл `knowledge_base.txt` в папке `data`:
   ```
   Компания TechAI предоставляет передовые AI-услуги. Мы разрабатываем чат-боты для автоматизации клиентской поддержки. Наши аналитические платформы обрабатывают большие данные в реальном времени. Системы рекомендаций повышают вовлеченность пользователей. Все решения интегрируются с существующими системами клиентов. Мы обеспечиваем круглосуточную техническую поддержку. Наши продукты применяются в ритейле, финансах и медиа. AI повышает эффективность бизнеса на 20-30%.
   ```

2. Код:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate

documents = SimpleDirectoryReader("data").load_data()
parser = SentenceSplitter(chunk_size=256, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

prompt_template = PromptTemplate(
    "Уважаемый клиент, на основе наших услуг: {context_str}\nВаш вопрос: {query_str}\nНаш ответ:"
)
query_engine = index.as_query_engine(response_mode="tree_summarize", prompt_template=prompt_template)
response = query_engine.query("Какие AI-услуги вы предоставляете?")
print(response)
```

**Объяснение**: `SentenceSplitter` разбивает текст на чанки (256 символов с перекрытием 50). `VectorStoreIndex` индексирует чанки для точного поиска. `PromptTemplate` задает вежливый формат ответа. Запрос возвращает, например: "Уважаемый клиент, на основе наших услуг: TechAI разрабатывает чат-боты, аналитические платформы и системы рекомендаций... Наш ответ: Мы предоставляем AI-услуги, включая чат-боты для поддержки, аналитические платформы для больших данных и системы рекомендаций для ритейла, финансов и медиа."