# Решения упражнений к главе 5

## Упражнение 1: Гибридный RAG с мульти-документным поиском

1. Предполагаем, что в папке `data` есть документы о компании TechAI, включая отзывы (например, файлы с текстом отзывов о услугах).

2. Код:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List

# 1. Настройка чанкинга с перекрытием
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)

# 2. Загрузка документов и создание индекса
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 3. Создание ретривера с мульти-документным поиском
retriever = index.as_retriever(similarity_top_k=5)

# 4. Функция для преобразования узлов LlamaIndex в текст
def format_docs(nodes: List) -> str:
    """Convert LlamaIndex nodes to formatted text"""
    return "\n\n".join([
        f"Отзыв {i+1}:\n{node.text}\n(Источник: {node.metadata.get('source', 'неизвестно')}"
        for i, node in enumerate(nodes)
    ])

# 5. LangChain-совместимая функция поиска
def lc_retriever(query: str) -> str:
    """Retrieve and format documents for LangChain"""
    nodes = retriever.retrieve(query)
    return format_docs(nodes)

# 6. Создание промпта с четкими инструкциями
template = """
Анализируя следующие отзывы клиентов, определи:
1. Три самых популярных услуги компании
2. Причины их популярности (с цитатами из отзывов)
3. Общую удовлетворенность клиентов

Отзывы клиентов:
{context}

Вопрос: {question}

Формат ответа:
- Услуга 1: [название]
  • Причина: [обоснование]
  • Цитата: "[прямая цитата]"
  
- Услуга 2: ...

Общая удовлетворенность клиентов:
  - [Краткий вывод о том, как клиенты оценивают компанию в целом]
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. Инициализация модели
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# 8. Создание цепочки обработки
chain = (
    {"context": RunnablePassthrough() | lc_retriever,
     "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 9. Выполнение запроса
try:
    response = chain.invoke("На основе отзывов, какие услуги компании наиболее популярны и почему?")
    print(response.content)
except Exception as e:
    print(f"Ошибка: {e}")
```

**Объяснение**: Чанкинг с перекрытием в LlamaIndex улучшает релевантность контекста. Ретривер извлекает топ-5 чанков из нескольких документов. Добавлена функция форматирования для представления отзывов с номерами и источниками. Структурированный промпт направляет LLM на анализ топ-3 услуг с причинами, цитатами и общей удовлетворенностью. Гибридная цепочка LangChain использует кастомный ретривер для совместимости, генерируя детальный ответ на основе отзывов (например, выделяя AI-разработку как популярную услугу из-за эффективности, с цитатами).

## Упражнение 2: Агент с несколькими инструментами и API-заглушкой

1. Используйте документы о компании TechAI в папке `data`.

2. Код:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_core.prompts import PromptTemplate

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()

def retrieve_company_info(query: str) -> str:
    return str(retriever.retrieve(query))

def check_okved(company: str) -> str:
    return "62.01 - Разработка ПО" if company == "TechAI" else "Не найдено"

llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = [
    Tool(name="CompanyRetriever", func=retrieve_company_info, description="Извлекает информацию о компании из документов"),
    Tool(name="OKVEDChecker", func=check_okved, description="Проверяет ОКВЭД для компании")
]

prompt = PromptTemplate(
    input_variables=["question"],
    template="Найди название компании, извлеки описание деятельности, вызови OKVEDChecker и проверь соответствие. Return a JSON object: {{'prediction': 'True' или 'False', 'reason': объясни свое решение по-русски, 'OKVED_code': return от OKVEDChecker}}"
)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=5, handle_parsing_errors=False)
chain = {"question": lambda x: x} | prompt | agent | (lambda x: x['output']) #| JsonOutputParser()

response = chain.invoke("Соответствует ли деятельность TechAI ее ОКВЭД?")
print(response)
```

**Объяснение**: Агент LangChain использует ретривер LlamaIndex как инструмент для извлечения информации о TechAI. Второй инструмент — заглушка API для ОКВЭД, которая возвращает код только для TechAI. Агент следует промпту: находит название компании, извлекает описание, проверяет ОКВЭД, сравнивает и возвращает JSON с предсказанием соответствия ('True' или 'False'), мотивацией на русском и кодом ОКВЭД.

## Упражнение 3: Оптимизация с reranking и структурированным выводом

1. Создайте документы inline или в файлах, но для примера используем в коде.

2. Код:

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.postprocessor.cohere_rerank import CohereRerank
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Документы
docs = [
    Document(text="AI ускоряет разработку ПО."),
    Document(text="Автоматизация снижает затраты на 30%.")
]
index = VectorStoreIndex.from_documents(docs)
reranker = CohereRerank(api_key="your_cohere_api_key", top_n=2)  # Требуется API ключ Cohere
retriever = index.as_retriever(node_postprocessors=[reranker])

def get_context(query: str) -> str:
    nodes = retriever.retrieve(query)
    return "\n\n".join([node.node.text for node in nodes])

context_runnable = RunnableLambda(get_context)

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Контекст: {context}\nВопрос: {question}\n
    Ответь в формате: 
    Основной ответ: [ответ]
    Источник: [документ]
    Уверенность: [высокая/средняя/низкая]"""
)
chain = {"context": context_runnable, "question": RunnablePassthrough()} | prompt | llm
response = chain.invoke("Как AI влияет на IT-бизнес?")
print(response.content)
```

**Объяснение**: Reranking с CohereRerank оптимизирует релевантность извлеченных узлов. Ретривер LlamaIndex оборачивается в RunnableLambda для совместимости с цепочкой LangChain, избегая TypeError. Кастомный промпт обеспечивает структурированный вывод с основным ответом (например, AI ускоряет разработку и снижает затраты), источником и уверенностью (высокая, если контекст напрямую релевантен).