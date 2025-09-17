# Глава 5: Интеграция LangChain и LlamaIndex

В этой главе вы научитесь комбинировать LangChain и LlamaIndex для создания гибридных RAG-систем, где LlamaIndex отвечает за эффективную индексацию данных, а LangChain — за сложную логику обработки запросов. Мы рассмотрим ключевые аспекты интеграции, оптимизацию производительности и точности ответов, а также практическое применение в бизнес-задачах. 

## 5.1 Почему комбинировать LangChain и LlamaIndex?

LangChain и LlamaIndex дополняют друг друга: LlamaIndex обеспечивает мощную индексацию и поиск данных, а LangChain предлагает гибкость в создании цепочек и агентов. Их совместное использование позволяет строить RAG-системы, оптимизированные для сложных задач, таких как чат-боты с динамической логикой или аналитика больших текстов. Это востребовано в российских компаниях для автоматизации клиентской поддержки и обработки документов.

## 5.2 Настройка гибридного RAG-пайплайна

Гибридный RAG-пайплайн использует LlamaIndex для индексации данных и LangChain для обработки запросов и генерации ответов. LlamaIndex создает векторный индекс, который передается в LangChain как ретривер, а LangChain управляет промптами и LLM. Такой подход упрощает масштабирование и настройку под конкретные бизнес-задачи.

**Пример**: Гибридный RAG с LlamaIndex для индексации и LangChain для обработки запроса.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Индексация с LlamaIndex (используя embeddings из LangChain для совместимости)
documents = SimpleDirectoryReader("data").load_data()
embed_model = LangchainEmbedding(OpenAIEmbeddings())
llama_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Создание нативного ретривера LlamaIndex и обертка для LangChain
def langchain_retriever(query: str):
    nodes = llama_index.as_retriever(compact="tree_summarize").retrieve(query)
    return [Document(page_content=node.node.text, metadata=node.node.metadata) for node in nodes]

# Настройка LangChain
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(input_variables=["context", "question"], template="Контекст: {context}\nВопрос: {question}\nОтвет:")
chain = {"context": langchain_retriever, "question": RunnablePassthrough()} | prompt | llm
response = chain.invoke("Что содержится в документах?")
print(response.content)
```

## 5.3 Оптимизация производительности и точности

Оптимизация гибридных систем требует настройки чанков в LlamaIndex (для эффективной индексации) и промптов в LangChain (для точных ответов). Используйте чанкинг с перекрытием и кастомные промпты, чтобы минимизировать галлюцинации и ускорить поиск. Это критично для задач, где важна скорость и релевантность, например, в клиентской аналитике.

**Пример промпта для точности**:  
```
Контекст: {context}
Вопрос: {question}
Ответь кратко, используя только информацию из контекста. Если данных недостаточно, укажи это.
```

**Пример промпта для структурированного вывода**:  
```
Контекст: {context}
Вопрос: {question}
Ответь в формате: 
- Основной ответ: [ответ]
- Источник: [источник данных]
```

## 5.4 Управление сложной логикой с LangChain

LangChain позволяет добавлять сложную логику в RAG-системы, например, через агентов или цепочки с условными переходами. Это полезно для задач, где требуется динамическая обработка запросов, например, выбор между поиском в базе или вызовом API. Такие системы востребованы в финтехе и e-commerce для автоматизации сложных процессов.

**Пример**: Агент в LangChain, использующий LlamaIndex как ретривер.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_core.prompts import PromptTemplate

# Индексация с LlamaIndex
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Кастомный инструмент для LangChain
def search_index(query: str) -> str:
    return str(query_engine.query(query))

llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [Tool(name="SearchIndex", func=search_index, description="Поиск в индексе документов")]

# Кастомный ReAct-промпт для агента (для динамической логики)
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# Создание агента с использованием современного API
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "Какие услуги предоставляет компания?"})
print(response['output'])
```

## 5.5 Практическое применение в бизнесе

Гибридные RAG-системы применяются в задачах, требующих точного поиска и сложной логики, таких как автоматизация клиентской поддержки или анализ контрактов. В России в 2025 году они востребованы в маркетинге (анализ отзывов) и финтехе (обработка документов). Комбинация LangChain и LlamaIndex позволяет создавать масштабируемые решения с высокой точностью.

## Упражнения

1. **Гибридный RAG с мульти-документным поиском**: Создайте папку `data` с несколькими файлами (например, `company_profile.txt` с описанием компании, `reviews.txt` с отзывами клиентов). Настройте гибридный RAG-пайплайн, чтобы ответить на сложный вопрос: "На основе отзывов, какие услуги компании наиболее популярны и почему?". Используйте чанкинг с перекрытием в LlamaIndex для улучшения релевантности.

2. **Агент с несколькими инструментами и API-заглушкой**: Настройте агента в LangChain с двумя инструментами: один — поиск в индексе LlamaIndex (на основе документов о компании, например, "Название компании: TechCorp. Деятельность: разработка ПО."), другой — заглушка API для проверки ОКВЭД (симулируйте функцию, возвращающую коды, например, def check_okved(company: str) -> str: return "62.01 - Разработка ПО" if company == "TechCorp" else "Не найдено"). Агент должен найти название компании из документов, извлечь описание деятельности, вызвать API для ОКВЭД и проверить соответствие (например, на вопрос: "Соответствует ли деятельность TechCorp ее ОКВЭД?").

3. **Оптимизация с reranking и структурированным выводом**: Настройте RAG-пайплайн с кастомным промптом для структурированного ответа (формат: "Основной ответ: [ответ], Источник: [документ], Уверенность: [высокая/средняя/низкая]"). Добавьте reranking в LlamaIndex (используйте CohereRerank или аналогичный, если доступен API). Документы: "AI ускоряет разработку ПО." и "Автоматизация снижает затраты на 30%." для вопроса "Как AI влияет на IT-бизнес?".

## Рекомендации по выполнению

- Понадобятся библиотеки `langchain`, `llama-index`, `langchain-openai`, `langchain-community`.
- Для упражнений 1 и 2 используте файл в папке `data`.
- Для reranking в упражнении 3 добавьте библиотеки `cohere` и `llama-index-postprocessor-cohere-rerank` и настройте API-ключ, если используете внешний reranker.