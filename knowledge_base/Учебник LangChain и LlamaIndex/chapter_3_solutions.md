# Решения упражнений к главе 3

## Упражнение 1: Анализ финансовых отчетов с RAG

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

docs = ["Выручка в Q1 2025 выросла на 15%.", "Затраты на маркетинг сократились на 10%."]
embeddings = OpenAIEmbeddings()
vectorstore = Qdrant.from_texts(docs, embeddings, location=":memory:")
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Контекст: {context}\nВопрос: {question}\nОтвет в формате маркированного списка"
)
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
response = chain.invoke("Каковы ключевые финансовые изменения в Q1 2025?")
print(response.content)
```

**Объяснение**: Документы индексируются в Qdrant с использованием эмбеддингов OpenAI. Ретривер извлекает релевантный контекст, который передается в кастомный промпт, требующий ответ в формате маркированного списка. LLM генерирует структурированный ответ, например:
- Выручка в Q1 2025 выросла на 15%.
- Затраты на маркетинг сократились на 10%.

## Упражнение 2: Агент для управления данными сотрудников

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
import requests
from datetime import datetime, date

def get_employee_info(employee_id: str) -> str:
    response = requests.get(f"https://dummyjson.com/users/{employee_id}")
    return str(response.json())

def calculate_age_in_months(birth_date: str) -> str:
    birth = datetime.strptime(birth_date, "%Y-%m-%d").date()
    today = date(2025, 8, 7)  # Текущая дата
    months = (today.year - birth.year) * 12 + today.month - birth.month
    return str(months)

llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [
    Tool(name="GetEmployeeInfo", func=get_employee_info, description="Получает данные сотрудника по ID через API"),
    Tool(name="CalculateAgeInMonths", func=calculate_age_in_months, description="Вычисляет возраст в месяцах по дате рождения (YYYY-MM-DD)")
]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
response = agent.run("Получи информацию о сотруднике с ID 7 и рассчитай его возраст в месяцах на текущую дату.")
print(response)
```

**Объяснение**: Агент использует два инструмента: `get_employee_info` запрашивает данные сотрудника через API, а `calculate_age_in_months` вычисляет возраст в месяцах на основе поля `birthDate`. Текущая дата фиксируется как 7 августа 2025 года. Агент возвращает информацию о сотруднике и его возраст, например: "Сотрудник: {...}, возраст в месяцах: 420".

## Упражнение 3: Чат-бот с памятью и динамическими инструментами

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
import random
from datetime import datetime

def get_current_date(_):
    return datetime(2025, 8, 7).strftime("%Y-%m-%d")

def get_random_number(_):
    return str(random.randint(1, 100))

llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory()

tools = [
    Tool(
        name="GetCurrentDate",
        func=get_current_date,
        description="Возвращает текущую дату в формате YYYY-MM-DD. Входные данные игнорируются."
    ),
    Tool(
        name="GetRandomNumber",
        func=get_random_number,
        description="Возвращает случайное число от 1 до 100. Входные данные игнорируются."
    ),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

print(agent.run("Какая сегодня дата?"))
print(agent.run("Назови случайное число"))
print(agent.run("Что я спрашивал в самом начале?"))
```

**Объяснение**: Агент, созданный с помощью `initialize_agent` и `ConversationBufferMemory`, сохраняет историю диалога и использует два кастомных инструмента: `get_current_date` (возвращает 2025-08-07) и `get_random_number` (случайное число от 1 до 100). Параметры `verbose=True`, `handle_parsing_errors=True` и `max_iterations=5` обеспечивают отладку и ограничение итераций. Последовательные запросы возвращают:
1. "2025-08-07"
2. Случайное число, например, "53"
3. "Вы спрашивали: Какая сегодня дата?"