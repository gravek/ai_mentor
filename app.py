import streamlit as st
import json
import glob
import os
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import hashlib  # Для хэширования пароля (опционально, здесь используем plaintext для простоты)
import cProfile

# Настройка API-ключа OpenAI
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
    api_key = st.text_input("Введите ваш API-ключ OpenAI:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key.strip()

# Функция для извлечения заголовков глав из markdown-файлов
def get_chapter_titles():
    chapters = []
    chapter_files = {}
    for file in glob.glob("knowledge_base/Учебник LangChain и LlamaIndex/chapter_*.md"):
        if "solutions" not in file:  # Исключаем файлы решений
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'^# (.+)$', content, re.MULTILINE)
                if match:
                    title = match.group(1)
                    chapters.append(title)
                    chapter_files[title] = file
    return sorted(chapters), chapter_files

# Функция для извлечения упражнений из главы
def get_exercises_from_chapter(chapter_title, chapter_files):
    chapter_file = chapter_files.get(chapter_title)
    if not chapter_file:
        return []
    
    exercises = []
    with open(chapter_file, 'r', encoding='utf-8') as f:
        content = f.read()
        sections = re.split(r'## Упражнения\n', content)
        if len(sections) < 2:
            return []
        
        exercise_section = sections[1]
        exercise_list = re.findall(r'(\d+\.\s*.*?)(?=\n\d+\.|\n##|$)', exercise_section, re.DOTALL)
        for i, ex in enumerate(exercise_list, 1):
            title = re.search(r'^\d+\.\s*(.+?)(?=\n|$)', ex, re.MULTILINE)
            description = ex.strip()
            exercises.append({
                "title": f"Упражнение {i}: {title.group(1).strip()}" if title else f"Упражнение {i}",
                "description": description
            })
    return exercises

# Функция для извлечения решений упражнений
def get_exercise_solutions(chapter_title):
    chapter_num = re.search(r'Глава (\d+)', chapter_title)
    if not chapter_num:
        return ['решения не найдены']
    chapter_file = f"knowledge_base/Учебник LangChain и LlamaIndex/chapter_{chapter_num.group(1)}_solutions.md"
    if not os.path.exists(chapter_file):
        return ['глава не найдена']
    
    exercise_solutions = []
    with open(chapter_file, 'r', encoding='utf-8') as f:
        content = f.read()
        sections = re.split(r'## Упражнение \d+:', content)
        for i, section in enumerate(sections[1:], 1):
            title_match = re.search(r'(.+?)\n', section)
            title = f"Упражнение {i}: {title_match.group(1).strip()}" if title_match else f"Упражнение {i}"
            code_match = re.search(r'```python\n(.+?)\n```', section, re.DOTALL)
            explanation_match = re.search(r'\*\*Объяснение\*\*:\s*(.+?)(?=\n##|$)', section, re.DOTALL)
            code = code_match.group(1).strip() if code_match else ""
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            exercise_solutions.append({"title": title, "code": code, "explanation": explanation})
    return exercise_solutions



# Инициализация (проверено на langchain==0.3.2)
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = DirectoryLoader('knowledge_base/', glob="**/*.md")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    return FAISS.from_documents(splits, embeddings)
vectorstore = load_vectorstore()

# Инициализация LLM
llm = ChatOpenAI(model="o4-mini")

# RAG-цепочка
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# JSON для студентов
def load_students():
    try:
        with open('students.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"students": []}

def save_students(data):
    with open('students.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

students_data = load_students()

# Функция для получения текущего студента
# @st.cache_data
def get_current_student(student_id):
    return next((s for s in students_data['students'] if s['id'] == student_id), None)

# Функция авторизации
def authenticate_user(name, password):
    student = next((s for s in students_data['students'] if s['name'] == name), None)
    if student:
        # Опционально: hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        # if student['password'] == hashed_pw:
        if student['password'] == password:  # Plaintext для простоты
            return student['id']
    return None

# Инициализация сессии
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'student_id' not in st.session_state:
    st.session_state.student_id = None
if 'last_chapter_for_lec' not in st.session_state:
    st.session_state.last_chapter_for_lec = None
if 'lec_messages' not in st.session_state:
    st.session_state.lec_messages = []
if 'last_chapter_for_ex' not in st.session_state:
    st.session_state.last_chapter_for_ex = None
if 'ex_messages' not in st.session_state:
    st.session_state.ex_messages = []

st.title("Основы LangChain и LlamaIndex")
st.badge("и ваш виртуальный ментор по Data Science", icon=":material/cognition:", color="green")

# Авторизация
if not st.session_state.authenticated:
    st.header("Авторизация")
    with st.form("auth_form"):
        name = st.text_input("Имя студента")
        password = st.text_input("Пароль", type="password")
        create_new = st.checkbox("Создать новый профиль (если не существует)")
        if create_new:
            age = st.number_input("Возраст", min_value=18, max_value=100, step=1)
            level = st.selectbox("Уровень", ["junior", "middle", "senior"])
            preferences = st.multiselect("Предпочтения", ["финтех", "e-commerce", "маркетинг", "визуализации данных", "анализ текстов"])
        submit = st.form_submit_button("Войти")
        if submit:
            if create_new and name and password and age and level and preferences:
                if any(s['name'] == name for s in students_data['students']):
                    st.error("Имя уже существует!")
                else:
                    new_id = max([s['id'] for s in students_data['students']], default=0) + 1
                    new_student = {
                        "id": new_id,
                        "name": name,
                        "password": password,
                        "age": age,
                        "level": level,
                        "preferences": preferences,
                        "current_state": {"progress": {}, "weak_areas": []},
                        "dialog_history": {}
                    }
                    students_data['students'].append(new_student)
                    save_students(students_data)
                    st.session_state.student_id = new_id
                    st.session_state.authenticated = True
                    st.success(f"Профиль {name} создан!")
                    st.rerun()
            else:
                student_id = authenticate_user(name, password)
                if student_id:
                    st.session_state.student_id = student_id
                    st.session_state.authenticated = True
                    st.success(f"Добро пожаловать, {name}!")
                    st.rerun()
                else:
                    st.error("Неверные данные!")
else:

    student = get_current_student(st.session_state.student_id)
    if not student:
        st.error("Ошибка: студент не найден!")
        st.stop()

    # Shared selectbox в sidebar
    with st.sidebar:
        st.header("Профиль")

        # Кнопка выхода
        if st.button("Выйти"):
            st.session_state.authenticated = False
            st.session_state.student_id = None
            st.session_state.last_chapter_for_lec = None
            st.session_state.last_chapter_for_ex = None
            st.rerun()
        st.header("Навигация по главам")
        chapters, chapter_files = get_chapter_titles()
        if chapters:
            default_index = 0 if st.session_state.last_chapter_for_lec is None else chapters.index(st.session_state.last_chapter_for_lec) if st.session_state.last_chapter_for_lec in chapters else 0
            selected_chapter = st.selectbox("Выберите главу", chapters, index=default_index, key="shared_chapter_select")
            if st.session_state.last_chapter_for_lec != selected_chapter:
                st.session_state.last_chapter_for_lec = selected_chapter
                st.session_state.last_chapter_for_ex = selected_chapter
                st.rerun()
            if student and selected_chapter:
                chapter_key = selected_chapter.lower()
                if 'dialog_history' in student and chapter_key in student['dialog_history']:
                    st.session_state.lec_messages = student['dialog_history'][chapter_key].get('lecture_chat', [])
                    st.session_state.ex_messages = student['dialog_history'][chapter_key].get('exercise_chat', [])
                else:
                    st.session_state.lec_messages = []
                    st.session_state.ex_messages = []
        else:
            selected_chapter = None

    tab1, tab2, tab3 = st.tabs(["Профиль студента", "Учебный контент", "Решение упражнений"])

    # Вкладка: Профиль студента
    with tab1:
        st.header("Управление профилем студента")
        st.write(f"Текущий профиль: {student['name']}")
        st.write(f"Возраст: {student['age']}")
        st.write(f"Уровень: {student['level']}")
        st.write(f"Предпочтения: {', '.join(student['preferences'])}")
        st.write(f"Прогресс: {student['current_state']['progress']}")
        st.write(f"Слабые области: {student['current_state']['weak_areas']}")

        # Редактирование профиля
        with st.form("edit_profile"):
            new_age = st.number_input("Возраст", value=student['age'])
            new_level = st.selectbox("Уровень", ["junior", "middle", "senior"], index=["junior", "middle", "senior"].index(student['level']))
            new_prefs = st.multiselect("Предпочтения", ["финтех", "e-commerce", "маркетинг", "визуализации данных", "анализ текстов"], default=student['preferences'])
            submit_edit = st.form_submit_button("Сохранить изменения")
            if submit_edit:
                student['age'] = new_age
                student['level'] = new_level
                student['preferences'] = new_prefs
                save_students(students_data)
                st.success("Профиль обновлен!")
                st.rerun()

        # Выбор другого профиля
        other_names = [s['name'] for s in students_data['students'] if s['id'] != student['id']]
        selected_other = st.selectbox("Переключить профиль", [""] + other_names)
        if selected_other:
            other_pw = st.text_input(f"Пароль для {selected_other}:", type="password")
            if st.button("Переключить"):
                new_id = authenticate_user(selected_other, other_pw)
                if new_id:
                    st.session_state.student_id = new_id
                    st.rerun()
                else:
                    st.error("Неверный пароль!")

        st.download_button(label='Скачать сохраненную историю', data=json.dumps(student, ensure_ascii=False, indent=2))

    # Вкладка: Учебный контент
    with tab2:
        st.header("Учебный контент")
        if not chapters or not selected_chapter:
            st.warning("Выберите главу в боковой панели.")
            st.stop()

        chapter_key = selected_chapter.lower()
        if st.session_state.last_chapter_for_lec != selected_chapter:
            student = get_current_student(st.session_state.student_id)
            if student and 'dialog_history' in student and chapter_key in student['dialog_history']:
                st.session_state.lec_messages = student['dialog_history'][chapter_key].get('lecture_chat', [])
            else:
                st.session_state.lec_messages = []
            st.session_state.last_chapter_for_lec = selected_chapter

        st.subheader("Лекция по главе из учебника")
        
        lecture_exists = 'lecture' in student['dialog_history'][chapter_key]
        if not lecture_exists:
            if st.button("Сгенерировать персонализированную лекцию"):
                with st.spinner("Ментор готовит лекцию..."):
                    prompt = f"Краткая лекция по теме '{selected_chapter}' для уровня {student['level']}, опираясь на учебник. Учти слабые области: {student['current_state']['weak_areas']}. Предпочтения: {student['preferences']}"
                    lec = qa_chain.run(prompt)
                    student['dialog_history'][chapter_key]['lecture'] = lec
                    # Обновление прогресса
                    student['current_state']['progress'][chapter_key] = "started"
                    st.write(lec)
                summary_prompt = f"Суммаризируй лекцию: {lec}"
                summary = llm.invoke(summary_prompt).content
                student['dialog_history'][chapter_key]['lecture_summary'] = summary
                save_students(students_data)
                st.rerun()
        else:
            with st.container(height=500, key="lec_cont"):
                st.write(student['dialog_history'][chapter_key]['lecture'])

        st.subheader("Учебный материал и упражнения")
        if st.button("Показать главу из учебника"):
            if selected_chapter in chapter_files:
                with st.container(height=500, key="chap_cont"):
                    with open(chapter_files[selected_chapter], 'r', encoding='utf-8') as f:
                        content = f.read()
                        st.markdown(content)
            else:
                st.error("Файл главы не найден!")

        st.subheader("Чат по учебному материалу выбранной главы")
        # Всегда перезагружать messages при смене главы
        if st.session_state.last_chapter_for_lec != selected_chapter:
            st.session_state.lec_messages = student['dialog_history'].get(chapter_key, {}).get('lecture_chat', [])
            st.session_state.last_chapter_for_lec = selected_chapter
            # Debug (удалите в продакшене)
            st.info(f"Загружено {len(st.session_state.lec_messages)} сообщений для лекций главы '{chapter_key}'")

        # # включить для сессии без подгрузки сообщений
        # if 'lec_messages' not in st.session_state:
        #     st.session_state.lec_messages = []

        with st.container(height=500, key="lec_chat"):
                st.chat_message("assistant").write("Здравствуйте! Я ваш виртуальный ментор. Задавайте вопросы!")
                if st.session_state.lec_messages:
                    for msg in st.session_state.lec_messages[-10:]:
                        st.chat_message("user").write(msg["user"])
                        st.chat_message("assistant").write(msg["bot"])

        st.write(f"Loaded {len(st.session_state.lec_messages[-10:])} messages for chapter {chapter_key}")

        user_input = st.chat_input("Ваш вопрос виртуальному ментору:")
        if user_input:
            with st.spinner("Ментор думает..."):
                prompt = f"Ответь как ментор по правилам, опираясь на выбранную главу и весь учебник: {user_input}. Учти уровень {student['level']}, предпочтения {student['preferences']}, историю: {student['dialog_history'].get(chapter_key, {}).get('lecture_chat', [])}"
                response = qa_chain.run(prompt)
            st.session_state.lec_messages.append({"user": user_input, "bot": response})
            # Сохранение
            student['dialog_history'][chapter_key].setdefault('lecture_chat', []).append({"user": user_input, "bot": response})
            save_students(students_data)
            st.rerun()

        st.subheader("История взаимодействия для этой главы")
                
        if st.button("Просмотр истории главы", key="view_lec_hist"):
            if student:
                st.write(student['dialog_history'].get(chapter_key, "История для этой главы пуста"))
            else:
                st.error("Выберите профиль студента!")

        if st.button("Очистить историю главы"):
            if student:
                student['dialog_history'][chapter_key] = {}
                save_students(students_data)
                st.session_state.lec_messages = []
                st.success("История главы очищена!")
                st.rerun()

    # Вкладка: Решение упражнений
    with tab3:
        st.header("Решение упражнений")
        if not chapters or not selected_chapter:
            st.warning("Выберите главу в боковой панели.")
            st.stop()

        chapter_key = selected_chapter.lower()
        if st.session_state.last_chapter_for_ex != selected_chapter:
            student = get_current_student(st.session_state.student_id)
            if student and 'dialog_history' in student and chapter_key in student['dialog_history']:
                st.session_state.ex_messages = student['dialog_history'][chapter_key].get('exercise_chat', [])
            else:
                st.session_state.ex_messages = []
            st.session_state.last_chapter_for_ex = selected_chapter

        exercises = get_exercises_from_chapter(selected_chapter, chapter_files)
        exercise_solutions = get_exercise_solutions(selected_chapter)
        if not exercise_solutions:
            st.warning("Упражнения для этой главы не найдены!")
        else:
            selected_ex = st.selectbox("Выберите упражнение", [ex['title'] for ex in exercise_solutions], key="ex_select")
            if st.button("Показать описание решения"):
                selected_ex_data = next(ex for ex in exercise_solutions if ex['title'] == selected_ex)
                st.write(selected_ex_data['explanation'])   


            with st.container(height=500, key="ex_chat"):
                st.chat_message("assistant").write("Привет! Я помогу с упражнениями. Задавайте вопросы или показывайте решения!")
                for msg in st.session_state.ex_messages[-10:]:
                    st.chat_message("user").write(msg["user"])
                    st.chat_message("assistant").write(msg["bot"])
            
            user_input_ex = st.chat_input("Решение или вопрос:")
            if user_input_ex:
                with st.spinner("Ментор думает..."):
                    selected_ex_data = next(ex for ex in exercise_solutions if ex['title'] == selected_ex)
                    prompt = f"""
                    Отвечай компактно.
                    Если студент задает вопрос по упражнению '{selected_ex}', не предоставив код, ответь, не показывая полный код решения, а только самую необходимую часть как пример.
                    Если студент показывает свое решение, проверь его по:
                    -упражнению {selected_ex} 
                    -соответствию ожидаемму решению {selected_ex_data['code']} из учебника.
                    - дай конструктивный фидбек, указав на ошибки и предложив улучшения, опираясь на выбранную главу, весь учебник и правила для уровня {student['level']}.
                    Ответ студента: {user_input_ex}
                    Объяснение решения (если понадобится его разъясненить): {selected_ex_data['explanation']}
                    """
                    response = qa_chain.run(prompt)
                    # Обновление weak_areas
                    if any(word in response.lower() for word in ['ошибка', 'исправь', 'неверно']):
                        if 'weak_areas' not in student['current_state']:
                            student['current_state']['weak_areas'] = []
                        if selected_chapter not in student['current_state']['weak_areas']:
                            student['current_state']['weak_areas'].append(selected_chapter)
                    student['current_state']['progress'][chapter_key] = "exercises_started"
                st.session_state.ex_messages.append({"user": user_input_ex, "bot": response})
                # Сохранение
                student['dialog_history'][chapter_key].setdefault('exercise_chat', []).append({"user": user_input_ex, "bot": response})
                save_students(students_data)
                st.rerun()

            if st.button("Очистить историю упражнений главы"):
                student['dialog_history'][chapter_key].pop('exercise_chat', None)
                st.session_state.ex_messages = []
                save_students(students_data)
                st.success("История упражнений очищена!")
                st.rerun()