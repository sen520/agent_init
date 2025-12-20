knowledge_content = 'knowledge_content'
knowledge_chunk = 'knowledge_chunk'
knowledge_question = 'knowledge_question'

create_content_db = f'''
create table if not exists {knowledge_content} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    content TEXT NOT NULL
)
'''


create_chunk_db = f'''
create table if not exists {knowledge_chunk} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id INTEGER NOT NULL,
    chunk TEXT,
    chunk_vector Text
)
'''


create_question_db = f'''
create table if not exists {knowledge_question} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    question TEXT,
    question_vector TEXT,
    answer TEXT
)
'''