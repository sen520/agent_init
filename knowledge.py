import os
import json
import torch
import torchvision
from qdrant_client.grpc import PointId, Vector, Vectors
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from spire.doc import Document, FileFormat
import platform
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from transformers import AutoTokenizer

from utils.emb import get_qdrant_client
from knowledge.chunk import split_by_markdown_recursive
from knowledge.question import make_question
from utils.logger import create_logger
from utils.office_to_pdf import convert_file_to_pdf
from knowledge.convert import parse_doc
from knowledge.static import *
from utils.utils import unzip_file
from utils.sqlite_db import SQLiteDB
from utils.customModel import CustomEmbedding

load_dotenv()

pdf_suffixes = ['.pdf']
office_suffixes = ['.ppt', '.pptx', '.doc', '.docx', 'xls', 'xlsx']
image_suffixes = ['.png', '.jpeg', '.jpg']
compress = ['.zip']
logger = create_logger()


def parse(pdf_files_dir, output_dir):
    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob('*'):
        if guess_suffix_by_path(doc_path) in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)
    result = parse_doc(doc_path_list, output_dir=output_dir, backend="pipeline")
    print(result)


def run(whole_dir):
    output_dir = 'knowledge/converted'
    tmp_dir = 'knowledge/tmp'
    file = os.path.basename(whole_dir)
    suffix = Path(whole_dir).suffix
    logger.debug(f'{whole_dir}')
    if suffix in pdf_suffixes + image_suffixes:
        md = parse_doc([Path(whole_dir)], output_dir=output_dir, backend="pipeline")
    elif suffix in office_suffixes:
        new_file_name = os.path.join(tmp_dir, file.replace(suffix, '.pdf'))
        # 转化的两种方式
        # 1. spire 需要用会员
        # doc = Document()
        # doc.LoadFromFile(whole_dir)
        # doc.SaveToFile(new_file_name, FileFormat.PDF)

        # 2.libreoffice需要安装libreoffice
        convert_file_to_pdf(whole_dir, tmp_dir)
        md = parse_doc([Path(new_file_name)], output_dir, backend="pipeline")
    elif suffix in compress:
        unzip_dir = os.path.join(tmp_dir, file.replace(suffix, ''))
        unzip_file(whole_dir, unzip_dir)
        md = []
        for zip_file in os.listdir(unzip_dir):
            whole_zip = os.path.join(unzip_dir, zip_file)
            single_zip = run(whole_zip)
            md.extend(single_zip)

    else:
        md = []
        logger.error(
            f'Suffixes Error {suffix}. Only support {pdf_suffixes + image_suffixes + office_suffixes + compress}')
    return md


def split_chunk(model_name, sql):
    md_list = sql.execute_query('select * from knowledge_content')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunks = []
    for md_obj in md_list:
        chunks = split_by_markdown_recursive(md_obj['content'], tokenizer, chunk_size=512)
        for chunk in chunks:
            if len(chunk) < 50:
                continue
            sql.execute_update(f'insert into {knowledge_chunk}  (content_id, chunk) values (?, ?)',
                               (md_obj['id'], chunk))
    return chunks


def init_db(sql):
    tables = sql.execute_query('''SELECT name 
FROM sqlite_master 
WHERE type = 'table' 
  AND name NOT LIKE 'sqlite_%';''')
    table_names = [i['name'] for i in tables]
    if 'knowledge_content' not in table_names:
        sql.execute_update(create_content_db)
    if 'knowledge_chunk' not in table_names:
        sql.execute_update(create_chunk_db)
    if 'knowledge_question' not in table_names:
        sql.execute_update(create_question_db)


def make_qa(sql):
    chunk_objs = sql.execute_query(f'select * from {knowledge_chunk}')
    for chunk_obj in chunk_objs:
        qustion = make_question(chunk_obj['chunk'])
        for q in qustion:
            sql.execute_update(f'insert into {knowledge_question}  (chunk_id, question, answer) values (?, ?, ?)',
                               (chunk_obj['id'], q['question'], q['answer']))


def embedding(sql):
    chunk_objs = sql.execute_query(f'select * from {knowledge_chunk}')
    embedding_model = CustomEmbedding(model_name='Qwen3-Embedding-8B', use_local=False, api_url='http://127.0.0.1:5000')
    logger.debug('emb chunk')
    for chunk_obj in tqdm(chunk_objs, total=len(chunk_objs)):
        emb = embedding_model.embed_query(chunk_obj['chunk'])
        sql.execute_update(f'update {knowledge_chunk} set chunk_vector = ? where id = ?',
                           (json.dumps(emb), chunk_obj['id']))

    logger.debug('emb question')
    question_objs = sql.execute_query(f'select * from {knowledge_question}')
    for question_obj in tqdm(question_objs, total=len(question_objs)):
        emb = embedding_model.embed_query(question_obj['question'])
        sql.execute_update(f'update {knowledge_question} set question_vector = ? where id = ?',
                           (json.dumps(emb), question_obj['id']))


def main():
    need_convert = 'knowledge/files'
    model_name = "./models/bge-small-zh-v1.5"

    db = 'main.db'
    sql = SQLiteDB(db)
    init_db(sql)

    # 1.file 转 md
    for file in os.listdir(need_convert):
        whole_dir = os.path.join(need_convert, file)
        data = run(whole_dir)
        content = '\n'.join(data).strip()
        logger.debug(f'{file=} {data=}')
        if content:
            sql.execute_update(f'insert into {knowledge_content} (filename, content) values (?, ?)', (file, '\n'.join(data)))

    # 2. md切分chunk
    split_chunk(model_name, sql)

    # 3. 提取qa
    make_qa(sql)

    # 4. embedding
    embedding(sql)

    # 5. 知识库
    lines = sql.execute_query(f'select * from {knowledge_chunk} limit 1')
    client = get_qdrant_client()

    vector = json.loads(lines[0]['chunk_vector'])
    point = PointStruct(
        id=lines[0]['id'],
        vector=vector,
        payload={"chunk": lines[0]['chunk']}
    )

    client.upsert(collection_name='test', wait=True, points=[point])


if __name__ == '__main__':
    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    main()
