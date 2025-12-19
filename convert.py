import os
import torch
import torchvision
from sentence_transformers import SentenceTransformer
from spire.doc import Document, FileFormat
import platform
from pathlib import Path
from dotenv import load_dotenv
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from transformers import AutoTokenizer

from knowledge.chunk import split_by_markdown_recursive
from knowledge.question import make_question
from utils.logger import create_logger
from utils.office_to_pdf import convert_file_to_pdf
from knowledge.convert import parse_doc
from utils.utils import unzip_file

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


def split_chunk(md_list, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = []
    for md_obj in md_list:
        file_name = md_obj['file']
        chunks = []
        for text in md_obj['data']:
            chunks.extend(split_by_markdown_recursive(text['md'], tokenizer, chunk_size=512))
        data.append({'file_name': file_name, 'chunks': chunks})
    return data


def main():
    need_convert = 'knowledge/files'
    model_name = "./models/bge-small-zh-v1.5"
    md_list = []

    # 1.file 转 md
    for file in os.listdir(need_convert):
        whole_dir = os.path.join(need_convert, file)
        data = run(whole_dir)
        logger.debug(f'{file=} {data=}')
        md_list.append({'file': file, 'data': data})

    # 2. md切分chunk
    chunks = split_chunk(md_list, model_name)

    all_qa_list = []  # [{'file_name': '', 'qa_list': [{'question': '', 'answer': ''}]}]
    # 3. 提取qa
    for chunk_obj in chunks:
        file_name = chunk_obj['file_name']
        qa_list = []
        for chunk in chunk_obj['chunks']:
            if len(chunk) < 50: continue
            qustion = make_question(chunk)
            qa_list.extend(qustion)
            break
        all_qa_list.append({'file_name': file_name, 'qa_list': qa_list})
    print(all_qa_list)


if __name__ == '__main__':
    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    main()
