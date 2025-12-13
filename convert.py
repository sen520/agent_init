import os
import torch
import torchvision
from spire.doc import Document, FileFormat
import platform
from pathlib import Path
from dotenv import load_dotenv
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

from utils.logger import create_logger
from utils.office_to_pdf import convert_file_to_pdf
from knowledge.convert import parse_doc

load_dotenv()

pdf_suffixes = ['.pdf']
office_suffixes = ['.ppt', '.pptx', '.doc', '.docx', 'xls', 'xlsx']
image_suffixes = ['.png', '.jpeg', '.jpg']
compress = ['.zip', '.rar']
logger = create_logger()


def parse(pdf_files_dir, output_dir):
    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob('*'):
        if guess_suffix_by_path(doc_path) in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)
    result = parse_doc(doc_path_list, output_dir=output_dir, backend="pipeline")
    print(result)


def main():
    need_convert = 'knowledge/files'
    output_dir = 'knowledge/converted'
    tmp_dir = 'knowledge/tmp'
    for file in os.listdir(need_convert):
        whole_dir = os.path.join(need_convert, file)
        suffix = Path(whole_dir).suffix
        if suffix in pdf_suffixes + image_suffixes:
            md = parse_doc([Path(whole_dir)], output_dir=output_dir, backend="pipeline")
        elif suffix in office_suffixes:
            new_file_name = os.path.join(tmp_dir, file.replace(suffix, '.pdf'))
            if platform.system() == 'Windows':
                doc = Document()
                doc.LoadFromFile(whole_dir)
                doc.SaveToFile(new_file_name, FileFormat.PDF)
            else:
                convert_file_to_pdf(whole_dir, tmp_dir)
            md = parse_doc([Path(new_file_name)], output_dir, backend="pipeline")
        elif suffix in compress:
            pass
        else:
            logger.error(f'Suffixes Error {suffix}. Only support {pdf_suffixes + image_suffixes + office_suffixes + compress}')
        logger.debug(md)

if __name__ == '__main__':
    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    main()
