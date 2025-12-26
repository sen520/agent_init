import zipfile
import os
from loguru import logger


def unzip_file(zip_path, extract_dir):
    """
    解压 ZIP 文件
    :param zip_path: ZIP 文件的路径
    :param extract_dir: 解压后的文件保存目录
    """
    # 确保解压目录存在
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # 打开 ZIP 文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 解压所有文件
        zip_ref.extractall(extract_dir)
        logger.warning(f"ZIP 文件已解压到: {extract_dir}")

if __name__ == '__main__':

    # 调用示例
    unzip_file("../knowledge/files/test.zip", "../knowledge/files")