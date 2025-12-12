import os

from dotenv import load_dotenv

from utils.logger import create_logger

load_dotenv()

if __name__ == '__main__':
    logger = create_logger()
    logger.debug('`123`12123`' * 20)
