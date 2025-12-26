from dotenv import load_dotenv

from src.utils.logger import create_logger
from src.graph.base import build_graph

load_dotenv()
agent = build_graph()

if __name__ == '__main__':
    logger = create_logger()
    logger.debug('`123`12123`' * 20)
