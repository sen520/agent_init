#!/usr/bin/env python3
"""
日志配置 - 统一的日志系统
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from src.config.manager import get_config


def setup_logging(
    level: str = None,
    log_file: str = None,
    console_output: bool = True
) -> logging.Logger:
    """
    设置统一的日志配置
    
    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径
        console_output: 是否输出到控制台
        
    Returns:
        根 logger
    """
    # 从配置读取默认值
    config = get_config()
    if level is None:
        level = config.get('workflow.log_level', 'INFO')
    if log_file is None:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"optimizer_{datetime.now().strftime('%Y%m%d')}.log"
    
    # 转换日志级别
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # 创建根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    root_logger.handlers = []
    
    # 创建格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 设置第三方库日志级别
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取命名 logger
    
    Args:
        name: logger 名称（通常使用 __name__）
        
    Returns:
        logger 实例
    """
    return logging.getLogger(name)


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（用于控制台）"""
    
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'       # 重置
    }
    
    def format(self, record):
        # 保存原始级别名
        levelname = record.levelname
        
        # 添加颜色
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"
        
        result = super().format(record)
        record.levelname = levelname  # 恢复原始值
        return result


def setup_colored_logging(level: str = 'INFO'):
    """设置带颜色的控制台日志"""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # 配置根 logger
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 应用颜色格式化
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColoredFormatter(
                '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
                datefmt='%H:%M:%S'
            ))


if __name__ == "__main__":
    # 测试日志配置
    logger.info("📝 日志系统测试")
    logger.info("=" * 50)
    
    # 设置日志
    setup_colored_logging('DEBUG')
    logger = get_logger(__name__)
    
    # 测试各级别日志
    logger.debug("这是 DEBUG 消息")
    logger.info("这是 INFO 消息")
    logger.warning("这是 WARNING 消息")
    logger.error("这是 ERROR 消息")
    
    logger.info("\n✅ 日志系统测试完成")
