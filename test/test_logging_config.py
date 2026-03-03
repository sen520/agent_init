#!/usr/bin/env python3
"""
测试日志配置模块
"""
import pytest
import logging
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import (
    setup_logging,
    get_logger,
    setup_colored_logging,
    ColoredFormatter
)


class TestLoggingConfig:
    """日志配置测试类"""
    
    def test_setup_logging_basic(self):
        """测试基本日志设置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            # 设置日志
            logger = setup_logging(
                level='INFO',
                log_file=str(log_file),
                console_output=False
            )
            
            # 验证日志级别
            assert logger.level == logging.INFO
            
            # 验证处理器
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], logging.FileHandler)
    
    def test_setup_logging_with_console(self):
        """测试带控制台的日志设置"""
        logger = setup_logging(
            level='DEBUG',
            log_file=None,
            console_output=True
        )
        
        # 验证有两个处理器
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    def test_get_logger(self):
        """测试获取 logger"""
        test_logger = get_logger('test_module')
        assert test_logger.name == 'test_module'
        assert isinstance(test_logger, logging.Logger)
    
    def test_logger_levels(self, caplog):
        """测试不同日志级别"""
        setup_colored_logging('DEBUG')
        logger = get_logger('test_levels')
        
        with caplog.at_level(logging.DEBUG):
            logger.debug('debug message')
            logger.info('info message')
            logger.warning('warning message')
            logger.error('error message')
        
        assert 'debug message' in caplog.text
        assert 'info message' in caplog.text
        assert 'warning message' in caplog.text
        assert 'error message' in caplog.text
    
    def test_colored_formatter(self):
        """测试彩色格式化器"""
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
        
        # 创建日志记录
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert 'test message' in formatted
    
    def test_setup_colored_logging(self):
        """测试设置彩色日志"""
        setup_colored_logging('WARNING')
        
        root = logging.getLogger()
        assert root.level == logging.WARNING


class TestLoggingIntegration:
    """日志集成测试"""
    
    def test_log_to_file(self):
        """测试写入日志文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "app.log"
            
            setup_logging(
                level='INFO',
                log_file=str(log_file),
                console_output=False
            )
            
            logger = get_logger('file_test')
            test_message = "Test log message"
            logger.info(test_message)
            
            # 验证文件内容
            assert log_file.exists()
            content = log_file.read_text()
            assert test_message in content
    
    def test_log_level_filtering(self, caplog):
        """测试日志级别过滤"""
        setup_colored_logging('WARNING')
        logger = get_logger('filter_test')
        
        with caplog.at_level(logging.WARNING):
            logger.debug('debug - should not appear')
            logger.info('info - should not appear')
            logger.warning('warning - should appear')
            logger.error('error - should appear')
        
        assert 'debug - should not appear' not in caplog.text
        assert 'info - should not appear' not in caplog.text
        assert 'warning - should appear' in caplog.text
        assert 'error - should appear' in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
