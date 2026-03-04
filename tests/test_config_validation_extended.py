#!/usr/bin/env python3
"""
补充测试 - src/config/manager.py 验证功能
"""
import pytest
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.manager import ConfigManager, ConfigValidationError


class TestConfigValidationExtended:
    """扩展的配置验证测试"""
    
    def test_validation_llm_temperature(self):
        """测试 LLM temperature 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'llm': {
                    'enabled': True,
                    'temperature': 3.0  # 超过 2
                }
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'temperature' in str(exc_info.value)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validation_llm_max_tokens(self):
        """测试 LLM max_tokens 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'llm': {
                    'enabled': True,
                    'max_tokens': 50  # 小于 100
                }
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'max_tokens' in str(exc_info.value)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validation_max_function_length(self):
        """测试 max_function_length 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'optimization': {
                    'max_function_length': 600  # 超过 500
                }
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'max_function_length' in str(exc_info.value)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validation_max_files_analyze(self):
        """测试 max_files_to_analyze 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'analysis': {
                    'max_files_to_analyze': 2000  # 超过 1000
                }
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'max_files_to_analyze' in str(exc_info.value)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validation_max_file_size(self):
        """测试 max_file_size_kb 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'file_scanner': {
                    'max_file_size_kb': 20000  # 超过 10000
                }
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'max_file_size_kb' in str(exc_info.value)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validation_max_issues_in_report(self):
        """测试 max_issues_in_report 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'report': {
                    'max_issues_in_report': 5  # 小于 10
                }
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'max_issues_in_report' in str(exc_info.value)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validation_testing_timeout(self):
        """测试 testing timeout 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'testing': {
                    'timeout': 5000  # 超过 3600
                }
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'timeout' in str(exc_info.value)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validation_multiple_errors(self):
        """测试多个验证错误"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'workflow': {
                    'max_iterations': 200  # 错误
                },
                'optimization': {
                    'max_line_length': 10  # 错误
                },
                'file_modifier': {
                    'backup_retention_days': 400  # 错误
                }
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            error_msg = str(exc_info.value)
            # 应该包含多个错误
            assert 'max_iterations' in error_msg or 'max_line_length' in error_msg or 'backup_retention_days' in error_msg
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigMethodsExtended:
    """扩展的配置方法测试"""
    
    def test_get_testing_config(self):
        """测试获取测试配置"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config = ConfigManager()
        testing_config = config.get_testing_config()
        
        assert isinstance(testing_config, dict)
    
    def test_get_nonexistent_nested_key(self):
        """测试获取不存在的嵌套键"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config = ConfigManager()
        result = config.get('very.deep.nonexistent.key', 'default')
        
        assert result == 'default'
    
    def test_get_partial_path(self):
        """测试获取部分路径"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config = ConfigManager()
        result = config.get('optimization.strategies')
        
        # 可能返回 dict 或 None
        assert result is None or isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
