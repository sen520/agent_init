#!/usr/bin/env python3
"""
测试 ConfigManager 配置管理器
"""
import pytest
import json
import tempfile
import os
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.manager import ConfigManager, ConfigValidationError, get_config


class TestConfigManager:
    """ConfigManager 测试类"""
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        # 重置单例
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2
    
    def test_default_config(self):
        """测试默认配置"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config = ConfigManager("/nonexistent/path.json")
        
        assert config.get('workflow.max_iterations') == 5
        assert config.get('optimization.max_line_length') == 100
        assert config.get('optimization.max_function_length') == 50
    
    def test_get_nested_value(self):
        """测试获取嵌套配置值"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config = ConfigManager()
        
        # 测试存在的键
        assert config.get('optimization.max_line_length') == 100
        assert config.get('file_modifier.backup_dir') == '.optimization_backups'
        
        # 测试不存在的键（返回默认值）
        assert config.get('nonexistent.key', 'default') == 'default'
        assert config.get('nonexistent') is None
    
    def test_validation_max_iterations(self):
        """测试 max_iterations 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'workflow': {'max_iterations': 200}  # 超过 100
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'max_iterations' in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_validation_max_line_length(self):
        """测试 max_line_length 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'optimization': {'max_line_length': 10}  # 小于 50
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'max_line_length' in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_validation_backup_retention(self):
        """测试 backup_retention_days 验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'file_modifier': {'backup_retention_days': 400}  # 超过 365
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigManager(temp_path)
            assert 'backup_retention_days' in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_get_workflow_config(self):
        """测试获取工作流配置"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config = ConfigManager()
        workflow = config.get_workflow_config()
        
        assert isinstance(workflow, dict)
        assert 'max_iterations' in workflow
    
    def test_get_optimization_config(self):
        """测试获取优化配置"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config = ConfigManager()
        opt_config = config.get_optimization_config()
        
        assert isinstance(opt_config, dict)
        assert 'max_line_length' in opt_config
        assert 'strategies' in opt_config
    
    def test_valid_config_passes(self):
        """测试有效配置通过验证"""
        ConfigManager._instance = None
        ConfigManager._config = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'version': '1.0.0',
                'workflow': {'max_iterations': 10},
                'optimization': {
                    'max_line_length': 120,
                    'max_function_length': 60
                },
                'file_modifier': {
                    'backup_retention_days': 30
                }
            }, f)
            temp_path = f.name
        
        try:
            config = ConfigManager(temp_path)
            assert config.get('workflow.max_iterations') == 10
            assert config.get('optimization.max_line_length') == 120
        finally:
            os.unlink(temp_path)


class TestConfigHelperFunctions:
    """测试配置助手函数"""
    
    def test_helper_functions(self):
        """测试便捷函数"""
        from src.config.manager import (
            get_max_iterations,
            get_max_line_length,
            get_max_function_length,
            get_backup_dir,
            get_max_files_to_analyze,
            get_enabled_strategies
        )
        
        ConfigManager._instance = None
        ConfigManager._config = None
        
        # 这些函数不应该抛出异常
        assert isinstance(get_max_iterations(), int)
        assert isinstance(get_max_line_length(), int)
        assert isinstance(get_max_function_length(), int)
        assert isinstance(get_backup_dir(), str)
        assert isinstance(get_max_files_to_analyze(), int)
        assert isinstance(get_enabled_strategies(), list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
