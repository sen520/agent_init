"""
配置管理模块 - 统一管理项目配置
"""
import os
import logging
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self._config = {}
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None):
        """加载配置文件"""
        # 默认配置
        self._config = {
            'optimization': {
                'max_line_length': 100,
                'max_function_length': 50,
                'security_checks': True,
                'backup_enabled': True
            },
            'strategy_groups': {
                'default': [
                    'comment_optimizer',
                    'empty_line_optimizer', 
                    'import_optimizer'
                ],
                'safe': [
                    'comment_optimizer',
                    'empty_line_optimizer'
                ],
                'aggressive': [
                    'line_length_optimizer',
                    'variable_naming_optimizer',
                    'function_length_optimizer'
                ]
            }
        }
        
        # 如果指定了配置文件，尝试加载
        if config_path:
            if os.path.exists(config_path):
                self._load_from_file(config_path)
            else:
                logger.info(f"⚠️ 配置文件不存在: {config_path}")
        else:
            # 尝试从默认位置加载
            default_paths = [
                'config/default.yaml',
                'config.yaml',
                '.soa-config.yaml'
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    self._load_from_file(path)
                    break
    
    def _load_from_file(self, file_path: str):
        """从YAML文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            
            # 深度合并配置
            self._deep_merge(self._config, file_config)
            logger.info(f"✅ 已加载配置文件: {file_path}")
            
        except Exception as e:
            logger.info(f"⚠️ 加载配置文件失败 {file_path}: {e}")
    
    def _deep_merge(self, base: dict, update: dict):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """获取优化相关配置"""
        return self._config.get('optimization', {})
    
    def get_strategy_group(self, group_name: str) -> List[str]:
        """获取策略组"""
        return self.get(f'strategy_groups.{group_name}', [])
    
    def get_max_line_length(self) -> int:
        """获取最大行长度配置"""
        return self.get('optimization.max_line_length', 100)
    
    def get_max_function_length(self) -> int:
        """获取最大函数长度配置"""
        return self.get('optimization.max_function_length', 50)
    
    def is_backup_enabled(self) -> bool:
        """是否启用备份"""
        return self.get('optimization.backup_enabled', True)
    
    def is_security_checks_enabled(self) -> bool:
        """是否启用安全检查"""
        return self.get('optimization.security_checks', True)
    
    @property
    def raw_config(self) -> Dict[str, Any]:
        """获取原始配置字典"""
        return self._config.copy()


# 全局配置实例
_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def load_config(config_path: str):
    """加载指定配置文件"""
    global _global_config
    _global_config = Config(config_path)


# 使用示例
if __name__ == "__main__":
    config = get_config()
    logger.info("🔧 配置信息:")
    logger.info(f"   最大行长度: {config.get_max_line_length()}")
    logger.info(f"   最大函数长度: {config.get_max_function_length()}")
    logger.info(f"   启用备份: {config.is_backup_enabled()}")
    logger.info(f"   默认策略组: {config.get_strategy_group('default')}")