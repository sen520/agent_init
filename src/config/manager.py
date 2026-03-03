#!/usr/bin/env python3
"""
配置管理器 - 集中管理所有配置项
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List


class ConfigManager:
    """配置管理器 - 从 config.json 读取配置"""
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: str = None):
        """加载配置文件"""
        if config_path is None:
            # 默认从项目根目录查找
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.json"
        
        self.config_path = Path(config_path)
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️  配置文件解析错误: {e}，使用默认配置")
                self._config = self._default_config()
        else:
            print(f"⚠️  配置文件不存在: {self.config_path}，使用默认配置")
            self._config = self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            "workflow": {"max_iterations": 5},
            "analysis": {"max_files_to_analyze": 20, "max_files_to_optimize": 15},
            "optimization": {
                "max_line_length": 100,
                "max_function_length": 50,
                "enable_auto_fix": True,
                "strategies": {
                    "enabled": ["line_length_optimizer", "import_optimizer", "comment_optimizer"]
                }
            },
            "file_modifier": {
                "backup_dir": ".optimization_backups",
                "backup_retention_days": 7
            },
            "report": {"output_dir": "reports", "max_issues_in_report": 50},
            "file_scanner": {"max_file_size_kb": 100}
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项（支持点号分隔的嵌套键）
        
        Args:
            key: 配置键，如 "optimization.max_line_length"
            default: 默认值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_workflow_config(self) -> Dict:
        """获取工作流配置"""
        return self._config.get('workflow', {})
    
    def get_analysis_config(self) -> Dict:
        """获取分析配置"""
        return self._config.get('analysis', {})
    
    def get_optimization_config(self) -> Dict:
        """获取优化配置"""
        return self._config.get('optimization', {})
    
    def get_file_scanner_config(self) -> Dict:
        """获取文件扫描器配置"""
        return self._config.get('file_scanner', {})
    
    def get_file_modifier_config(self) -> Dict:
        """获取文件修改器配置"""
        return self._config.get('file_modifier', {})
    
    def get_report_config(self) -> Dict:
        """获取报告配置"""
        return self._config.get('report', {})
    
    def get_code_analyzer_config(self) -> Dict:
        """获取代码分析器配置"""
        return self._config.get('code_analyzer', {})
    
    def get_llm_config(self) -> Dict:
        """获取 LLM 配置"""
        return self._config.get('llm', {})
    
    def get_self_optimization_config(self) -> Dict:
        """获取自优化配置"""
        return self._config.get('self_optimization', {})
    
    def reload(self):
        """重新加载配置"""
        self._load_config(self.config_path)
    
    def to_dict(self) -> Dict:
        """返回完整配置字典"""
        return self._config.copy()


# 全局配置实例
_config_instance = None

def get_config(config_path: str = None) -> ConfigManager:
    """获取配置管理器实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance


# 便捷函数
def get_max_iterations() -> int:
    """获取最大迭代次数"""
    return get_config().get('workflow.max_iterations', 5)

def get_max_line_length() -> int:
    """获取最大行长度"""
    return get_config().get('optimization.max_line_length', 100)

def get_max_function_length() -> int:
    """获取最大函数长度"""
    return get_config().get('optimization.max_function_length', 50)

def get_backup_dir() -> str:
    """获取备份目录"""
    return get_config().get('file_modifier.backup_dir', '.optimization_backups')

def get_max_files_to_analyze() -> int:
    """获取最大分析文件数"""
    return get_config().get('analysis.max_files_to_analyze', 20)

def get_max_files_to_optimize() -> int:
    """获取最大优化文件数"""
    return get_config().get('analysis.max_files_to_optimize', 15)

def get_enabled_strategies() -> List[str]:
    """获取启用的优化策略"""
    return get_config().get('optimization.strategies.enabled', [])

def get_report_output_dir() -> str:
    """获取报告输出目录"""
    return get_config().get('report.output_dir', 'reports')


if __name__ == "__main__":
    # 测试配置管理器
    config = get_config()
    
    print("🔧 配置管理器测试")
    print("=" * 50)
    print(f"配置文件路径: {config.config_path}")
    print(f"版本: {config.get('version', '未知')}")
    print()
    print("配置项:")
    print(f"  最大迭代次数: {get_max_iterations()}")
    print(f"  最大行长度: {get_max_line_length()}")
    print(f"  最大函数长度: {get_max_function_length()}")
    print(f"  备份目录: {get_backup_dir()}")
    print(f"  最大分析文件数: {get_max_files_to_analyze()}")
    print(f"  启用策略: {', '.join(get_enabled_strategies())}")
