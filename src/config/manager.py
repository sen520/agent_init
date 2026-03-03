#!/usr/bin/env python3
"""
配置管理器 - 集中管理所有配置项（带验证）
"""
import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigManager:
    """配置管理器 - 从 config.json 读取配置（带验证）"""
    
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
                logger.info(f"配置加载成功: {self.config_path}")
                
                # 验证配置
                self._validate_config()
                
            except json.JSONDecodeError as e:
                logger.error(f"配置文件解析错误: {e}")
                self._config = self._default_config()
            except ConfigValidationError as e:
                logger.error(f"配置验证失败: {e}")
                self._config = self._default_config()
        else:
            logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
            self._config = self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            "version": "1.0.0",
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
            "file_scanner": {"max_file_size_kb": 100},
            "testing": {"timeout": 300}
        }
    
    def _validate_config(self):
        """验证配置有效性"""
        errors = []
        
        # 验证 workflow
        max_iterations = self._config.get('workflow', {}).get('max_iterations')
        if max_iterations is not None:
            if not isinstance(max_iterations, int) or max_iterations < 1 or max_iterations > 100:
                errors.append(f"workflow.max_iterations 必须在 1-100 之间，当前: {max_iterations}")
        
        # 验证 analysis
        analysis = self._config.get('analysis', {})
        max_files_analyze = analysis.get('max_files_to_analyze')
        if max_files_analyze is not None:
            if not isinstance(max_files_analyze, int) or max_files_analyze < 1 or max_files_analyze > 1000:
                errors.append(f"analysis.max_files_to_analyze 必须在 1-1000 之间，当前: {max_files_analyze}")
        
        max_files_optimize = analysis.get('max_files_to_optimize')
        if max_files_optimize is not None:
            if not isinstance(max_files_optimize, int) or max_files_optimize < 1 or max_files_optimize > 1000:
                errors.append(f"analysis.max_files_to_optimize 必须在 1-1000 之间，当前: {max_files_optimize}")
        
        # 验证 optimization
        optimization = self._config.get('optimization', {})
        max_line_length = optimization.get('max_line_length')
        if max_line_length is not None:
            if not isinstance(max_line_length, int) or max_line_length < 50 or max_line_length > 500:
                errors.append(f"optimization.max_line_length 必须在 50-500 之间，当前: {max_line_length}")
        
        max_function_length = optimization.get('max_function_length')
        if max_function_length is not None:
            if not isinstance(max_function_length, int) or max_function_length < 10 or max_function_length > 500:
                errors.append(f"optimization.max_function_length 必须在 10-500 之间，当前: {max_function_length}")
        
        # 验证 file_scanner
        max_file_size = self._config.get('file_scanner', {}).get('max_file_size_kb')
        if max_file_size is not None:
            if not isinstance(max_file_size, int) or max_file_size < 1 or max_file_size > 10000:
                errors.append(f"file_scanner.max_file_size_kb 必须在 1-10000 之间，当前: {max_file_size}")
        
        # 验证 file_modifier
        backup_retention = self._config.get('file_modifier', {}).get('backup_retention_days')
        if backup_retention is not None:
            if not isinstance(backup_retention, int) or backup_retention < 1 or backup_retention > 365:
                errors.append(f"file_modifier.backup_retention_days 必须在 1-365 之间，当前: {backup_retention}")
        
        # 验证 report
        max_issues = self._config.get('report', {}).get('max_issues_in_report')
        if max_issues is not None:
            if not isinstance(max_issues, int) or max_issues < 10 or max_issues > 1000:
                errors.append(f"report.max_issues_in_report 必须在 10-1000 之间，当前: {max_issues}")
        
        # 验证 llm
        llm_config = self._config.get('llm', {})
        if llm_config.get('enabled', False):
            max_tokens = llm_config.get('max_tokens')
            if max_tokens is not None:
                if not isinstance(max_tokens, int) or max_tokens < 100 or max_tokens > 8000:
                    errors.append(f"llm.max_tokens 必须在 100-8000 之间，当前: {max_tokens}")
            
            temperature = llm_config.get('temperature')
            if temperature is not None:
                if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
                    errors.append(f"llm.temperature 必须在 0-2 之间，当前: {temperature}")
        
        # 验证 testing
        timeout = self._config.get('testing', {}).get('timeout')
        if timeout is not None:
            if not isinstance(timeout, int) or timeout < 10 or timeout > 3600:
                errors.append(f"testing.timeout 必须在 10-3600 秒之间，当前: {timeout}")
        
        if errors:
            error_msg = "\n  - ".join([""] + errors)
            raise ConfigValidationError(f"配置验证失败:{error_msg}")
        
        logger.info("配置验证通过")
    
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
    
    def get_testing_config(self) -> Dict:
        """获取测试配置"""
        return self._config.get('testing', {})
    
    def reload(self):
        """重新加载配置"""
        logger.info("重新加载配置...")
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
    logging.basicConfig(level=logging.INFO)
    print("🔧 配置管理器测试")
    print("=" * 50)
    
    try:
        config = get_config()
        print(f"✅ 配置加载成功: {config.config_path}")
        print(f"版本: {config.get('version', '未知')}")
        print()
        print("配置项:")
        print(f"  最大迭代次数: {get_max_iterations()}")
        print(f"  最大行长度: {get_max_line_length()}")
        print(f"  最大函数长度: {get_max_function_length()}")
        print(f"  备份目录: {get_backup_dir()}")
        print(f"  最大分析文件数: {get_max_files_to_analyze()}")
        print(f"  启用策略: {', '.join(get_enabled_strategies())}")
        print()
        print("✅ 配置验证通过！")
    except ConfigValidationError as e:
        print(f"❌ 配置验证失败: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")
