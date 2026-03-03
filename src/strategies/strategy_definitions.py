"""
策略定义 - 从测试文件中分离出来的实际策略定义
"""
from typing import Any, Dict

class BaseStrategy:
    """策略基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.__class__.__name__,
            "config": self.config
        }

class SecurityStrategy(BaseStrategy):
    """安全加固策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            "encryption_enabled": True,
            "authentication_required": True,
            "audit_logging": True
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)


class PerformanceStrategy(BaseStrategy):
    """性能优化策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            "caching_enabled": True,
            "database_indexing": True,
            "async_processing": True,
            "resource_monitoring": True
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)


class ScalabilityStrategy(BaseStrategy):
    """可扩展性策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            "load_balancing": True,
            "horizontal_scaling": True,
            "containerization": True,
            "microservices": True
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)


class ComplianceStrategy(BaseStrategy):
    """合规性策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            "data_protection": True,
            "privacy_controls": True,
            "regulatory_reporting": True,
            "secure_documentation": True
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)


class CostOptimizationStrategy(BaseStrategy):
    """成本优化策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            "resource_optimization": True,
            "auto_scaling": True,
            "spot_instances": True,
            "lifecycle_management": True
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)


# 策略注册表
STRATEGY_REGISTRY = {
    "security": SecurityStrategy,
    "performance": PerformanceStrategy,
    "scalability": ScalabilityStrategy,
    "compliance": ComplianceStrategy,
    "cost_optimization": CostOptimizationStrategy,
}


def get_strategy_type(strategy_name: str) -> BaseStrategy:
    """
    获取策略类型
    
    Args:
        strategy_name: 策略名称
        
    Returns:
        BaseStrategy: 策略实例
        
    Raises:
        ValueError: 如果策略名称不存在
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略类型: {strategy_name}. 可用策略: {list(STRATEGY_REGISTRY.keys())}")
    
    return STRATEGY_REGISTRY[strategy_name]


def list_available_strategies() -> list:
    """获取可用策略列表"""
    return list(STRATEGY_REGISTRY.keys())


def create_strategy(strategy_name: str, config: Dict[str, Any] = None) -> BaseStrategy:
    """
    创建策略实例
    
    Args:
        strategy_name: 策略名称
        config: 策略配置
        
    Returns:
        BaseStrategy: 策略实例
    """
    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"未知策略: {strategy_name}")
    
    return strategy_class(config)