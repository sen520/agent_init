"""
测试策略定义
"""
import pytest
from src.strategies.strategy_definitions import (
    SecurityStrategy, PerformanceStrategy, ScalabilityStrategy,
    ComplianceStrategy, CostOptimizationStrategy,
    get_strategy_type, list_available_strategies, create_strategy
)


class TestStrategyDefinitions:
    """测试策略定义"""
    
    def test_security_strategy_default_config(self):
        """测试安全策略默认配置"""
        strategy = SecurityStrategy()
        
        assert strategy.config["encryption_enabled"] is True
        assert strategy.config["authentication_required"] is True
        assert strategy.config["audit_logging"] is True
    
    def test_security_strategy_custom_config(self):
        """测试安全策略自定义配置"""
        custom_config = {"encryption_enabled": False}
        strategy = SecurityStrategy(custom_config)
        
        assert strategy.config["encryption_enabled"] is False
        assert strategy.config["authentication_required"] is True  # 默认值
    
    def test_performance_strategy_default_config(self):
        """测试性能策略默认配置"""
        strategy = PerformanceStrategy()
        
        assert strategy.config["caching_enabled"] is True
        assert strategy.config["database_indexing"] is True
        assert strategy.config["async_processing"] is True
        assert strategy.config["resource_monitoring"] is True
    
    def test_scalability_strategy_default_config(self):
        """测试可扩展性策略默认配置"""
        strategy = ScalabilityStrategy()
        
        assert strategy.config["load_balancing"] is True
        assert strategy.config["horizontal_scaling"] is True
        assert strategy.config["containerization"] is True
        assert strategy.config["microservices"] is True
    
    def test_compliance_strategy_default_config(self):
        """测试合规性策略默认配置"""
        strategy = ComplianceStrategy()
        
        assert strategy.config["data_protection"] is True
        assert strategy.config["privacy_controls"] is True
        assert strategy.config["regulatory_reporting"] is True
        assert strategy.config["secure_documentation"] is True
    
    def test_cost_optimization_strategy_default_config(self):
        """测试成本优化策略默认配置"""
        strategy = CostOptimizationStrategy()
        
        assert strategy.config["resource_optimization"] is True
        assert strategy.config["auto_scaling"] is True
        assert strategy.config["spot_instances"] is True
        assert strategy.config["lifecycle_management"] is True
    
    def test_strategy_to_dict(self):
        """测试策略转换为字典"""
        strategy = SecurityStrategy()
        strategy_dict = strategy.to_dict()
        
        assert strategy_dict["type"] == "SecurityStrategy"
        assert "config" in strategy_dict
        assert strategy_dict["config"]["encryption_enabled"] is True
    
    def test_list_available_strategies(self):
        """测试列出可用策略"""
        strategies = list_available_strategies()
        
        expected_strategies = [
            "security", "performance", "scalability", 
            "compliance", "cost_optimization"
        ]
        
        for strategy in expected_strategies:
            assert strategy in strategies
    
    def test_get_strategy_type_valid(self):
        """测试获取有效策略类型"""
        strategy = get_strategy_type("security")
        
        assert isinstance(strategy, SecurityStrategy)
    
    def test_get_strategy_type_invalid(self):
        """测试获取无效策略类型"""
        with pytest.raises(ValueError) as excinfo:
            get_strategy_type("invalid_strategy")
        
        assert "未知策略类型" in str(excinfo.value)
        assert "invalid_strategy" in str(excinfo.value)
    
    def test_create_strategy_valid(self):
        """测试创建有效策略"""
        config = {"encryption_enabled": False}
        strategy = create_strategy("security", config)
        
        assert isinstance(strategy, SecurityStrategy)
        assert strategy.config["encryption_enabled"] is False
    
    def test_create_strategy_invalid(self):
        """测试创建无效策略"""
        with pytest.raises(ValueError):
            create_strategy("invalid_strategy")
    
    def test_all_strategies_have_default_configs(self):
        """测试所有策略都有默认配置"""
        strategies = list_available_strategies()
        
        for strategy_name in strategies:
            strategy = get_strategy_type(strategy_name)
            assert hasattr(strategy, 'config')
            assert len(strategy.config) > 0