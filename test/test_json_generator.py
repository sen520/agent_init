#!/usr/bin/env python3
"""
测试 JSON Generator - src/tools/json_generator.py
"""
import pytest
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.json_generator import JSONGenerationStrategy, OptimizationStrategy


class TestJSONGenerationStrategy:
    """JSONGenerationStrategy 测试类"""
    
    def test_init(self):
        """测试初始化"""
        generator = JSONGenerationStrategy()
        assert generator is not None
        assert 'code_optimization' in generator.templates
    
    def test_create_strategy(self):
        """测试创建策略"""
        generator = JSONGenerationStrategy()
        
        strategy = generator.create_strategy("code_optimization")
        
        assert strategy is not None
        assert strategy.name == "通用代码优化"
    
    def test_create_strategy_not_found(self):
        """测试创建不存在的策略"""
        generator = JSONGenerationStrategy()
        
        strategy = generator.create_strategy("nonexistent")
        
        assert strategy is None
    
    def test_get_available_strategies(self):
        """测试获取可用策略"""
        generator = JSONGenerationStrategy()
        
        strategies = generator.get_available_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0


class TestOptimizationStrategy:
    """OptimizationStrategy 模型测试"""
    
    def test_create_strategy(self):
        """测试创建策略模型"""
        strategy = OptimizationStrategy(
            name="test_strategy",
            description="Test description",
            file_patterns=["*.py"],
            transformations=[{"type": "test"}],
            priority=3
        )
        
        assert strategy.name == "test_strategy"
        assert strategy.priority == 3
    
    def test_strategy_to_json(self):
        """测试策略转 JSON"""
        strategy = OptimizationStrategy(
            name="test_strategy",
            description="Test description",
            file_patterns=["*.py"],
            transformations=[{"type": "test"}]
        )
        
        json_str = strategy.model_dump_json()
        
        assert "test_strategy" in json_str
        assert "Test description" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
