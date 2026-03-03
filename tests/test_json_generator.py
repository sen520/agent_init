"""
测试JSON策略生成器
"""
import pytest
from datetime import datetime
from base64 import b64encode, b64decode
from src.tools.json_generator import JSONGenerationStrategy, OptimizationStrategy


class TestJSONGenerationStrategy:
    """测试JSON策略生成器"""
    
    def test_initialization(self):
        """测试初始化"""
        generator = JSONGenerationStrategy()
        assert len(generator.templates) == 3
        assert "code_optimization" in generator.templates
        assert "performance_optimization" in generator.templates
        assert "security_optimization" in generator.templates
    
    def test_generate_code_optimization_strategy(self):
        """测试生成代码优化策略"""
        generator = JSONGenerationStrategy()
        analysis_data = {
            "issues": [
                {"type": "style", "message": "命名问题"}
            ]
        }
        
        strategy = generator.generate_strategy(analysis_data)
        
        assert isinstance(strategy, OptimizationStrategy)
        assert "通用代码优化" in strategy.name
        assert len(strategy.transformations) > 0
        assert "*.py" in strategy.file_patterns
    
    def test_generate_security_optimization_strategy(self):
        """测试生成安全优化策略"""
        generator = JSONGenerationStrategy()
        analysis_data = {
            "issues": [
                {"type": "security", "message": "安全漏洞"}
            ]
        }
        
        strategy = generator.generate_strategy(analysis_data)
        
        assert isinstance(strategy, OptimizationStrategy)
        assert "安全优化策略" in strategy.name
        assert strategy.priority == 9
    
    def test_generate_performance_optimization_strategy(self):
        """测试生成性能优化策略"""
        generator = JSONGenerationStrategy()
        analysis_data = {
            "issues": [
                {"type": "performance", "message": "性能瓶颈"}
            ]
        }
        
        strategy = generator.generate_strategy(analysis_data)
        
        assert isinstance(strategy, OptimizationStrategy)
        assert "性能优化策略" in strategy.name
        assert strategy.priority == 8
    
    def test_customize_transformations_with_complexity_issues(self):
        """测试根据复杂度问题自定义转换"""
        generator = JSONGenerationStrategy()
        analysis_data = {
            "issues": [
                {"type": "complexity", "message": "代码复杂"}
            ]
        }
        
        strategy = generator.generate_strategy(analysis_data)
        
        # 应该添加复杂度降低转换
        complexity_transforms = [
            t for t in strategy.transformations 
            if t.get("type") == "complexity_reduction"
        ]
        assert len(complexity_transforms) > 0
    
    def test_save_and_load_strategy(self, tmp_path):
        """测试保存和加载策略"""
        generator = JSONGenerationStrategy()
        analysis_data = {"issues": []}
        
        strategy = generator.generate_strategy(analysis_data)
        
        # 保存策略
        save_path = tmp_path / "strategy.json"
        generator.save_strategy(strategy, str(save_path))
        
        assert save_path.exists()
        
        # 加载策略
        loaded_strategy = generator.load_strategy(str(save_path))
        
        assert loaded_strategy.name == strategy.name
        assert loaded_strategy.description == strategy.description
    
    def test_add_custom_template(self):
        """测试添加自定义模板"""
        generator = JSONGenerationStrategy()
        
        custom_template = {
            "name": "自定义优化",
            "description": "自定义优化策略",
            "file_patterns": ["*.js"],
            "transformations": [
                {
                    "type": "custom_transform",
                    "description": "自定义转换",
                    "action": "custom_action"
                }
            ],
            "priority": 7
        }
        
        generator.add_template("custom", custom_template)
        
        assert "custom" in generator.templates
        assert generator.templates["custom"]["name"] == "自定义优化"