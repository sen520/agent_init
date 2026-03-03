#!/usr/bin/env python3
"""
JSON优化策略生成器 - 实际的工具类
从测试文件中分离出来
"""
import json
from typing import Dict, Any, List
from pydantic import BaseModel
from datetime import datetime

class OptimizationStrategy(BaseModel):
    """优化策略模型"""
    name: str
    description: str
    file_patterns: List[str]
    transformations: List[Dict[str, Any]]
    priority: int = 5
    
class JSONGenerationStrategy:
    """JSON优化策略生成器"""
    
    def __init__(self):
        """初始化生成器"""
        self.templates = {
            "code_optimization": {
                "name": "通用代码优化",
                "description": "提升代码质量和性能的通用优化策略",
                "file_patterns": ["*.py"],
                "transformations": [
                    {
                        "type": "refactor_function",
                        "description": "重构过长的函数",
                        "max_lines": 50,
                        "action": "extract_to_function"
                    },
                    {
                        "type": "optimize_imports", 
                        "description": "优化导入语句",
                        "action": "remove_unused_imports"
                    },
                    {
                        "type": "improve_naming",
                        "description": "改进变量和函数命名",
                        "action": "apply_snake_case"
                    }
                ],
                "priority": 5
            },
            "performance_optimization": {
                "name": "性能优化策略",
                "description": "针对性能瓶颈的优化策略",
                "file_patterns": ["*.py"],
                "transformations": [
                    {
                        "type": "algorithm_optimization",
                        "description": "算法复杂度优化",
                        "action": "replace_with_efficient_algorithm"
                    },
                    {
                        "type": "memory_optimization",
                        "description": "内存使用优化", 
                        "action": "reduce_memory_footprint"
                    },
                    {
                        "type": "caching",
                        "description": "添加缓存机制",
                        "action": "implement_caching"
                    }
                ],
                "priority": 8
            },
            "security_optimization": {
                "name": "安全优化策略",
                "description": "提升代码安全性的优化策略",
                "file_patterns": ["*.py"],
                "transformations": [
                    {
                        "type": "input_validation",
                        "description": "输入参数验证",
                        "action": "add_parameter_validation"
                    },
                    {
                        "type": "error_handling",
                        "description": "安全错误处理",
                        "action": "implement_secure_error_handling"
                    },
                    {
                        "type": "dependency_update",
                        "description": "更新安全依赖",
                        "action": "update_vulnerable_dependencies"
                    }
                ],
                "priority": 9
            }
        }
    
    def generate_strategy(self, analysis_data: Dict[str, Any]) -> OptimizationStrategy:
        """
        根据分析数据生成优化策略
        
        Args:
            analysis_data: 代码分析数据
            
        Returns:
            OptimizationStrategy: 生成的优化策略
        """
        # 根据分析数据选择合适的模板
        template_type = self._select_template_type(analysis_data)
        template = self.templates[template_type]
        
        # 自定义策略
        strategy_data = template.copy()
        strategy_data["name"] = f"{template['name']}_{datetime.now().strftime('%Y%m%d')}"
        
        # 根据具体问题调整转换
        strategy_data["transformations"] = self._customize_transformations(
            template["transformations"], 
            analysis_data
        )
        
        return OptimizationStrategy(**strategy_data)
    
    def _select_template_type(self, analysis_data: Dict[str, Any]) -> str:
        """
        根据分析数据选择模板类型
        
        Args:
            analysis_data: 代码分析数据
            
        Returns:
            str: 模板类型
        """
        # 简单的选择逻辑，实际可以根据更复杂的规则
        issues = analysis_data.get("issues", [])
        
        security_issues = [i for i in issues if "security" in i.get("type", "").lower()]
        performance_issues = [i for i in issues if "performance" in i.get("type", "").lower()]
        
        if security_issues:
            return "security_optimization"
        elif performance_issues:
            return "performance_optimization"
        else:
            return "code_optimization"
    
    def _customize_transformations(
        self, 
        base_transformations: List[Dict[str, Any]], 
        analysis_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        根据分析数据自定义转换
        
        Args:
            base_transformations: 基础转换列表
            analysis_data: 分析数据
            
        Returns:
            List[Dict[str, Any]]: 自定义的转换列表
        """
        transformations = base_transformations.copy()
        
        # 根据具体问题添加或修改转换
        issues = analysis_data.get("issues", [])
        
        # 如果有复杂度问题，添加相关转换
        complexity_issues = [i for i in issues if "complexity" in i.get("type", "").lower()]
        if complexity_issues:
            transformations.append({
                "type": "complexity_reduction",
                "description": "降低代码复杂度",
                "action": "simplify_complex_logic"
            })
        
        return transformations
    
    def save_strategy(self, strategy: OptimizationStrategy, output_path: str) -> None:
        """
        保存策略到JSON文件
        
        Args:
            strategy: 要保存的策略
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(strategy.model_dump(), f, indent=2, ensure_ascii=False)
    
    def load_strategy(self, input_path: str) -> OptimizationStrategy:
        """
        从JSON文件加载策略
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            OptimizationStrategy: 加载的策略
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return OptimizationStrategy(**data)
    
    def get_available_templates(self) -> List[str]:
        """获取可用的模板列表"""
        return list(self.templates.keys())
    
    def add_template(self, template_type: str, template_data: Dict[str, Any]) -> None:
        """
        添加新的模板
        
        Args:
            template_type: 模板类型
            template_data: 模板数据
        """
        self.templates[template_type] = template_data