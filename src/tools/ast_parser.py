"""
AST 解析器 - 真实分析 Python 代码
"""
import ast
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class ASTParser:
    """Python AST 解析器"""
    
    def __init__(self):
        self.imports = []
        self.functions = []
        self.classes = []
        self.complexities = []
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析单个 Python 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 收集信息
            imports = self._extract_imports(tree)
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            complexity = self._calculate_complexity(tree)
            
            return {
                "file_path": file_path,
                "lines_of_code": len(content.splitlines()),
                "imports": imports,
                "functions": functions,
                "classes": classes,
                "cyclomatic_complexity": complexity,
                "has_docstring": self._has_docstring(tree),
                "ast_valid": True
            }
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"语法错误: {e}",
                "ast_valid": False
            }
        except Exception as e:
            return {
                "file_path": file_path,
                "error": f"分析错误: {e}",
                "ast_valid": False
            }
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """提取所有导入"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """提取所有函数定义"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": len(node.args.args),
                    "has_docstring": ast.get_docstring(node) is not None,
                    "complexity": self._function_complexity(node)
                }
                functions.append(func_info)
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """提取所有类定义"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    "has_docstring": ast.get_docstring(node) is not None
                }
                classes.append(class_info)
        return classes
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """计算代码复杂度"""
        complexities = []
        
        # 计算每个函数的复杂度
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexities.append(self._function_complexity(node))
        
        return sum(complexities) if complexities else 0.0
    
    def _function_complexity(self, node: ast.FunctionDef) -> int:
        """计算单个函数的圈复杂度"""
        complexity = 1  # 基础复杂度
        
        for child in ast.walk(node):
            # 各种控制流结构增加复杂度
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _has_docstring(self, tree: ast.AST) -> bool:
        """检查模块是否有文档字符串"""
        docstring = ast.get_docstring(tree)
        return docstring is not None
    
    def find_issues(self, file_path: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测代码问题"""
        issues = []
        
        if not analysis.get("ast_valid", False):
            issues.append({
                "type": "syntax_error",
                "description": analysis.get("error", "语法错误"),
                "line": 1,
                "severity": "high"
            })
            return issues
        
        # 检查缺少文档字符串的函数
        for func in analysis.get("functions", []):
            if not func.get("has_docstring", False):
                issues.append({
                    "type": "documentation",
                    "description": f"函数 '{func['name']}' 缺少文档字符串",
                    "line": func.get("line", 1),
                    "severity": "medium"
                })
        
        # 检查缺少文档字符串的类
        for cls in analysis.get("classes", []):
            if not cls.get("has_docstring", False):
                issues.append({
                    "type": "documentation",
                    "description": f"类 '{cls['name']}' 缺少文档字符串",
                    "line": cls.get("line", 1),
                    "severity": "medium"
                })
        
        # 检查复杂度高的函数
        for func in analysis.get("functions", []):
            if func.get("complexity", 0) > 5:
                issues.append({
                    "type": "complexity",
                    "description": f"函数 '{func['name']}' 圈复杂度较高 ({func['complexity']})",
                    "line": func.get("line", 1),
                    "severity": "medium"
                })
        
        return issues