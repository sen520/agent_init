from collections import defaultdict
from pathlib import Path
from src.tools.ast_parser import ASTParser
import ast
import os
import pytest
import sys
import sys

#!/usr/bin/env python3
"""
测试 AST Parser - src/tools/ast_parser.py
"""

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestASTParser:
    """ASTParser 测试类"""
    
    def test_init(self):
        """测试初始化"""
        parser = ASTParser()
        assert parser is not None
        assert parser.imports == []
        assert parser.functions == []
        assert parser.classes == []
    
    def test_analyze_file(self, tmp_path):
        """测试分析文件"""
        parser = ASTParser()
        
        # 创建测试文件
        test_file = tmp_path / "test.py"
        test_file.write_text("""

def hello():
    return "Hello"

class MyClass:

    def method(self):
        pass
""")
        
        result = parser.analyze_file(str(test_file))
        
        assert 'imports' in result
        assert 'functions' in result
        assert 'classes' in result
        assert 'cyclomatic_complexity' in result
    
    def test_analyze_nonexistent_file(self):
        """测试分析不存在的文件"""
        parser = ASTParser()
        
        result = parser.analyze_file("/nonexistent/file.py")
        
        assert 'error' in result
    
    def test_extract_imports(self):
        """测试提取导入"""
        parser = ASTParser()
        
        code = """
import os
import sys
from collections import defaultdict
"""
        tree = ast.parse(code)
        imports = parser._extract_imports(tree)
        
        assert len(imports) >= 2
    
    def test_extract_functions(self):
        """测试提取函数"""
        parser = ASTParser()
        
        code = """

def func1():
    pass

def func2(x, y):
    return x + y
"""
        tree = ast.parse(code)
        functions = parser._extract_functions(tree)
        
        assert len(functions) == 2
        assert functions[0]['name'] == 'func1'
        assert functions[1]['name'] == 'func2'
    
    def test_extract_classes(self):
        """测试提取类"""
        parser = ASTParser()
        
        code = """

class Class1:
    pass

class Class2:

    def method(self):
        pass
"""
        tree = ast.parse(code)
        classes = parser._extract_classes(tree)
        
        assert len(classes) == 2
        assert classes[0]['name'] == 'Class1'
    
    def test_calculate_complexity(self):
        """测试计算复杂度"""
        parser = ASTParser()
        
        code = """

def simple():
    return 1

def complex_func(x):
    if x > 0:
        if x < 10:
            return x
    return 0
"""
        tree = ast.parse(code)
        complexity = parser._calculate_complexity(tree)
        
        assert complexity > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])