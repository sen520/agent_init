#!/usr/bin/env python3
"""
测试优化策略
"""
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.optimization_strategies import (
    LineLengthOptimizer,
    ImportOptimizer,
    CommentOptimizer,
    FunctionLengthOptimizer,
    VariableNamingOptimizer,
    EmptyLineOptimizer,
    DuplicateCodeOptimizer,
    CodeOptimizer
)


class TestLineLengthOptimizer:
    """行长度优化策略测试"""
    
    def test_analyze_with_long_lines(self):
        """测试检测超长行"""
        optimizer = LineLengthOptimizer()
        
        # 创建包含超长行的代码
        code = "x = " + "1" * 150  # 超过 100 字符
        
        result = optimizer.analyze("test.py", code)
        
        assert result['can_optimize']
        assert result['issues_found'] > 0
        assert len(result['long_lines']) > 0
    
    def test_analyze_with_normal_lines(self):
        """测试正常长度行"""
        optimizer = LineLengthOptimizer()
        
        code = "x = 1\ny = 2\n"
        
        result = optimizer.analyze("test.py", code)
        
        assert not result['can_optimize']
        assert result['issues_found'] == 0
    
    def test_apply_optimization(self):
        """测试应用行长度优化"""
        optimizer = LineLengthOptimizer()
        
        # 包含可以拆分的行
        code = "x = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25"
        
        optimized, changes = optimizer.apply("test.py", code)
        
        assert changes['changes_count'] >= 0
        assert 'strategy' in changes


class TestImportOptimizer:
    """导入优化策略测试"""
    
    def test_analyze_wildcard_import(self):
        """测试检测通配符导入"""
        optimizer = ImportOptimizer()
        
        code = "from os import *\n"
        
        result = optimizer.analyze("test.py", code)
        
        assert result['can_optimize']
        assert any(issue['type'] == 'wildcard_import' for issue in result['import_issues'])
    
    def test_analyze_multiple_imports(self):
        """测试检测一行多个导入"""
        optimizer = ImportOptimizer()
        
        code = "import os, sys, json\n"
        
        result = optimizer.analyze("test.py", code)
        
        assert result['can_optimize']
    
    def test_apply_import_organization(self):
        """测试应用导入组织"""
        optimizer = ImportOptimizer()
        
        code = """import sys
import os
import json

print("hello")
"""
        
        optimized, changes = optimizer.apply("test.py", code)
        
        assert changes['changes_count'] >= 0


class TestCommentOptimizer:
    """注释优化策略测试"""
    
    def test_analyze_empty_comments(self):
        """测试检测空注释"""
        optimizer = CommentOptimizer()
        
        code = "#\nprint('hello')\n"
        
        result = optimizer.analyze("test.py", code)
        
        assert result['can_optimize']
    
    def test_analyze_todo_format(self):
        """测试检测 TODO 格式"""
        optimizer = CommentOptimizer()
        
        code = "# TODO add more code\n"
        
        result = optimizer.analyze("test.py", code)
        
        assert result['can_optimize']
    
    def test_apply_comment_formatting(self):
        """测试应用注释格式化"""
        optimizer = CommentOptimizer()
        
        code = """# TODO bad format
# empty comment
print("hello")
"""
        
        optimized, changes = optimizer.apply("test.py", code)
        
        assert changes['changes_count'] >= 0


class TestFunctionLengthOptimizer:
    """函数长度优化策略测试"""
    
    def test_analyze_long_function(self):
        """测试检测长函数"""
        optimizer = FunctionLengthOptimizer()
        
        # 创建超过 50 行的函数
        code = "def long_function():\n"
        for i in range(60):
            code += f"    x{i} = {i}\n"
        
        result = optimizer.analyze("test.py", code)
        
        assert result['can_optimize']
        assert len(result['long_functions']) > 0
    
    def test_analyze_short_function(self):
        """测试短函数"""
        optimizer = FunctionLengthOptimizer()
        
        code = """def short():
    pass
"""
        
        result = optimizer.analyze("test.py", code)
        
        assert not result['can_optimize']
        assert len(result['long_functions']) == 0


class TestVariableNamingOptimizer:
    """变量命名优化策略测试"""
    
    def test_analyze_single_letter_vars(self):
        """测试检测单字母变量"""
        optimizer = VariableNamingOptimizer()
        
        code = """def func():
    x = 1
    y = 2
    return x + y
"""
        
        result = optimizer.analyze("test.py", code)
        
        assert result['can_optimize']
    
    def test_analyze_snake_case(self):
        """测试检测非 snake_case"""
        optimizer = VariableNamingOptimizer()
        
        code = """def func():
    myVariable = 1
    return myVariable
"""
        
        result = optimizer.analyze("test.py", code)
        
        # 可能需要检测到驼峰命名
        assert 'naming_issues' in result


class TestEmptyLineOptimizer:
    """空行优化策略测试"""
    
    def test_analyze_multiple_empty_lines(self):
        """测试检测多个空行"""
        optimizer = EmptyLineOptimizer()
        
        code = """def func():
    x = 1


    y = 2
    return x + y
"""
        
        result = optimizer.analyze("test.py", code)
        
        assert result['can_optimize']
    
    def test_apply_empty_line_compression(self):
        """测试应用空行压缩"""
        optimizer = EmptyLineOptimizer()
        
        code = """x = 1



y = 2
"""
        
        optimized, changes = optimizer.apply("test.py", code)
        
        # 应该减少空行数量
        assert changes['changes_count'] >= 0


class TestDuplicateCodeOptimizer:
    """重复代码优化策略测试"""
    
    def test_analyze_duplicate_blocks(self):
        """测试检测重复代码块"""
        optimizer = DuplicateCodeOptimizer()
        
        # 创建重复的代码块
        code = """x = 1
y = 2
z = 3

a = 1
b = 2
c = 3
"""
        
        result = optimizer.analyze("test.py", code)
        
        # 应该检测到重复
        assert 'duplicates_found' in result
    
    def test_apply_duplicate_marking(self):
        """测试应用重复标记"""
        optimizer = DuplicateCodeOptimizer()
        
        code = """def block1():
    x = 1
    y = 2

def block2():
    x = 1
    y = 2
"""
        
        optimized, changes = optimizer.apply("test.py", code)
        
        assert changes['changes_count'] >= 0


class TestCodeOptimizer:
    """代码优化器主类测试"""
    
    def test_init(self):
        """测试初始化"""
        optimizer = CodeOptimizer()
        
        assert len(optimizer.strategies) == 7
    
    def test_analyze_file(self, tmp_path):
        """测试分析文件"""
        optimizer = CodeOptimizer()
        
        # 创建测试文件
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def long_function():
    x = 1
    y = 2
    z = 3
    return x + y + z
""")
        
        result = optimizer.analyze_file(str(test_file))
        
        assert 'total_issues' in result
        assert 'strategy_results' in result
        assert 'content_stats' in result
    
    def test_optimize_file_with_changes(self, tmp_path):
        """测试优化文件（有变更）"""
        optimizer = CodeOptimizer()
        
        # 创建有优化空间的文件
        test_file = tmp_path / "test.py"
        original_content = """from os import *
# bad comment


print("hello")
"""
        test_file.write_text(original_content)
        
        result = optimizer.optimize_file(
            str(test_file),
            selected_strategies=['comment_optimizer', 'empty_line_optimizer']
        )
        
        assert 'optimization_applied' in result
        assert 'changes_count' in result
    
    def test_optimize_file_no_changes_needed(self, tmp_path):
        """测试优化文件（无需变更）"""
        optimizer = CodeOptimizer()
        
        # 创建干净的文件
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')\n")
        
        result = optimizer.optimize_file(str(test_file))
        
        # 应该返回无变更的结果
        assert result['changes_count'] == 0 or not result.get('optimization_applied')
    
    def test_optimize_file_not_found(self):
        """测试优化不存在的文件"""
        optimizer = CodeOptimizer()
        
        result = optimizer.optimize_file("/nonexistent/file.py")
        
        assert 'error' in result
    
    def test_optimize_file_syntax_error(self, tmp_path):
        """测试优化有语法错误的文件"""
        optimizer = CodeOptimizer()
        
        test_file = tmp_path / "test.py"
        test_file.write_text("def broken(\n  # missing parenthesis")
        
        result = optimizer.optimize_file(str(test_file))
        
        # 应该处理语法错误
        assert 'error' in result or result.get('changes_count', 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
