#!/usr/bin/env python3
"""
测试 simple_analyzer - src/tools/simple_analyzer.py
"""
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.simple_analyzer import SimpleCodeAnalyzer


class TestSimpleCodeAnalyzer:
    """SimpleCodeAnalyzer 测试类"""
    
    def test_init(self):
        """测试初始化"""
        analyzer = SimpleCodeAnalyzer()
        assert analyzer is not None
        assert analyzer.config['max_line_length'] == 100
    
    def test_analyze_valid_file(self, tmp_path):
        """测试分析有效文件"""
        analyzer = SimpleCodeAnalyzer()
        
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')\n")
        
        result = analyzer.analyze_file(str(test_file))
        
        assert 'file_path' in result
        assert 'total_lines' in result
        assert result['total_lines'] == 1
    
    def test_analyze_with_long_lines(self, tmp_path):
        """测试分析包含超长行的文件"""
        analyzer = SimpleCodeAnalyzer()
        
        test_file = tmp_path / "long.py"
        test_file.write_text("x = " + "1" * 150 + "\n")
        
        result = analyzer.analyze_file(str(test_file))
        
        assert result['long_lines_count'] > 0
    
    def test_analyze_with_functions(self, tmp_path):
        """测试分析包含函数的文件"""
        analyzer = SimpleCodeAnalyzer()
        
        test_file = tmp_path / "funcs.py"
        test_file.write_text("""
def func1():
    pass

def func2():
    pass
""")
        
        result = analyzer.analyze_file(str(test_file))
        
        assert result['functions_count'] == 2
    
    def test_analyze_with_classes(self, tmp_path):
        """测试分析包含类的文件"""
        analyzer = SimpleCodeAnalyzer()
        
        test_file = tmp_path / "classes.py"
        test_file.write_text("""
class Class1:
    pass

class Class2:
    pass
""")
        
        result = analyzer.analyze_file(str(test_file))
        
        assert result['classes_count'] == 2
    
    def test_analyze_with_imports(self, tmp_path):
        """测试分析包含导入的文件"""
        analyzer = SimpleCodeAnalyzer()
        
        test_file = tmp_path / "imports.py"
        test_file.write_text("""
import os
import sys
from collections import defaultdict
""")
        
        result = analyzer.analyze_file(str(test_file))
        
        assert result['imports_count'] == 3
    
    def test_analyze_nonexistent_file(self):
        """测试分析不存在的文件"""
        analyzer = SimpleCodeAnalyzer()
        
        result = analyzer.analyze_file("/nonexistent/file.py")
        
        assert 'error' in result
    
    def test_analyze_with_syntax_error(self, tmp_path):
        """测试分析有语法错误的文件"""
        analyzer = SimpleCodeAnalyzer()
        
        test_file = tmp_path / "broken.py"
        test_file.write_text("def broken(")
        
        result = analyzer.analyze_file(str(test_file))
        
        assert 'syntax_error' in result or 'error' in result
    
    def test_analyze_project(self, tmp_path):
        """测试分析整个项目"""
        analyzer = SimpleCodeAnalyzer()
        
        # 创建多个文件
        (tmp_path / "main.py").write_text("print('main')")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        
        result = analyzer.analyze_project(str(tmp_path))
        
        assert 'total_files' in result
        assert 'total_lines' in result
        assert result['total_files'] >= 2
    
    def test_get_summary(self, tmp_path):
        """测试获取摘要"""
        analyzer = SimpleCodeAnalyzer()
        
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')\n")
        
        result = analyzer.analyze_file(str(test_file))
        summary = analyzer.get_summary()
        
        assert isinstance(summary, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
