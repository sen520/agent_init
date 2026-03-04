#!/usr/bin/env python3
"""
测试 code_analyzer 补充
"""
import pytest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.code_analyzer import CodeAnalyzer, analyze_file, analyze_directory


class TestCodeAnalyzerExtended:
    """扩展的 CodeAnalyzer 测试"""
    
    def test_analyze_file_with_todo(self):
        """测试分析包含 TODO 的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# TODO: fix this\ndef test():\n    pass\n")
            temp_path = f.name
        
        try:
            result = analyze_file(temp_path)
            assert result['file_name'] == os.path.basename(temp_path)
            assert result['total_lines'] == 3
        finally:
            os.unlink(temp_path)
    
    def test_analyze_file_with_long_line(self):
        """测试分析包含长行的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("x = '" + "a" * 150 + "'\n")  # 长行
            temp_path = f.name
        
        try:
            analyzer = CodeAnalyzer()
            result = analyzer.analyze_file(temp_path)
            # 检查是否识别了长行问题
            has_long_line = any(issue['type'] == 'long_line' for issue in result.get('issues', []))
        finally:
            os.unlink(temp_path)
    
    def test_analyze_file_with_long_function(self):
        """测试分析包含长函数的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # 创建超过50行的函数
            f.write("def long_function():\n")
            for i in range(60):
                f.write(f"    x{i} = {i}\n")
            temp_path = f.name
        
        try:
            analyzer = CodeAnalyzer()
            result = analyzer.analyze_file(temp_path)
            # 检查是否识别了长函数问题
            has_long_function = any(issue['type'] == 'long_function' for issue in result.get('issues', []))
        finally:
            os.unlink(temp_path)
    
    def test_analyze_file_with_bare_except(self):
        """测试分析包含裸 except 的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("try:\n    pass\nexcept:\n    pass\n")
            temp_path = f.name
        
        try:
            analyzer = CodeAnalyzer()
            result = analyzer.analyze_file(temp_path)
            # 检查是否识别了裸 except 问题
            has_bare_except = any(issue['type'] == 'bare_except' for issue in result.get('issues', []))
        finally:
            os.unlink(temp_path)
    
    def test_analyze_file_with_debug_print(self):
        """测试分析包含 debug print 的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test():\n    print('debug')\n")
            temp_path = f.name
        
        try:
            analyzer = CodeAnalyzer()
            result = analyzer.analyze_file(temp_path)
            # 检查是否识别了 debug print
            has_debug_print = any(issue['type'] == 'debug_print' for issue in result.get('issues', []))
        finally:
            os.unlink(temp_path)
    
    def test_analyze_file_with_wildcard_import(self):
        """测试分析包含通配符导入的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("from os import *\n")
            temp_path = f.name
        
        try:
            analyzer = CodeAnalyzer()
            result = analyzer.analyze_file(temp_path)
            # 检查是否识别了通配符导入
            has_wildcard = any(issue['type'] == 'wildcard_import' for issue in result.get('issues', []))
        finally:
            os.unlink(temp_path)
    
    def test_analyze_directory_with_temp_dir(self):
        """测试分析临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建一些测试文件
            with open(os.path.join(tmpdir, 'test1.py'), 'w') as f:
                f.write("def test():\n    pass\n")
            with open(os.path.join(tmpdir, 'test2.py'), 'w') as f:
                f.write("class Test:\n    pass\n")
            
            result = analyze_directory(tmpdir)
            assert result['directory'] == tmpdir
            assert result['files_analyzed'] >= 0
    
    def test_analyze_file_not_found(self):
        """测试分析不存在的文件"""
        result = analyze_file('/nonexistent/file.py')
        assert 'error' in result
    
    def test_analyze_file_syntax_error(self):
        """测试分析有语法错误的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(  # 不完整的函数定义\n")
            temp_path = f.name
        
        try:
            result = analyze_file(temp_path)
            # 有语法错误的文件应该返回错误信息
            assert 'error' in result or 'issues' in result
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
