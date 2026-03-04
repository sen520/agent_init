#!/usr/bin/env python3
"""
测试 code_analyzer 模块 - 补充测试
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCodeAnalyzer:
    """CodeAnalyzer 测试"""

    def test_code_analyzer_import(self):
        """测试 CodeAnalyzer 可以导入"""
        from src.tools.code_analyzer import CodeAnalyzer
        assert CodeAnalyzer is not None

    def test_code_analyzer_init(self):
        """测试 CodeAnalyzer 初始化"""
        from src.tools.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer()
        assert analyzer is not None

    def test_analyze_file(self):
        """测试分析文件"""
        from src.tools.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer()
        # 分析当前测试文件
        result = analyzer.analyze_file(__file__)
        assert 'file_path' in result
        assert 'total_lines' in result

    def test_analyze_directory(self):
        """测试分析目录"""
        from src.tools.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer()
        result = analyzer.analyze_directory(str(Path(__file__).parent))
        assert 'directory' in result
        assert 'files_analyzed' in result


class TestCodeAnalyzerFunctions:
    """CodeAnalyzer 便捷函数测试"""

    def test_analyze_file_function(self):
        """测试 analyze_file 便捷函数"""
        from src.tools.code_analyzer import analyze_file

        result = analyze_file(__file__)
        assert 'file_path' in result

    def test_analyze_directory_function(self):
        """测试 analyze_directory 便捷函数"""
        from src.tools.code_analyzer import analyze_directory

        result = analyze_directory(str(Path(__file__).parent))
        assert 'directory' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
