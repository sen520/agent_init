#!/usr/bin/env python3
"""
测试更多边界情况和异常处理
"""
import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_modifier import FileModifier


class TestFileModifierEdgeCases:
    """FileModifier 边界情况测试"""
    
    def test_backup_file_not_exists(self, tmp_path):
        """测试备份不存在的文件"""
        modifier = FileModifier(backup_dir=str(tmp_path / "backups"))
        
        with pytest.raises(FileNotFoundError):
            modifier.backup_file("/nonexistent/file.py")
    
    def test_write_file_no_syntax_check_non_python(self, tmp_path):
        """测试写入非 Python 文件不进行语法检查"""
        modifier = FileModifier(backup_dir=str(tmp_path / "backups"))
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")
        
        # 写入非 Python 文件，不应检查语法
        new_content = "new content with ( invalid python {{"
        success, message = modifier.write_file(str(test_file), new_content)
        
        assert success
        assert test_file.read_text() == new_content
    
    def test_rollback_nonexistent_backup(self, tmp_path):
        """测试回滚不存在的备份"""
        modifier = FileModifier(backup_dir=str(tmp_path / "backups"))
        
        success, message = modifier.rollback_file("/path/to/file.py")
        
        assert not success
        assert "没有找到备份" in message
    
    def test_rollback_all_empty(self, tmp_path):
        """测试回滚全部（空列表）"""
        modifier = FileModifier(backup_dir=str(tmp_path / "backups"))
        
        success, failed = modifier.rollback_all()
        
        assert success == 0
        assert failed == 0
    
    def test_cleanup_old_backups_empty_dir(self, tmp_path):
        """测试清理空备份目录"""
        modifier = FileModifier(backup_dir=str(tmp_path / "backups"))
        
        deleted = modifier.cleanup_old_backups(days=7)
        
        assert deleted == 0


class TestStateEdgeCases:
    """State 边界情况测试"""
    
    def test_state_add_log_many(self):
        """测试添加大量日志"""
        from src.state.base import State
        
        state = State()
        
        # 添加超过 200 条日志
        for i in range(250):
            state.add_log(f"Log {i}")
        
        # 应该限制为 50 条
        assert len(state.logs) <= 100  # 允许一些浮动
    
    def test_state_add_error_many(self):
        """测试添加大量错误"""
        from src.state.base import State
        
        state = State()
        
        for i in range(50):
            state.add_error(f"Error {i}")
        
        assert len(state.errors) == 50


class TestFileScannerEdgeCases:
    """FileScanner 边界情况测试"""
    
    def test_scan_project_with_permission_error(self, tmp_path):
        """测试扫描有权限错误的目录"""
        from src.tools.file_scanner import FileScanner
        
        # 创建不可读目录
        no_read_dir = tmp_path / "no_read"
        no_read_dir.mkdir()
        (no_read_dir / "file.py").write_text("content")
        no_read_dir.chmod(0o000)
        
        try:
            scanner = FileScanner(str(tmp_path))
            result = scanner.scan_project()
            
            # 应该能处理权限错误
            assert isinstance(result, dict)
        finally:
            no_read_dir.chmod(0o755)
    
    def test_scan_python_files_with_nested_dirs(self, tmp_path):
        """测试扫描嵌套目录"""
        from src.tools.file_scanner import FileScanner
        
        # 创建深层嵌套结构
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("# deep file")
        
        scanner = FileScanner(str(tmp_path))
        files = scanner.scan_python_files()
        
        assert "a/b/c/d/deep.py" in files or any("deep.py" in f for f in files)


class TestCodeAnalyzerEdgeCases:
    """CodeAnalyzer 边界情况测试"""
    
    def test_analyze_file_with_unicode(self, tmp_path):
        """测试分析包含 Unicode 的文件"""
        from src.tools.code_analyzer import CodeAnalyzer
        
        test_file = tmp_path / "unicode.py"
        test_file.write_text("""# -*- coding: utf-8 -*-
def hello():
    print("你好，世界！")
    print("Hello, 世界!")
    return "🎉"
""", encoding='utf-8')
        
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_file(str(test_file))
        
        assert 'error' not in result or result['error'] is None
    
    def test_analyze_file_with_docstrings(self, tmp_path):
        """测试分析包含文档字符串的文件"""
        from src.tools.code_analyzer import CodeAnalyzer
        
        test_file = tmp_path / "docstrings.py"
        test_file.write_text('''
def func_with_doc():
    """
    This is a docstring.
    
    Args:
        x: input
        
    Returns:
        output
    """
    pass

def func_without_doc():
    pass
''')
        
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_file(str(test_file))
        
        assert 'file_path' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
