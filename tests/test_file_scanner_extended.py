#!/usr/bin/env python3
"""
测试 file_scanner 模块 - 补充测试
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFileScannerExtended:
    """FileScanner 扩展测试"""

    def test_file_scanner_init(self):
        """测试 FileScanner 初始化"""
        from src.tools.file_scanner import FileScanner

        scanner = FileScanner(str(Path(__file__).parent))
        assert scanner is not None

    def test_scan_python_files(self):
        """测试扫描 Python 文件"""
        from src.tools.file_scanner import FileScanner

        scanner = FileScanner(str(Path(__file__).parent))
        files = scanner.scan_python_files()
        assert isinstance(files, list)

    def test_get_project_stats(self):
        """测试获取项目统计"""
        from src.tools.file_scanner import FileScanner

        scanner = FileScanner(str(Path(__file__).parent))
        stats = scanner.get_project_stats()
        assert 'total_lines' in stats
        assert 'total_files_by_type' in stats

    def test_find_large_files(self):
        """测试查找大文件"""
        from src.tools.file_scanner import FileScanner

        scanner = FileScanner(str(Path(__file__).parent))
        large_files = scanner.find_large_files(max_size_kb=1)
        assert isinstance(large_files, list)

    def test_find_duplicate_filenames(self):
        """测试查找重复文件名"""
        from src.tools.file_scanner import FileScanner

        scanner = FileScanner(str(Path(__file__).parent))
        duplicates = scanner.find_duplicate_filenames()
        assert isinstance(duplicates, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
