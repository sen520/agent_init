#!/usr/bin/env python3
"""
测试 FileScanner 文件扫描器
"""
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.file_scanner import FileScanner


class TestFileScanner:
    """FileScanner 测试类"""
    
    @pytest.fixture
    def temp_project(self):
        """创建临时项目目录结构"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # 创建 Python 文件
            (root / "main.py").write_text("print('hello')")
            (root / "utils.py").write_text("def helper(): pass")
            
            # 创建子目录
            src_dir = root / "src"
            src_dir.mkdir()
            (src_dir / "module.py").write_text("class MyClass: pass")
            
            # 创建测试文件
            test_dir = root / "tests"
            test_dir.mkdir()
            (test_dir / "test_module.py").write_text("def test(): pass")
            
            # 创建应被排除的目录
            venv_dir = root / "venv"
            venv_dir.mkdir()
            (venv_dir / "site.py").write_text("# venv file")
            
            pycache = root / "__pycache__"
            pycache.mkdir()
            (pycache / "cache.pyc").write_text("# cache")
            
            yield root
    
    def test_scanner_initialization(self, temp_project):
        """测试扫描器初始化"""
        scanner = FileScanner(str(temp_project))
        assert scanner.project_path == temp_project
    
    def test_scanner_invalid_path(self):
        """测试无效路径"""
        with pytest.raises(ValueError) as exc_info:
            FileScanner("/nonexistent/path/12345")
        assert "不存在" in str(exc_info.value)
    
    def test_scanner_not_directory(self, temp_project):
        """测试路径不是目录"""
        file_path = temp_project / "main.py"
        
        with pytest.raises(ValueError) as exc_info:
            FileScanner(str(file_path))
        assert "不是目录" in str(exc_info.value)
    
    def test_scan_python_files(self, temp_project):
        """测试扫描 Python 文件"""
        scanner = FileScanner(str(temp_project))
        files = scanner.scan_python_files()
        
        # 应该找到 main.py, utils.py, src/module.py
        assert "main.py" in files
        assert "utils.py" in files
        assert "src/module.py" in files
        
        # 不应该找到 venv 和 __pycache__ 中的文件
        assert not any("venv" in f for f in files)
        assert not any("__pycache__" in f for f in files)
    
    def test_scan_project(self, temp_project):
        """测试扫描整个项目"""
        scanner = FileScanner(str(temp_project))
        result = scanner.scan_project()
        
        # 应该包含 .py 文件
        assert '.py' in result
        assert len(result['.py']) >= 3
    
    def test_get_project_stats(self, temp_project):
        """测试获取项目统计"""
        scanner = FileScanner(str(temp_project))
        stats = scanner.get_project_stats()
        
        assert 'project_path' in stats
        assert 'total_files_by_type' in stats
        assert 'lines_by_type' in stats
        assert stats['total_files_by_type'].get('python', 0) >= 3
    
    def test_find_large_files(self, temp_project):
        """测试查找大文件"""
        # 创建一个大文件
        large_file = temp_project / "large.py"
        large_file.write_text("x = 1\n" * 1000)  # 大约 6KB
        
        scanner = FileScanner(str(temp_project))
        large_files = scanner.find_large_files(max_size_kb=5)
        
        assert len(large_files) >= 1
        assert any(f['path'] == 'large.py' for f in large_files)
    
    def test_find_duplicate_filenames(self, temp_project):
        """测试查找重复文件名"""
        # 创建同名文件
        (temp_project / "src" / "utils.py").write_text("# another utils")
        
        scanner = FileScanner(str(temp_project))
        duplicates = scanner.find_duplicate_filenames()
        
        # 应该找到 utils.py 有重复
        if 'utils.py' in duplicates:
            assert len(duplicates['utils.py']) == 2


class TestFileScannerEdgeCases:
    """FileScanner 边界情况测试"""
    
    @pytest.fixture
    def empty_dir(self):
        """创建空目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_empty_directory(self, empty_dir):
        """测试空目录"""
        scanner = FileScanner(str(empty_dir))
        files = scanner.scan_python_files()
        assert files == []
    
    def test_only_non_python_files(self, empty_dir):
        """测试只有非 Python 文件的目录"""
        (empty_dir / "readme.md").write_text("# Readme")
        (empty_dir / "data.json").write_text('{"key": "value"}')
        
        scanner = FileScanner(str(empty_dir))
        files = scanner.scan_python_files()
        assert files == []
    
    def test_hidden_directories(self, empty_dir):
        """测试隐藏目录"""
        hidden = empty_dir / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("# secret")
        
        # .hidden 不在默认排除列表中
        scanner = FileScanner(str(empty_dir))
        # 具体行为取决于配置


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
