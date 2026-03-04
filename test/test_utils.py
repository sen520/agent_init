#!/usr/bin/env python3
"""
测试 Utils 工具函数 - src/utils/utils.py
"""
import pytest
import zipfile
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.utils import unzip_file


class TestUnzipFile:
    """测试 unzip_file 函数"""
    
    def test_unzip_valid_zip(self, tmp_path):
        """测试解压有效的 ZIP 文件"""
        # 创建 ZIP 文件
        zip_path = tmp_path / "test.zip"
        extract_dir = tmp_path / "extracted"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", "Hello, World!")
            zf.writestr("subdir/nested.txt", "Nested content")
        
        # 解压
        unzip_file(str(zip_path), str(extract_dir))
        
        # 验证文件被解压
        assert (extract_dir / "test.txt").exists()
        assert (extract_dir / "test.txt").read_text() == "Hello, World!"
        assert (extract_dir / "subdir" / "nested.txt").exists()
    
    def test_unzip_creates_directory(self, tmp_path):
        """测试解压时自动创建目录"""
        zip_path = tmp_path / "test.zip"
        extract_dir = tmp_path / "new_directory" / "nested"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        # 目录不存在时解压
        assert not extract_dir.exists()
        unzip_file(str(zip_path), str(extract_dir))
        
        assert extract_dir.exists()
        assert (extract_dir / "file.txt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
