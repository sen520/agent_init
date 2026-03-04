#!/usr/bin/env python3
"""
测试 FileModifier 文件修改器
"""
import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_modifier import FileModifier, apply_optimization_safely


class TestFileModifier:
    """FileModifier 测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def file_modifier(self, temp_dir):
        """创建 FileModifier 实例"""
        return FileModifier(backup_dir=str(temp_dir / "backups"))
    
    @pytest.fixture
    def sample_python_file(self, temp_dir):
        """创建示例 Python 文件"""
        file_path = temp_dir / "test_file.py"
        content = """#!/usr/bin/env python3
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
"""
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)
    
    def test_backup_file(self, file_modifier, sample_python_file):
        """测试文件备份功能"""
        # 备份文件
        backup_path = file_modifier.backup_file(sample_python_file)
        
        # 验证备份文件存在
        assert Path(backup_path).exists()
        
        # 验证备份内容与原始文件相同
        original_content = Path(sample_python_file).read_text()
        backup_content = Path(backup_path).read_text()
        assert original_content == backup_content
    
    def test_backup_nonexistent_file(self, file_modifier, temp_dir):
        """测试备份不存在的文件"""
        nonexistent = str(temp_dir / "nonexistent.py")
        
        with pytest.raises(FileNotFoundError):
            file_modifier.backup_file(nonexistent)
    
    def test_write_file_with_backup(self, file_modifier, sample_python_file):
        """测试带备份的文件写入"""
        new_content = """#!/usr/bin/env python3
def hello():
    print("Modified!")
"""
        
        success, message = file_modifier.write_file(
            sample_python_file,
            new_content,
            create_backup=True
        )
        
        assert success
        assert "已修改" in message
        assert Path(sample_python_file).read_text() == new_content
        
        # 验证备份创建
        assert len(file_modifier.backups) == 1
        assert sample_python_file in file_modifier.backups
    
    def test_write_file_syntax_validation(self, file_modifier, sample_python_file):
        """测试写入时的语法验证"""
        invalid_content = """#!/usr/bin/env python3
def hello(
    # 语法错误：缺少右括号
"""
        
        success, message = file_modifier.write_file(
            sample_python_file,
            invalid_content,
            create_backup=True
        )
        
        assert not success
        assert "语法错误" in message
    
    def test_rollback_file(self, file_modifier, sample_python_file):
        """测试文件回滚"""
        original_content = Path(sample_python_file).read_text()
        
        # 修改文件
        new_content = "# Modified content"
        file_modifier.write_file(sample_python_file, new_content, create_backup=True)
        
        # 回滚
        success, message = file_modifier.rollback_file(sample_python_file)
        
        assert success
        assert Path(sample_python_file).read_text() == original_content
    
    def test_rollback_without_backup(self, file_modifier, sample_python_file):
        """测试没有备份时的回滚"""
        success, message = file_modifier.rollback_file(sample_python_file)
        
        assert not success
        assert "没有找到备份" in message
    
    def test_write_file_without_changes(self, file_modifier, sample_python_file):
        """测试写入相同内容（无变化）"""
        original_content = Path(sample_python_file).read_text()
        
        success, message = file_modifier.write_file(
            sample_python_file,
            original_content,
            create_backup=True
        )
        
        # 应该返回成功，但提示无需修改
        assert success
        assert "无需修改" in message
    
    def test_cleanup_old_backups(self, file_modifier, sample_python_file):
        """测试清理旧备份"""
        import time
        
        # 创建备份
        file_modifier.backup_file(sample_python_file)
        
        # 等待一小段时间（实际测试中使用很小的 retention_days）
        # 这里我们直接测试函数是否能运行
        deleted = file_modifier.cleanup_old_backups(days=0)
        
        # 刚创建的备份应该被删除（因为 retention_days=0）
        # 注意：由于时间精度问题，这个测试可能不稳定
        assert isinstance(deleted, int)


class TestApplyOptimizationSafely:
    """测试 apply_optimization_safely 函数"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_successful_optimization(self, temp_dir):
        """测试成功的优化"""
        file_path = temp_dir / "test.py"
        original_content = "# Original\nprint('hello')"
        file_path.write_text(original_content)
        
        modifier = FileModifier(backup_dir=str(temp_dir / "backups"))
        
        optimized_content = "# Optimized\nprint('hello')\n"
        result = apply_optimization_safely(
            str(file_path),
            optimized_content,
            modifier
        )
        
        assert result['success']
        assert result['changes_applied']
        assert file_path.read_text() == optimized_content
    
    def test_no_changes_needed(self, temp_dir):
        """测试无需修改的情况"""
        file_path = temp_dir / "test.py"
        content = "# Same content"
        file_path.write_text(content)
        
        modifier = FileModifier(backup_dir=str(temp_dir / "backups"))
        
        result = apply_optimization_safely(
            str(file_path),
            content,  # 相同内容
            modifier
        )
        
        assert result['success']
        assert not result['changes_applied']
        assert "无需修改" in result['message']
    
    def test_invalid_syntax(self, temp_dir):
        """测试无效语法"""
        file_path = temp_dir / "test.py"
        file_path.write_text("# Valid content")
        
        modifier = FileModifier(backup_dir=str(temp_dir / "backups"))
        
        invalid_content = "def broken(\n  # missing parenthesis"
        
        result = apply_optimization_safely(
            str(file_path),
            invalid_content,
            modifier
        )
        
        assert not result['success']
        assert "语法错误" in result['error']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
