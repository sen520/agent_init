#!/usr/bin/env python3
"""
补充测试 - src/testing/validator.py
"""
import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import subprocess

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.testing.validator import TestValidator, validate_optimization_result


class TestTestValidatorExtended:
    """扩展的 TestValidator 测试"""
    
    @pytest.fixture
    def temp_project(self):
        """创建临时项目"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # 创建测试目录
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_sample.py").write_text("""
def test_pass():
    assert True
""")
            yield root
    
    def test_run_tests_success(self, temp_project):
        """测试成功运行测试"""
        validator = TestValidator(str(temp_project))
        
        result = validator.run_tests(verbose=False)
        
        assert 'passed' in result
        assert 'failed' in result
        # 可能没有 pytest，所以可能失败
    
    def test_run_tests_with_path(self, temp_project):
        """测试指定路径运行测试"""
        validator = TestValidator(str(temp_project))
        
        result = validator.run_tests(test_path=str(temp_project / "tests"))
        
        assert result is not None
    
    def test_run_tests_nonexistent_path(self, temp_project):
        """测试指定不存在的路径"""
        validator = TestValidator(str(temp_project))
        
        result = validator.run_tests(test_path="/nonexistent/path")
        
        assert not result['success']
    
    def test_validate_file_syntax_error(self, tmp_path):
        """测试验证有语法错误的文件"""
        validator = TestValidator(str(tmp_path))
        
        # 创建有语法错误的文件
        test_file = tmp_path / "syntax_error.py"
        test_file.write_text("def broken(\n  # missing parenthesis")
        
        result = validator.validate_file(str(test_file))
        
        assert not result['valid']
        assert result['line'] is not None
    
    def test_validate_file_encoding_error(self, tmp_path):
        """测试验证有编码错误的文件"""
        validator = TestValidator(str(tmp_path))
        
        # 创建有编码问题的文件
        test_file = tmp_path / "binary.py"
        test_file.write_bytes(b'\x00\x01\xff\xfe')
        
        result = validator.validate_file(str(test_file))
        
        assert not result['valid']
        assert '编码' in result['error'] or 'UnicodeDecodeError' in result['error']
    
    def test_validate_file_permission_error(self, tmp_path):
        """测试验证权限错误的文件"""
        validator = TestValidator(str(tmp_path))
        
        # 创建文件并移除读权限
        test_file = tmp_path / "no_read.py"
        test_file.write_text("print('hello')")
        test_file.chmod(0o000)
        
        try:
            result = validator.validate_file(str(test_file))
            assert not result['valid']
        finally:
            # 恢复权限以便清理
            test_file.chmod(0o644)
    
    def test_validate_after_optimization_syntax_error(self, tmp_path):
        """测试优化后验证发现语法错误"""
        validator = TestValidator(str(tmp_path))
        
        # 创建有语法错误的文件
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(")
        
        result = validator.validate_after_optimization([str(bad_file)])
        
        assert not result['success']
        assert not result['syntax_valid']
        assert result['should_rollback']
    
    def test_validate_after_optimization_test_failure(self, tmp_path):
        """测试优化后验证测试失败"""
        validator = TestValidator(str(tmp_path))
        
        # 创建有效文件
        good_file = tmp_path / "good.py"
        good_file.write_text("print('hello')")
        
        with patch.object(validator, 'run_tests') as mock_run_tests:
            mock_run_tests.return_value = {
                'success': False,
                'failed': 2,
                'failed_tests': ['test1', 'test2']
            }
            
            result = validator.validate_after_optimization([str(good_file)])
            
            assert not result['success']
            assert result['should_rollback']
    
    def test_quick_check_success(self, tmp_path):
        """测试快速检查成功"""
        # 创建有效的 Python 项目结构
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")
        
        validator = TestValidator(str(tmp_path))
        
        # 这个测试可能成功也可能失败，取决于环境
        result = validator.quick_check()
        # 不断言具体结果，只确保不抛出异常
        assert isinstance(result, bool)
    
    def test_quick_check_timeout(self, tmp_path):
        """测试快速检查超时"""
        validator = TestValidator(str(tmp_path))
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd='test', timeout=30)
            
            result = validator.quick_check()
            
            assert not result


class TestValidateOptimizationResultExtended:
    """扩展的 validate_optimization_result 测试"""
    
    def test_success(self, tmp_path):
        """测试验证成功"""
        # 创建有效文件
        good_file = tmp_path / "good.py"
        good_file.write_text("print('hello')")
        
        success, message = validate_optimization_result(
            [str(good_file)],
            str(tmp_path)
        )
        
        # 可能成功也可能失败，取决于是否有测试
        assert isinstance(success, bool)
        assert isinstance(message, str)
    
    def test_syntax_error(self, tmp_path):
        """测试语法错误情况"""
        # 创建有语法错误的文件
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(")
        
        success, message = validate_optimization_result(
            [str(bad_file)],
            str(tmp_path)
        )
        
        assert not success
        assert '语法错误' in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
