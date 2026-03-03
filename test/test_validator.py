#!/usr/bin/env python3
"""
测试 TestValidator 测试验证器
"""
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.testing.validator import TestValidator, validate_optimization_result


class TestTestValidator:
    """TestValidator 测试类"""
    
    @pytest.fixture
    def temp_project(self):
        """创建临时项目"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def valid_python_file(self, temp_project):
        """创建有效的 Python 文件"""
        file_path = temp_project / "valid.py"
        file_path.write_text("""
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    print(hello())
""")
        return str(file_path)
    
    @pytest.fixture
    def invalid_python_file(self, temp_project):
        """创建无效的 Python 文件"""
        file_path = temp_project / "invalid.py"
        file_path.write_text("""
def broken(
    # 缺少右括号
    print("broken")
""")
        return str(file_path)
    
    def test_validator_initialization(self, temp_project):
        """测试验证器初始化"""
        validator = TestValidator(str(temp_project))
        assert validator.project_path == temp_project
    
    def test_validate_valid_file(self, temp_project, valid_python_file):
        """测试验证有效的 Python 文件"""
        validator = TestValidator(str(temp_project))
        result = validator.validate_file(valid_python_file)
        
        assert result['valid']
        assert result['file_path'] == valid_python_file
        assert result['error'] is None
    
    def test_validate_invalid_file(self, temp_project, invalid_python_file):
        """测试验证无效的 Python 文件"""
        validator = TestValidator(str(temp_project))
        result = validator.validate_file(invalid_python_file)
        
        assert not result['valid']
        assert '语法错误' in result['error']
        assert result['line'] is not None
    
    def test_validate_nonexistent_file(self, temp_project):
        """测试验证不存在的文件"""
        validator = TestValidator(str(temp_project))
        result = validator.validate_file("/nonexistent/file.py")
        
        assert not result['valid']
        assert '不存在' in result['error']
    
    def test_validate_empty_file(self, temp_project):
        """测试验证空文件"""
        empty_file = temp_project / "empty.py"
        empty_file.write_text("")
        
        validator = TestValidator(str(temp_project))
        result = validator.validate_file(str(empty_file))
        
        assert not result['valid']
        assert '为空' in result['error']
    
    def test_parse_pytest_output(self, temp_project):
        """测试解析 pytest 输出"""
        validator = TestValidator(str(temp_project))
        
        sample_output = """
============================= test session starts ==============================
platform linux -- Python 3.12.0
 collected 5 items

test_example.py::test_passed PASSED                                      [ 20%]
test_example.py::test_failed FAILED                                      [ 40%]
test_example.py::test_error ERROR                                        [ 60%]
test_example.py::test_skipped SKIPPED                                    [ 80%]
test_example.py::test_passed2 PASSED                                     [100%]

=========================== short test summary info ============================
FAILED test_example.py::test_failed - assert 0
ERROR test_example.py::test_error - NameError
========================= 2 passed, 1 failed, 1 error, 1 skipped in 0.5s =========================
"""
        result = validator._parse_pytest_output(sample_output)
        
        assert result['passed'] == 2
        assert result['failed'] == 1
        assert result['errors'] == 1
        assert result['skipped'] == 1
        assert result['total'] == 5
        assert len(result['failed_tests']) == 2
    
    def test_validate_after_optimization_empty(self, temp_project):
        """测试优化后验证（空文件列表）"""
        validator = TestValidator(str(temp_project))
        result = validator.validate_after_optimization([])
        
        assert result['success']
        assert result['file_results'] == []
    
    def test_quick_check_no_src(self, temp_project):
        """测试快速检查（无 src 模块）"""
        validator = TestValidator(str(temp_project))
        # 临时项目中没有 src 模块，应该失败
        result = validator.quick_check()
        assert not result


class TestValidateOptimizationResult:
    """测试便捷函数"""
    
    @pytest.fixture
    def temp_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_no_modified_files(self, temp_project):
        """测试没有修改文件的情况"""
        success, message = validate_optimization_result([], str(temp_project))
        assert success
        assert "验证通过" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
