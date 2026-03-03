#!/usr/bin/env python3
"""
测试验证器 - 使用 pytest 验证代码优化结果
"""
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.config.manager import get_config


class TestValidator:
    """测试验证器"""
    
    def __init__(self, project_path: str = None):
        """
        初始化测试验证器
        
        Args:
            project_path: 项目路径（默认为当前目录）
        """
        self.project_path = Path(project_path or os.getcwd())
        self.test_results = []
    
    def run_tests(self, test_path: str = None, verbose: bool = True) -> Dict[str, Any]:
        """
        运行 pytest 测试
        
        Args:
            test_path: 测试文件或目录路径
            verbose: 是否显示详细输出
            
        Returns:
            测试结果字典
        """
        result = {
            "success": False,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "total": 0,
            "duration": 0.0,
            "output": "",
            "failed_tests": []
        }
        
        # 构建 pytest 命令
        cmd = [sys.executable, "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend(["--tb=short", "--no-header"])
        
        # 添加测试路径
        if test_path:
            cmd.append(str(test_path))
        else:
            # 默认测试目录
            tests_dir = self.project_path / "tests"
            if tests_dir.exists():
                cmd.append(str(tests_dir))
            else:
                result["output"] = "未找到测试目录"
                return result
        
        try:
            # 运行测试
            process = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            # 解析输出
            output = process.stdout + process.stderr
            result["output"] = output
            result["returncode"] = process.returncode
            
            # 解析 pytest 输出
            result.update(self._parse_pytest_output(output))
            
            # 确定是否成功
            result["success"] = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            result["output"] = "测试运行超时（超过5分钟）"
            result["errors"] += 1
        except Exception as e:
            result["output"] = f"运行测试时出错: {e}"
            result["errors"] += 1
        
        return result
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """解析 pytest 输出"""
        result = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "total": 0,
            "duration": 0.0,
            "failed_tests": []
        }
        
        lines = output.split('\n')
        
        for line in lines:
            # 解析摘要行 (例如: "5 passed, 2 failed in 0.5s")
            if 'passed' in line or 'failed' in line or 'error' in line:
                # 提取数量
                if 'passed' in line:
                    try:
                        result["passed"] = int(line.split('passed')[0].split()[-1])
                    except:
                        pass
                if 'failed' in line:
                    try:
                        result["failed"] = int(line.split('failed')[0].split()[-1])
                    except:
                        pass
                if 'error' in line:
                    try:
                        result["errors"] = int(line.split('error')[0].split()[-1])
                    except:
                        pass
                if 'skipped' in line:
                    try:
                        result["skipped"] = int(line.split('skipped')[0].split()[-1])
                    except:
                        pass
                
                # 提取时间
                if 'in' in line and 's' in line:
                    try:
                        time_part = line.split('in')[1].strip()
                        result["duration"] = float(time_part.replace('s', ''))
                    except:
                        pass
            
            # 收集失败的测试名
            if line.startswith('FAILED ') or line.startswith('ERROR '):
                test_name = line.split()[1] if len(line.split()) > 1 else line
                result["failed_tests"].append(test_name)
        
        result["total"] = result["passed"] + result["failed"] + result["errors"] + result["skipped"]
        return result
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        验证单个文件的语法正确性
        
        Args:
            file_path: Python 文件路径
            
        Returns:
            验证结果
        """
        result = {
            "valid": False,
            "file_path": file_path,
            "error": None
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # 编译检查语法
            compile(source, file_path, 'exec')
            result["valid"] = True
            
        except SyntaxError as e:
            result["error"] = f"语法错误: {e}"
        except Exception as e:
            result["error"] = f"验证失败: {e}"
        
        return result
    
    def validate_after_optimization(self, modified_files: List[str]) -> Dict[str, Any]:
        """
        优化后验证 - 检查语法并运行相关测试
        
        Args:
            modified_files: 被修改的文件列表
            
        Returns:
            验证结果
        """
        result = {
            "success": True,
            "syntax_valid": True,
            "tests_passed": True,
            "file_results": [],
            "test_results": None,
            "should_rollback": False
        }
        
        print("\n🔍 验证优化结果...")
        
        # 1. 验证所有修改文件的语法
        for file_path in modified_files:
            validation = self.validate_file(file_path)
            result["file_results"].append(validation)
            
            if not validation["valid"]:
                result["syntax_valid"] = False
                result["success"] = False
                print(f"   ❌ 语法错误: {Path(file_path).name}")
                print(f"      {validation['error']}")
            else:
                print(f"   ✅ 语法正确: {Path(file_path).name}")
        
        # 2. 如果语法都正确，运行测试
        if result["syntax_valid"]:
            print("\n🧪 运行测试...")
            test_result = self.run_tests()
            result["test_results"] = test_result
            
            if not test_result["success"]:
                result["tests_passed"] = False
                result["success"] = False
                result["should_rollback"] = True
                print(f"   ❌ 测试失败: {test_result['failed']} 个失败")
                for failed in test_result.get("failed_tests", [])[:5]:
                    print(f"      - {failed}")
            else:
                print(f"   ✅ 测试通过: {test_result['passed']} 个")
        else:
            result["should_rollback"] = True
        
        return result
    
    def quick_check(self) -> bool:
        """
        快速检查 - 验证项目是否能导入
        
        Returns:
            是否通过检查
        """
        try:
            # 尝试导入主模块
            result = subprocess.run(
                [sys.executable, "-c", "import src"],
                cwd=self.project_path,
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except:
            return False


def validate_optimization_result(
    modified_files: List[str],
    project_path: str = None
) -> Tuple[bool, str]:
    """
    便捷函数：验证优化结果
    
    Returns:
        (是否成功, 消息)
    """
    validator = TestValidator(project_path)
    result = validator.validate_after_optimization(modified_files)
    
    if result["success"]:
        return True, "✅ 验证通过"
    elif not result["syntax_valid"]:
        return False, "❌ 存在语法错误"
    elif not result["tests_passed"]:
        return False, f"❌ 测试失败: {result['test_results']['failed']} 个"
    else:
        return False, "❌ 验证失败"


if __name__ == "__main__":
    # 测试验证器
    print("🧪 测试验证器测试")
    print("=" * 50)
    
    validator = TestValidator()
    
    # 测试语法验证
    test_file = __file__
    result = validator.validate_file(test_file)
    print(f"语法验证: {'✅ 通过' if result['valid'] else '❌ 失败'}")
    
    # 测试 pytest
    print("\n运行测试...")
    test_result = validator.run_tests(verbose=False)
    print(f"测试结果: {test_result['passed']} passed, {test_result['failed']} failed")
