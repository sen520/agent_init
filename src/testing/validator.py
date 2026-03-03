#!/usr/bin/env python3
"""
测试验证器 - 使用 pytest 验证代码优化结果
"""
import subprocess
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.config.manager import get_config

logger = logging.getLogger(__name__)


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
        
        # 加载配置
        config = get_config()
        self.timeout = config.get('testing.timeout', 300)  # 默认5分钟
        
        logger.info(f"测试验证器初始化: {self.project_path}")
    
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
            test_path_obj = Path(test_path)
            if not test_path_obj.exists():
                logger.error(f"测试路径不存在: {test_path}")
                result["output"] = f"测试路径不存在: {test_path}"
                return result
            cmd.append(str(test_path))
        else:
            # 默认测试目录
            tests_dir = self.project_path / "tests"
            if tests_dir.exists():
                cmd.append(str(tests_dir))
            else:
                logger.warning(f"未找到测试目录: {tests_dir}")
                result["output"] = "未找到测试目录"
                return result
        
        try:
            logger.info(f"运行测试: {' '.join(cmd)}")
            # 运行测试
            process = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # 解析输出
            output = process.stdout + process.stderr
            result["output"] = output
            result["returncode"] = process.returncode
            
            # 解析 pytest 输出
            result.update(self._parse_pytest_output(output))
            
            # 确定是否成功
            result["success"] = process.returncode == 0
            
            if result["success"]:
                logger.info(f"测试通过: {result['passed']} 个")
            else:
                logger.warning(f"测试失败: {result['failed']} 失败, {result['errors']} 错误")
            
        except subprocess.TimeoutExpired:
            logger.error(f"测试运行超时（超过 {self.timeout} 秒）")
            result["output"] = f"测试运行超时（超过 {self.timeout} 秒）"
            result["errors"] += 1
        except FileNotFoundError:
            logger.error("pytest 未安装")
            result["output"] = "pytest 未安装，请运行: pip install pytest"
            result["errors"] += 1
        except PermissionError as e:
            logger.error(f"权限错误: {e}")
            result["output"] = f"权限错误: {e}"
            result["errors"] += 1
        except Exception as e:
            logger.exception("运行测试时出错")
            result["output"] = f"运行测试时出错: {type(e).__name__}: {e}"
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
                    except (ValueError, IndexError):
                        pass
                if 'failed' in line:
                    try:
                        result["failed"] = int(line.split('failed')[0].split()[-1])
                    except (ValueError, IndexError):
                        pass
                if 'error' in line:
                    try:
                        result["errors"] = int(line.split('error')[0].split()[-1])
                    except (ValueError, IndexError):
                        pass
                if 'skipped' in line:
                    try:
                        result["skipped"] = int(line.split('skipped')[0].split()[-1])
                    except (ValueError, IndexError):
                        pass
                
                # 提取时间
                if 'in' in line and 's' in line:
                    try:
                        time_part = line.split('in')[1].strip()
                        result["duration"] = float(time_part.replace('s', ''))
                    except (ValueError, IndexError):
                        pass
            
            # 收集失败的测试名
            if line.startswith('FAILED ') or line.startswith('ERROR '):
                parts = line.split()
                if len(parts) > 1:
                    test_name = parts[1]
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
            "error": None,
            "line": None,
            "column": None
        }
        
        # 检查文件是否存在
        path = Path(file_path)
        if not path.exists():
            result["error"] = f"文件不存在: {file_path}"
            logger.error(result["error"])
            return result
        
        if not path.is_file():
            result["error"] = f"路径不是文件: {file_path}"
            logger.error(result["error"])
            return result
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # 检查是否为空文件
            if not source.strip():
                result["error"] = "文件为空"
                logger.warning(f"{file_path} 为空文件")
                return result
            
            # 编译检查语法
            compile(source, file_path, 'exec')
            result["valid"] = True
            logger.debug(f"{file_path} 语法验证通过")
            
        except SyntaxError as e:
            result["error"] = f"语法错误: {e}"
            result["line"] = e.lineno
            result["column"] = e.offset
            logger.error(f"{file_path} 语法错误: 行 {e.lineno}, 列 {e.offset}")
        except UnicodeDecodeError as e:
            result["error"] = f"文件编码错误: {e}"
            logger.error(f"{file_path} 编码错误: {e}")
        except PermissionError as e:
            result["error"] = f"权限错误，无法读取文件: {e}"
            logger.error(f"{file_path} 权限错误: {e}")
        except OSError as e:
            result["error"] = f"IO 错误: {e}"
            logger.error(f"{file_path} IO 错误: {e}")
        except Exception as e:
            result["error"] = f"验证失败: {type(e).__name__}: {e}"
            logger.exception(f"{file_path} 验证时发生未知错误")
        
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
            "should_rollback": False,
            "errors": []
        }
        
        if not modified_files:
            logger.info("没有文件被修改，跳过验证")
            result["success"] = True
            return result
        
        logger.info(f"开始验证 {len(modified_files)} 个修改的文件")
        
        # 1. 验证所有修改文件的语法
        for file_path in modified_files:
            validation = self.validate_file(file_path)
            result["file_results"].append(validation)
            
            if not validation["valid"]:
                result["syntax_valid"] = False
                result["success"] = False
                result["errors"].append(f"语法错误: {validation['file_path']}")
                logger.error(f"语法验证失败: {Path(file_path).name} - {validation['error']}")
            else:
                logger.info(f"语法验证通过: {Path(file_path).name}")
        
        # 2. 如果语法都正确，运行测试
        if result["syntax_valid"]:
            logger.info("语法验证通过，运行测试...")
            test_result = self.run_tests()
            result["test_results"] = test_result
            
            if not test_result["success"]:
                result["tests_passed"] = False
                result["success"] = False
                result["should_rollback"] = True
                result["errors"].append(f"测试失败: {test_result['failed']} 个")
                logger.error(f"测试失败: {test_result['failed']} 个失败")
                for failed in test_result.get("failed_tests", [])[:5]:
                    logger.error(f"  失败测试: {failed}")
            else:
                logger.info(f"测试通过: {test_result['passed']} 个")
        else:
            result["should_rollback"] = True
            logger.warning("语法验证失败，建议回滚")
        
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
            if result.returncode == 0:
                logger.info("快速检查通过")
                return True
            else:
                logger.warning(f"快速检查失败: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("快速检查超时")
            return False
        except FileNotFoundError:
            logger.error("Python 解释器未找到")
            return False
        except Exception as e:
            logger.exception("快速检查出错")
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
        errors = "; ".join(result["errors"])
        return False, f"❌ 存在语法错误: {errors}"
    elif not result["tests_passed"]:
        return False, f"❌ 测试失败: {result['test_results']['failed']} 个"
    else:
        return False, f"❌ 验证失败: {'; '.join(result['errors'])}"


if __name__ == "__main__":
    # 测试验证器
    logging.basicConfig(level=logging.INFO)
    print("🧪 测试验证器测试")
    print("=" * 50)
    
    validator = TestValidator()
    
    # 测试语法验证
    test_file = __file__
    result = validator.validate_file(test_file)
    print(f"语法验证: {'✅ 通过' if result['valid'] else '❌ 失败'}")
    if not result['valid']:
        print(f"  错误: {result['error']}")
    
    # 测试 pytest
    print("\n运行测试...")
    test_result = validator.run_tests(verbose=False)
    print(f"测试结果: {test_result['passed']} passed, {test_result['failed']} failed")
    if test_result['failed'] > 0:
        print(f"  失败测试: {test_result['failed_tests']}")
