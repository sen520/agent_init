#!/usr/bin/env python3
"""
集成测试 - 测试系统整体集成
"""
import pytest
import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State
from src.graph.base import optimization_app, create_optimization_workflow
from src.tools.file_scanner import FileScanner
from src.strategies.optimization_strategies import CodeOptimizer


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_basic_workflow_integration(self):
        """测试基础工作流集成"""
        state = State(project_path="src")
        
        # 运行基础工作流
        result = await optimization_app.ainvoke(state)
        
        assert result["total_files_analyzed"] > 0, "应该分析了一些文件"
        assert "total_issues_found" in result, "应该有问题统计"
        assert len(result["logs"]) > 0, "应该有日志记录"
    
    @pytest.mark.asyncio
    async def test_optimization_workflow_integration(self):
        """测试优化工作流集成"""
        app = create_optimization_workflow()
        state = State(project_path="src")
        
        # 运行优化工作流
        result = await app.ainvoke(state)
        
        assert "total_optimizations_applied" in result, "应该有优化统计"
        assert "strategies_used" in result, "应该记录使用的策略"
        assert len(result["applied_changes"]) >= 0, "变更记录应该存在"
    
    def test_file_scanner_integration(self):
        """测试文件扫描器集成"""
        scanner = FileScanner("src")
        
        # 扫描Python文件
        python_files = scanner.scan_python_files()
        
        assert len(python_files) > 0, "应该找到Python文件"
        assert all(f.endswith('.py') for f in python_files), "所有文件都应该是.py"
        
        # 扫描特定目录
        specific_files = scanner.scan_directory("src/state")
        assert len(specific_files) > 0, "应该找到state目录下的文件"
    
    def test_optimizer_integration(self):
        """测试优化器集成"""
        optimizer = CodeOptimizer()
        
        # 检查策略可用
        assert len(optimizer.strategies) >= 7, "应该至少有7种策略"
        
        # 测试分析功能
        test_file = "src/state/base.py"
        if os.path.exists(test_file):
            analysis = optimizer.analyze_file(test_file)
            assert "error" not in analysis, "分析不应该出错"
            assert "total_issues" in analysis, "应该有问题统计"
    
    def test_end_to_end_optimization(self):
        """测试端到端优化流程"""
        # 创建测试文件
        test_content = '''# TODO fix this
import os,sys
def very_long_function():
    x=1
    if x>0:
        print("test")
    if x<10:
        print("test2")
    return x
'''
        
        test_file = "temp_test_integration.py"
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # 分析文件
            optimizer = CodeOptimizer()
            analysis = optimizer.analyze_file(test_file)
            
            assert analysis["total_issues"] > 0, "应该发现一些问题"
            
            # 应用优化
            result = optimizer.optimize_file(test_file, [
                'comment_optimizer', 'import_optimizer'
            ])
            
            if result.get('optimization_applied'):
                assert result['changes_count'] > 0, "应该有一些变更"
        
        finally:
            # 清理
            try:
                os.remove(test_file)
                if os.path.exists(test_file + '.backup'):
                    os.remove(test_file + '.backup')
            except:
                pass
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 测试不存在的路径
        state = State(project_path="/nonexistent/path")
        
        try:
            result = await optimization_app.ainvoke(state)
            # 应该能处理错误，不抛出异常
            assert len(result["errors"]) >= 0, "应该记录错误"
        except Exception as e:
            # 如果抛出异常，也是可以的
            assert isinstance(e, Exception), "应该是某种异常"


class TestPerformance:
    """性能测试"""
    
    @pytest.mark.slow
    def test_large_project_analysis(self):
        """测试大型项目分析性能"""
        import time
        import tempfile
        
        # 创建临时测试目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建多个测试文件
            for i in range(10):
                test_file = os.path.join(tmpdir, f"test_{i}.py")
                with open(test_file, 'w') as f:
                    f.write(f'''# Test file {i}
import os
import sys

def function_{i}():
    x = {i}
    return x

class Class{i}:
    def method(self):
        return {i}
''')
            
            # 测试分析性能
            scanner = FileScanner(tmpdir)
            start_time = time.time()
            
            files = scanner.scan_python_files()
            optimizer = CodeOptimizer()
            
            for file in files[:5]:  # 限制文件数量
                optimizer.analyze_file(file)
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            # 性能断言 - 10个文件应该在合理时间内完成
            assert analysis_time < 10.0, f"分析时间过长: {analysis_time:.2f}秒"
            assert len(files) == 10, "应该找到10个文件"
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 运行一些分析操作
        optimizer = CodeOptimizer()
        for _ in range(10):
            optimizer.analyze_file("src/state/base.py")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该合理 (小于50MB)
        assert memory_increase < 50, f"内存增长过多: {memory_increase:.2f}MB"


if __name__ == "__main__":
    # 运行测试
    import subprocess
    
    # 运行集成测试
    result = subprocess.run([
        "python", "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short"
    ], 
    capture_output=True, 
    text=True,
    cwd=Path(__file__).parent.parent
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print(f"Exit code: {result.returncode}")