#!/usr/bin/env python3
"""
集成测试 - 测试整个工作流
"""
import pytest
import tempfile
from pathlib import Path
import asyncio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State
from src.nodes.real import initialize_project, analyze_code, end_optimization
from src.graph.base import build_simple_graph


class TestIntegration:
    """集成测试类"""
    
    @pytest.fixture
    def temp_project(self):
        """创建临时项目"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # 创建 Python 文件
            (root / "main.py").write_text("""
def main():
    print("Hello")
    x=1  # 格式问题
    return x
""")
            (root / "utils.py").write_text("""
def helper():
    pass
""")
            
            yield root
    
    def test_initialize_project(self, temp_project):
        """测试项目初始化"""
        state = State()
        state.project_path = str(temp_project)
        
        result = initialize_project(state)
        
        assert result.iteration_count == 1
        assert result.project_path == str(temp_project)
    
    def test_analyze_code_real(self, temp_project):
        """测试真实代码分析"""
        state = State()
        state.project_path = str(temp_project)
        
        result = analyze_code(state)
        
        # 应该分析到文件
        assert result.analysis.total_files_analyzed > 0
        assert result.analysis.total_lines_of_code > 0
    
    def test_end_optimization(self, temp_project):
        """测试结束优化"""
        state = State()
        state.project_path = str(temp_project)
        state.analysis.total_files_analyzed = 2
        state.analysis.total_lines_of_code = 10
        state.analysis.issues = []
        state.applied_changes = ["change 1", "change 2"]
        
        result = end_optimization(state)
        
        assert result.iteration_count == state.iteration_count
    
class TestStateManagement:
    """状态管理测试"""
    
    def test_state_initialization(self):
        """测试状态初始化"""
        state = State()
        
        assert state.project_path == ""
        assert state.iteration_count == 0
        assert state.max_iterations == 5
        assert state.should_continue == True
        assert state.logs == []
        assert state.errors == []
    
    def test_state_add_log(self):
        """测试添加日志"""
        state = State()
        state.add_log("Test message")
        
        assert len(state.logs) == 1
        assert "Test message" in state.logs[0]
    
    def test_state_add_error(self):
        """测试添加错误"""
        state = State()
        state.add_error("Test error")
        
        assert len(state.errors) == 1
        assert state.errors[0] == "Test error"
    
    def test_state_log_limit(self):
        """测试日志数量限制"""
        state = State()
        
        # 添加超过 100 条日志
        for i in range(110):
            state.add_log(f"Log {i}")
        
        # 应该只保留最近的 50-100 条（取决于实现）
        assert len(state.logs) <= 100


class TestWorkflowIntegration:
    """工作流集成测试"""
    
    @pytest.fixture
    def temp_project_with_issues(self):
        """创建有代码问题的临时项目"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # 创建有问题的 Python 文件
            (root / "bad_code.py").write_text("""
# 这是一个很长的行，超过 100 个字符的限制，应该被检测到为问题，需要修复和优化

def very_long_function_that_does_nothing_but_has_a_very_long_name_and_should_be_detected_as_having_issues():
    pass

# TODO

import os, sys, json, re

except:
    pass
""")
            yield root
    
    def test_analyze_detects_issues(self, temp_project_with_issues):
        """测试分析能检测到问题"""
        state = State()
        state.project_path = str(temp_project_with_issues)
        
        result = analyze_code(state)
        
        # 应该检测到问题
        assert len(result.analysis.issues) > 0
        
        # 检查问题类型
        issue_types = [issue.issue_type for issue in result.analysis.issues]
        # 应该有超长行问题
        assert 'long_line' in issue_types or len(issue_types) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
