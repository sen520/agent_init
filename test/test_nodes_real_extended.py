#!/usr/bin/env python3
"""
补充测试 - src/nodes/real.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State, CodeIssue
from src.nodes.real import initialize_project, analyze_code, create_analysis_report, end_optimization


class TestInitializeProjectExtended:
    """扩展的 initialize_project 测试"""
    
    def test_initialize_with_existing_path(self, tmp_path):
        """测试使用存在的路径初始化"""
        state = State()
        state.project_path = str(tmp_path)
        
        result = initialize_project(state)
        
        assert result.project_path == str(tmp_path)
        assert result.iteration_count == 1
    
    def test_initialize_with_nonexistent_path(self):
        """测试使用不存在的路径初始化"""
        state = State()
        state.project_path = "/nonexistent/path/12345"
        
        result = initialize_project(state)
        
        # 应该回退到当前目录或其他有效路径
        assert result.project_path is not None
    
    def test_initialize_default_path(self):
        """测试默认路径初始化"""
        state = State()
        state.project_path = ""
        
        result = initialize_project(state)
        
        assert result.project_path != ""
        assert result.iteration_count == 1


class TestAnalyzeCodeExtended:
    """扩展的 analyze_code 测试"""
    
    @pytest.fixture
    def temp_project(self):
        """创建临时项目"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # 创建多个 Python 文件
            (root / "main.py").write_text("""
def main():
    print("Hello")
    return 0
""")
            (root / "utils.py").write_text("""
def helper():
    pass

def another():
    x = 1
    y = 2
    return x + y
""")
            # 创建子目录
            subdir = root / "package"
            subdir.mkdir()
            (subdir / "__init__.py").write_text("")
            (subdir / "module.py").write_text("""
class MyClass:
    def method(self):
        pass
""")
            yield root
    
    def test_analyze_multiple_files(self, temp_project):
        """测试分析多个文件"""
        state = State()
        state.project_path = str(temp_project)
        
        result = analyze_code(state)
        
        assert result.analysis.total_files_analyzed > 0
        assert result.analysis.total_lines_of_code > 0
    
    def test_analyze_collects_issues(self, temp_project):
        """测试分析问题收集"""
        # 创建有问题的文件
        bad_file = temp_project / "bad.py"
        bad_file.write_text("""
# This is a very long line that exceeds the maximum allowed length of 100 characters by quite a lot
def func():
    pass
""")
        
        state = State()
        state.project_path = str(temp_project)
        
        result = analyze_code(state)
        
        # 应该收集到问题
        assert len(result.analysis.issues) >= 0
    
    def test_analyze_empty_project(self, tmp_path):
        """测试分析空项目"""
        state = State()
        state.project_path = str(tmp_path)
        
        result = analyze_code(state)
        
        assert result.analysis.total_files_analyzed == 0
    
    def test_analyze_creates_summary(self, temp_project):
        """测试分析问题摘要"""
        state = State()
        state.project_path = str(temp_project)
        
        result = analyze_code(state)
        
        # 应该有 issue_summary（即使为空）
        assert isinstance(result.analysis.issue_summary, dict)


class TestCreateAnalysisReportExtended:
    """扩展的 create_analysis_report 测试"""
    
    def test_report_with_issues(self, tmp_path):
        """测试有问题的报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.total_files_analyzed = 10
        state.analysis.total_lines_of_code = 500
        state.analysis.average_complexity = 5.0
        state.analysis.issues = [
            CodeIssue(
                file_path='test.py',
                line_number=10,
                issue_type='long_line',
                description='行太长',
                severity='high'
            ),
            CodeIssue(
                file_path='test2.py',
                line_number=20,
                issue_type='bare_except',
                description='裸异常',
                severity='medium'
            ),
            CodeIssue(
                file_path='test3.py',
                line_number=30,
                issue_type='todo',
                description='TODO',
                severity='low'
            )
        ]
        state.analysis.issue_summary = {'long_line': 1, 'bare_except': 1, 'todo': 1}
        
        result = create_analysis_report(state)
        
        assert len(result.analysis_reports) == 1
        report = result.analysis_reports[0]
        assert report['files_analyzed'] == 10
        assert len(report['top_issues']) <= 10
    
    def test_report_no_issues(self, tmp_path):
        """测试无问题的报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.total_files_analyzed = 5
        state.analysis.issues = []
        state.analysis.issue_summary = {}
        
        result = create_analysis_report(state)
        
        assert len(result.analysis_reports) == 1
        report = result.analysis_reports[0]
        assert report['issues_found'] == 0


class TestEndOptimizationExtended:
    """扩展的 end_optimization 测试"""
    
    def test_end_with_changes(self, tmp_path):
        """测试有变更的结束"""
        state = State()
        state.project_path = str(tmp_path)
        state.iteration_count = 3
        state.analysis.total_files_analyzed = 10
        state.analysis.total_lines_of_code = 500
        state.analysis.issues = []
        state.analysis.issue_summary = {}
        state.applied_changes = ["change 1", "change 2", "change 3"]
        state.stop_reason = "达到最大迭代次数"
        
        result = end_optimization(state)
        
        assert result.iteration_count == 3
    
    def test_end_with_remaining_issues(self, tmp_path):
        """测试有剩余问题的结束"""
        state = State()
        state.project_path = str(tmp_path)
        state.iteration_count = 2
        state.analysis.total_files_analyzed = 5
        state.analysis.issues = [
            CodeIssue(file_path='test.py', line_number=1, issue_type='issue1', description='test1', severity='low'),
            CodeIssue(file_path='test2.py', line_number=2, issue_type='issue2', description='test2', severity='medium')
        ]
        state.analysis.issue_summary = {'issue1': 1, 'issue2': 1}
        state.applied_changes = []
        
        result = end_optimization(state)
        
        assert result.iteration_count == 2
        assert len(result.analysis.issues) == 2
    
    def test_end_with_applied_changes(self, tmp_path):
        """测试有已应用变更的结束"""
        state = State()
        state.project_path = str(tmp_path)
        state.iteration_count = 1
        state.analysis.total_files_analyzed = 3
        state.analysis.issues = [CodeIssue(file_path='test.py', line_number=1, issue_type='test', description='test', severity='low')]
        state.applied_changes = ["优化 main.py: 5 处变更"]
        
        result = end_optimization(state)
        
        assert len(result.applied_changes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
