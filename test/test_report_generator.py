#!/usr/bin/env python3
"""
测试报告生成器
"""
import pytest
from unittest.mock import patch, Mock
import tempfile
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State, CodeIssue, CodeAnalysis
from src.utils.report_generator import generate_html_report, create_report_node


class TestGenerateHTMLReport:
    """测试 HTML 报告生成"""
    
    def test_generate_basic_html_report(self, tmp_path):
        """测试生成基本 HTML 报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.iteration_count = 1
        state.analysis.total_files_analyzed = 10
        state.analysis.total_lines_of_code = 100
        state.analysis.average_complexity = 5.0
        state.analysis.issues = []
        state.applied_changes = []
        
        # 创建 reports 目录
        (tmp_path / 'reports').mkdir(exist_ok=True)
        
        output_path = tmp_path / 'reports' / 'test_report.html'
        
        result = generate_html_report(state, str(output_path))
        
        assert Path(result).exists()
        content = Path(result).read_text()
        assert '🤖 代码分析报告' in content
        assert '10' in content  # 文件数
    
    def test_generate_report_with_issues(self, tmp_path):
        """测试生成包含问题的报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.total_files_analyzed = 5
        state.analysis.issues = [
            CodeIssue(
                file_path='test.py',
                line_number=10,
                issue_type='long_line',
                description='行太长',
                severity='warning',
                suggestion='拆分行'
            ),
            CodeIssue(
                file_path='test2.py',
                line_number=20,
                issue_type='bare_except',
                description='裸异常',
                severity='high',
                suggestion='指定异常类型'
            )
        ]
        state.analysis.issue_summary = {'long_line': 1, 'bare_except': 1}
        
        (tmp_path / 'reports').mkdir(exist_ok=True)
        output_path = tmp_path / 'reports' / 'test_report.html'
        
        result = generate_html_report(state, str(output_path))
        
        content = Path(result).read_text()
        assert 'long_line' in content
        assert 'bare_except' in content
        assert '行太长' in content
    
    def test_generate_report_with_applied_changes(self, tmp_path):
        """测试生成包含已应用变更的报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.total_files_analyzed = 3
        state.applied_changes = [
            '优化 main.py: 2 处变更',
            '优化 utils.py: 1 处变更'
        ]
        
        (tmp_path / 'reports').mkdir(exist_ok=True)
        output_path = tmp_path / 'reports' / 'test_report.html'
        
        result = generate_html_report(state, str(output_path))
        
        content = Path(result).read_text()
        assert 'main.py' in content
        assert 'utils.py' in content
    
    def test_generate_report_default_path(self, tmp_path):
        """测试使用默认路径生成报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.issues = []
        
        result = generate_html_report(state)
        
        assert 'reports' in result
        assert 'analysis_report_' in result
        assert result.endswith('.html')
    
    def test_report_with_no_issues(self, tmp_path):
        """测试没有问题的报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.total_files_analyzed = 5
        state.analysis.issues = []
        
        (tmp_path / 'reports').mkdir(exist_ok=True)
        output_path = tmp_path / 'reports' / 'test_report.html'
        
        result = generate_html_report(state, str(output_path))
        
        content = Path(result).read_text()
        assert '没有发现代码问题' in content or '🎉' in content


class TestCreateReportNode:
    """测试报告节点"""
    
    @patch('src.utils.report_generator.generate_html_report')
    def test_create_report_node_success(self, mock_generate, tmp_path):
        """测试成功创建报告节点"""
        mock_generate.return_value = str(tmp_path / 'reports' / 'report.html')
        
        state = State()
        state.project_path = str(tmp_path)
        
        result = create_report_node(state)
        
        assert 'report.html' in result.logs[-1]
    
    @patch('src.utils.report_generator.generate_html_report')
    def test_create_report_node_error(self, mock_generate, tmp_path):
        """测试报告节点错误处理"""
        mock_generate.side_effect = ImportError("缺少 jinja2")
        
        state = State()
        state.project_path = str(tmp_path)
        
        result = create_report_node(state)
        
        assert '缺少 jinja2' in result.logs[-1] or '失败' in result.logs[-1]


class TestReportTemplateRendering:
    """测试报告模板渲染"""
    
    def test_card_styling(self, tmp_path):
        """测试卡片样式"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.total_files_analyzed = 10
        state.analysis.total_lines_of_code = 1000
        state.analysis.issues = []
        
        (tmp_path / 'reports').mkdir(exist_ok=True)
        output_path = tmp_path / 'reports' / 'test_report.html'
        
        result = generate_html_report(state, str(output_path))
        
        content = Path(result).read_text()
        # 检查 CSS 样式存在
        assert 'summary-cards' in content or 'card' in content
    
    def test_severity_styling(self, tmp_path):
        """测试严重级别样式"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.issues = [
            CodeIssue(
                file_path='test.py',
                line_number=1,
                issue_type='test',
                description='test',
                severity='high'
            ),
            CodeIssue(
                file_path='test2.py',
                line_number=1,
                issue_type='test',
                description='test',
                severity='medium'
            ),
            CodeIssue(
                file_path='test3.py',
                line_number=1,
                issue_type='test',
                description='test',
                severity='low'
            )
        ]
        
        (tmp_path / 'reports').mkdir(exist_ok=True)
        output_path = tmp_path / 'reports' / 'test_report.html'
        
        result = generate_html_report(state, str(output_path))
        
        content = Path(result).read_text()
        # 检查不同严重级别的样式类
        assert 'high' in content
        assert 'medium' in content or 'warning' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
