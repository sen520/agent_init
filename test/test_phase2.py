#!/usr/bin/env python3
"""
测试 Phase 2 节点
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State, CodeIssue
from src.nodes.phase2 import llm_analyze_issues, validate_optimization, generate_llm_report


class TestLLMAnalyzeIssues:
    """测试 LLM 分析节点"""
    
    @patch('src.nodes.phase2.get_config')
    def test_llm_disabled(self, mock_get_config):
        """测试 LLM 被禁用时"""
        mock_get_config.return_value.get_llm_config.return_value = {
            'enabled': False
        }
        
        state = State()
        state.analysis.issues = []
        
        result = llm_analyze_issues(state)
        
        assert "LLM 未启用" in result.logs[-1]
    
    @patch('src.nodes.phase2.get_config')
    @patch('src.nodes.phase2.LLMEnhancer')
    def test_llm_not_available(self, mock_enhancer_class, mock_get_config):
        """测试 LLM 客户端不可用时"""
        mock_get_config.return_value.get_llm_config.return_value = {
            'enabled': True
        }
        
        mock_enhancer = Mock()
        mock_enhancer.is_available.return_value = False
        mock_enhancer_class.return_value = mock_enhancer
        
        state = State()
        state.analysis.issues = []
        
        result = llm_analyze_issues(state)
        
        assert "LLM 客户端不可用" in result.logs[-1]
    
    @patch('src.nodes.phase2.get_config')
    @patch('src.nodes.phase2.LLMEnhancer')
    def test_no_issues_to_analyze(self, mock_enhancer_class, mock_get_config):
        """测试没有问题需要分析时"""
        mock_get_config.return_value.get_llm_config.return_value = {
            'enabled': True
        }
        
        mock_enhancer = Mock()
        mock_enhancer.is_available.return_value = True
        mock_enhancer_class.return_value = mock_enhancer
        
        state = State()
        state.analysis.issues = []
        
        result = llm_analyze_issues(state)
        
        assert "没有问题需要分析" in result.logs[-1]
    
    @patch('src.nodes.phase2.get_config')
    @patch('src.nodes.phase2.LLMEnhancer')
    def test_successful_analysis(self, mock_enhancer_class, mock_get_config, tmp_path):
        """测试成功分析"""
        mock_get_config.return_value.get_llm_config.return_value = {
            'enabled': True
        }
        
        mock_enhancer = Mock()
        mock_enhancer.is_available.return_value = True
        mock_enhancer.analyze_code_issues.return_value = "这是 LLM 分析结果"
        mock_enhancer_class.return_value = mock_enhancer
        
        # 创建临时文件
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.issues = [
            CodeIssue(
                file_path="test.py",
                line_number=1,
                issue_type="debug_print",
                description="使用 print",
                severity="info"
            )
        ]
        
        result = llm_analyze_issues(state)
        
        assert len(result.llm_suggestions) == 1
        assert result.llm_suggestions[0]['analysis'] == "这是 LLM 分析结果"
    
    @patch('src.nodes.phase2.get_config')
    def test_import_error(self, mock_get_config):
        """测试导入错误时"""
        mock_get_config.return_value.get_llm_config.return_value = {
            'enabled': True
        }
        
        with patch.dict('sys.modules', {'src.llm.enhancer': None}):
            state = State()
            state.analysis.issues = []
            
            result = llm_analyze_issues(state)
            
            assert "LLM 模块不可用" in result.logs[-1]


class TestValidateOptimization:
    """测试验证优化节点"""
    
    @patch('src.nodes.phase2.TestValidator')
    def test_no_modified_files(self, mock_validator_class):
        """测试没有修改文件时"""
        state = State()
        state.implementations = []
        
        result = validate_optimization(state)
        
        assert "没有文件被修改" in result.logs[-1]
    
    @patch('src.nodes.phase2.TestValidator')
    def test_validation_success(self, mock_validator_class):
        """测试验证成功"""
        mock_validator = Mock()
        mock_validator.validate_after_optimization.return_value = {
            'success': True,
            'syntax_valid': True,
            'tests_passed': True
        }
        mock_validator_class.return_value = mock_validator
        
        state = State()
        # 模拟有修改的实现
        mock_impl = Mock()
        mock_impl.lines_added = 5
        mock_impl.changed_files = ['/path/to/file.py']
        state.implementations = [mock_impl]
        
        result = validate_optimization(state)
        
        assert result.validation_result['success']
        assert "验证通过" in result.logs[-1]
    
    @patch('src.nodes.phase2.TestValidator')
    @patch('src.nodes.phase2.FileModifier')
    def test_validation_failure_with_rollback(self, mock_modifier_class, mock_validator_class):
        """测试验证失败并回滚"""
        mock_validator = Mock()
        mock_validator.validate_after_optimization.return_value = {
            'success': False,
            'syntax_valid': True,
            'tests_passed': False,
            'should_rollback': True
        }
        mock_validator_class.return_value = mock_validator
        
        mock_modifier = Mock()
        mock_modifier.rollback_all.return_value = (1, 0)
        mock_modifier_class.return_value = mock_modifier
        
        state = State()
        mock_impl = Mock()
        mock_impl.lines_added = 5
        mock_impl.changed_files = ['/path/to/file.py']
        state.implementations = [mock_impl]
        state.applied_changes = ["change 1"]
        
        result = validate_optimization(state)
        
        assert "验证失败" in result.logs[-2]
        assert result.applied_changes == []  # 应该清空
    
    @patch('src.nodes.phase2.TestValidator')
    def test_import_error(self, mock_validator_class):
        """测试导入错误时"""
        with patch.dict('sys.modules', {'src.testing.validator': None}):
            state = State()
            mock_impl = Mock()
            mock_impl.lines_added = 5
            mock_impl.changed_files = ['/path/to/file.py']
            state.implementations = [mock_impl]
            
            result = validate_optimization(state)
            
            assert "验证模块不可用" in result.logs[-1]


class TestGenerateLLMReport:
    """测试生成 LLM 报告节点"""
    
    def test_generate_html_report(self, tmp_path):
        """测试生成 HTML 报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.total_files_analyzed = 10
        state.analysis.total_lines_of_code = 100
        state.analysis.issues = []
        
        result = generate_llm_report(state)
        
        # 应该成功执行
        assert result is not None
    
    def test_generate_llm_markdown_report(self, tmp_path):
        """测试生成 LLM Markdown 报告"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.timestamp = tmp_path.stat().st_mtime
        state.analysis.issues = []
        state.llm_suggestions = [
            {
                'file': 'test.py',
                'analysis': '这是一个分析结果'
            }
        ]
        
        # 创建 reports 目录
        (tmp_path / 'reports').mkdir(exist_ok=True)
        
        result = generate_llm_report(state)
        
        # 应该成功执行
        assert result is not None
    
    def test_report_generation_error(self, tmp_path):
        """测试报告生成错误处理"""
        state = State()
        state.project_path = str(tmp_path)
        state.analysis.issues = []
        state.llm_suggestions = [
            {
                'file': 'test.py',
                'analysis': '分析结果'
            }
        ]
        
        # 模拟错误
        with patch('src.nodes.phase2.generate_html_report') as mock_generate:
            mock_generate.side_effect = Exception("生成错误")
            
            result = generate_llm_report(state)
            
            # 应该处理错误
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
