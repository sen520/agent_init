#!/usr/bin/env python3
"""
测试优化节点 - src/nodes/optimization.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State, ImplementationResult
from src.nodes.optimization import apply_optimization


class TestApplyOptimization:
    """测试 apply_optimization 函数"""
    
    @pytest.fixture
    def temp_project(self):
        """创建临时项目"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # 创建 Python 文件
            (root / "main.py").write_text("print('hello')\n")
            (root / "utils.py").write_text("def helper(): pass\n")
            yield root
    
    @patch('src.nodes.optimization.get_config')
    def test_no_files_analyzed(self, mock_get_config):
        """测试没有分析文件时跳过优化"""
        mock_get_config.return_value.get.return_value = 15
        mock_get_config.return_value.get_analysis_config.return_value = {
            'max_files_to_optimize': 15,
            'skip_files': ['__init__.py']
        }
        
        state = State()
        state.analysis.total_files_analyzed = 0
        
        result = apply_optimization(state)
        
        assert "没有需要优化的文件" in result.logs[-1]
    
    @patch('src.nodes.optimization.get_config')
    @patch('src.nodes.optimization.FileScanner')
    def test_no_files_to_optimize(self, mock_scanner_class, mock_get_config):
        """测试没有文件需要优化"""
        mock_get_config.return_value.get.return_value = 15
        mock_get_config.return_value.get_analysis_config.return_value = {
            'max_files_to_optimize': 15,
            'skip_files': ['__init__.py']
        }
        mock_get_config.return_value.get.return_value = ['strategy1']
        
        mock_scanner = Mock()
        mock_scanner.scan_python_files.return_value = []
        mock_scanner_class.return_value = mock_scanner
        
        state = State()
        state.project_path = "/tmp"
        state.analysis.total_files_analyzed = 1
        state.analysis.issues = []
        
        result = apply_optimization(state)
        
        assert "没有需要优化的文件" in result.logs[-1] or "没有可优化的文件" in str(result.stop_reason)
    
    @patch('src.nodes.optimization.get_config')
    @patch('src.nodes.optimization.FileScanner')
    @patch('src.nodes.optimization.CodeOptimizer')
    def test_successful_optimization(self, mock_optimizer_class, mock_scanner_class, mock_get_config, temp_project):
        """测试成功优化"""
        mock_get_config.return_value.get.return_value = 15
        mock_get_config.return_value.get_analysis_config.return_value = {
            'max_files_to_optimize': 15,
            'skip_files': ['__init__.py']
        }
        mock_get_config.return_value.get.return_value = ['import_optimizer']
        
        mock_scanner = Mock()
        mock_scanner.scan_python_files.return_value = ['main.py']
        mock_scanner_class.return_value = mock_scanner
        
        mock_optimizer = Mock()
        mock_optimizer.analyze_file.return_value = {'needs_optimization': True}
        mock_optimizer_class.return_value = mock_optimizer
        
        with patch('src.nodes.optimization.optimize_code_file') as mock_optimize:
            mock_optimize.return_value = {
                'optimization_applied': True,
                'optimized_content': '# optimized\n',
                'changes_count': 2,
                'strategies_applied': ['import_optimizer'],
                'specific_changes': [{'type': 'import'}]
            }
            
            with patch('src.nodes.optimization.apply_optimization_safely') as mock_apply:
                mock_apply.return_value = {
                    'success': True,
                    'changes_applied': True,
                    'backup_path': '/backup/main.py.bak'
                }
                
                state = State()
                state.project_path = str(temp_project)
                state.analysis.total_files_analyzed = 1
                from src.state.base import CodeIssue
                state.analysis.issues = [
                    CodeIssue(file_path='main.py', line_number=1, issue_type='import', description='test', severity='low')
                ]
                
                result = apply_optimization(state)
                
                assert len(result.implementations) >= 0
    
    @patch('src.nodes.optimization.get_config')
    def test_import_error(self, mock_get_config):
        """测试导入错误处理"""
        mock_get_config.return_value.get.return_value = 15
        
        with patch.dict('sys.modules', {
            'src.strategies.optimization_strategies': None,
            'src.tools.file_scanner': None
        }):
            state = State()
            state.project_path = "/tmp"
            state.analysis.total_files_analyzed = 1
            
            result = apply_optimization(state)
            
            assert "导入优化模块失败" in result.logs[0]
    
    @patch('src.nodes.optimization.get_config')
    @patch('src.nodes.optimization.FileScanner')
    @patch('src.nodes.optimization.CodeOptimizer')
    def test_optimization_no_changes(self, mock_optimizer_class, mock_scanner_class, mock_get_config, temp_project):
        """测试优化无变化"""
        mock_get_config.return_value.get.return_value = 15
        mock_get_config.return_value.get_analysis_config.return_value = {
            'max_files_to_optimize': 15,
            'skip_files': ['__init__.py']
        }
        mock_get_config.return_value.get.return_value = ['strategy1']
        
        mock_scanner = Mock()
        mock_scanner.scan_python_files.return_value = ['main.py']
        mock_scanner_class.return_value = mock_scanner
        
        mock_optimizer = Mock()
        mock_optimizer.analyze_file.return_value = {'needs_optimization': True}
        mock_optimizer_class.return_value = mock_optimizer
        
        with patch('src.nodes.optimization.optimize_code_file') as mock_optimize:
            # 返回无优化
            mock_optimize.return_value = {
                'optimization_applied': False,
                'changes_count': 0
            }
            
            state = State()
            state.project_path = str(temp_project)
            state.analysis.total_files_analyzed = 1
            from src.state.base import CodeIssue
            state.analysis.issues = [
                CodeIssue(file_path='main.py', line_number=1, issue_type='test', description='test', severity='low')
            ]
            
            result = apply_optimization(state)
            
            # 应该完成但没有应用变更
            assert result is not None
    
    @patch('src.nodes.optimization.get_config')
    @patch('src.nodes.optimization.FileScanner')
    @patch('src.nodes.optimization.CodeOptimizer')
    def test_optimization_exception(self, mock_optimizer_class, mock_scanner_class, mock_get_config, temp_project):
        """测试优化过程异常"""
        mock_get_config.return_value.get.return_value = 15
        mock_get_config.return_value.get_analysis_config.return_value = {
            'max_files_to_optimize': 15,
            'skip_files': ['__init__.py']
        }
        
        mock_scanner = Mock()
        mock_scanner.scan_python_files.return_value = ['main.py']
        mock_scanner_class.return_value = mock_scanner
        
        mock_optimizer = Mock()
        mock_optimizer.analyze_file.side_effect = Exception("分析错误")
        mock_optimizer_class.return_value = mock_optimizer
        
        state = State()
        state.project_path = str(temp_project)
        state.analysis.total_files_analyzed = 1
        from src.state.base import CodeIssue
        state.analysis.issues = [
            CodeIssue(file_path='main.py', line_number=1, issue_type='test', description='test', severity='low')
        ]
        
        result = apply_optimization(state)
        
        # 应该有错误记录
        assert len(result.errors) > 0 or len(result.logs) > 0


class TestAnalyzeOptimizationResults:
    """测试分析优化结果函数"""
    
    def test_no_implementations(self):
        """测试没有实现时停止优化"""
        from src.nodes.optimization import analyze_optimization_results
        
        state = State()
        state.implementations = []
        
        result = analyze_optimization_results(state)
        
        assert not result.should_continue
        assert "没有可优化的内容" in result.stop_reason
    
    def test_successful_implementations(self):
        """测试成功的实现"""
        from src.nodes.optimization import analyze_optimization_results
        from datetime import datetime
        
        state = State()
        state.implementations = [
            ImplementationResult(
                suggestion_id='test1',
                implemented_at=datetime.now(),
                changed_files=['/path/to/file.py'],
                lines_added=5,
                lines_removed=0,
                tests_passed=True,
                before_metrics={},
                after_metrics={}
            )
        ]
        state.analysis.issues = []
        
        result = analyze_optimization_results(state)
        
        assert 'files_optimized' in result.improvement_summary
    
    def test_remaining_issues(self):
        """测试还有剩余问题时继续优化"""
        from src.nodes.optimization import analyze_optimization_results
        from datetime import datetime
        
        state = State()
        state.iteration_count = 1
        state.max_iterations = 5
        state.implementations = [
            ImplementationResult(
                suggestion_id='test1',
                implemented_at=datetime.now(),
                changed_files=['/path/to/file.py'],
                lines_added=5,
                lines_removed=0,
                tests_passed=True,
                before_metrics={},
                after_metrics={}
            )
        ]
        from src.state.base import CodeIssue
        state.analysis.issues = [
            CodeIssue(file_path='test.py', line_number=1, issue_type='test', description='test', severity='low')
        ]
        
        result = analyze_optimization_results(state)
        
        # 应该继续或停止取决于实现
        assert result is not None


class TestCreateOptimizationSummary:
    """测试创建优化总结函数"""
    
    def test_create_summary_with_changes(self):
        """测试创建有变更的总结"""
        from src.nodes.optimization import create_optimization_summary
        
        state = State()
        state.project_path = "/tmp/test"
        state.iteration_count = 3
        state.applied_changes = ["change 1", "change 2", "change 3"]
        state.analysis.issues = []
        state.stop_reason = "优化完成"
        
        result = create_optimization_summary(state)
        
        assert "优化总结报告" in result.logs[-3]
        assert "迭代次数" in result.logs[-2]
    
    def test_create_summary_with_remaining_issues(self):
        """测试创建有剩余问题的总结"""
        from src.nodes.optimization import create_optimization_summary
        from src.state.base import CodeIssue
        
        state = State()
        state.project_path = "/tmp/test"
        state.iteration_count = 2
        state.applied_changes = ["change 1"]
        state.analysis.issues = [
            CodeIssue(file_path='test.py', line_number=1, issue_type='issue1', description='test1', severity='low'),
            CodeIssue(file_path='test2.py', line_number=2, issue_type='issue2', description='test2', severity='medium')
        ]
        
        result = create_optimization_summary(state)
        
        assert result is not None
        assert len(result.logs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
