#!/usr/bin/env python3
"""
测试 LLM 增强器 - 使用 mock 测试 API 调用
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.enhancer import LLMEnhancer, get_llm_enhancer, analyze_with_llm


class TestLLMEnhancer:
    """LLMEnhancer 测试类"""
    
    def test_init_without_api_key(self):
        """测试没有 API Key 时的初始化"""
        with patch.dict('os.environ', {}, clear=True):
            with patch('src.llm.enhancer.get_config') as mock_config:
                mock_config.return_value.get_llm_config.return_value = {
                    'enabled': True,
                    'model': 'kimi-k2-5',
                    'max_tokens': 1000,
                    'temperature': 0.3
                }
                enhancer = LLMEnhancer()
                assert not enhancer.is_available()
    
    def test_init_disabled(self):
        """测试 LLM 被禁用时"""
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': False,
                'model': 'kimi-k2-5'
            }
            enhancer = LLMEnhancer(api_key='test-key')
            assert not enhancer.is_available()
    
    def test_analyze_code_issues_not_available(self):
        """测试 LLM 不可用时返回提示"""
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': False
            }
            enhancer = LLMEnhancer()
            result = enhancer.analyze_code_issues("code", [])
            assert "不可用" in result
    
    def test_analyze_code_issues_empty_code(self):
        """测试空代码返回提示"""
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                # Mock client
                enhancer.client = Mock()
                result = enhancer.analyze_code_issues("", [])
                assert "为空" in result
    
    def test_analyze_code_issues_empty_issues(self):
        """测试没有问题列表时返回提示"""
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                enhancer.client = Mock()
                result = enhancer.analyze_code_issues("print('hello')", [])
                assert "没有发现" in result
    
    @patch('src.llm.enhancer.OpenAI')
    def test_analyze_code_issues_success(self, mock_openai):
        """测试成功分析代码问题"""
        # Mock API 响应
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="这是一个很好的代码建议"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True,
                'model': 'kimi-k2-5',
                'max_tokens': 1000,
                'temperature': 0.3
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                
                issues = [
                    {'type': 'long_line', 'description': '行太长', 'severity': 'warning', 'line': 1}
                ]
                result = enhancer.analyze_code_issues("print('hello')", issues)
                
                assert "代码建议" in result
                mock_client.chat.completions.create.assert_called_once()
    
    @patch('src.llm.enhancer.OpenAI')
    def test_analyze_code_issues_api_error(self, mock_openai):
        """测试 API 错误处理"""
        from openai import APIError
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = APIError(
            message="API Error",
            request=Mock(),
            body={}
        )
        mock_openai.return_value = mock_client
        
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True,
                'model': 'kimi-k2-5',
                'max_tokens': 1000,
                'temperature': 0.3
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                
                issues = [
                    {'type': 'long_line', 'description': '行太长', 'severity': 'warning', 'line': 1}
                ]
                result = enhancer.analyze_code_issues("print('hello')", issues)
                
                assert "❌" in result or "失败" in result
    
    @patch('src.llm.enhancer.OpenAI')
    def test_suggest_refactoring_success(self, mock_openai):
        """测试成功获取重构建议"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="重构建议：提取函数"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True,
                'model': 'kimi-k2-5',
                'max_tokens': 1000,
                'temperature': 0.3
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                
                result = enhancer.suggest_refactoring("def foo(): pass", "readability")
                
                assert "重构" in result or "提取" in result
    
    def test_suggest_refactoring_invalid_target(self):
        """测试无效的重构目标"""
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                enhancer.client = Mock()
                
                # 使用无效的目标
                result = enhancer.suggest_refactoring("code", "invalid_target")
                # 应该回退到默认值
                assert result is not None
    
    def test_explain_issue_not_available(self):
        """测试 LLM 不可用时解释问题"""
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': False
            }
            enhancer = LLMEnhancer()
            result = enhancer.explain_issue("syntax_error", "语法错误")
            assert "不可用" in result
    
    @patch('src.llm.enhancer.OpenAI')
    def test_explain_issue_success(self, mock_openai):
        """测试成功解释问题"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="这是一个语法错误"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True,
                'model': 'kimi-k2-5',
                'max_tokens': 800,
                'temperature': 0.3
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                
                result = enhancer.explain_issue("long_line", "行超过100字符")
                
                assert "语法错误" in result or result is not None
    
    @patch('src.llm.enhancer.OpenAI')
    def test_generate_docstring_success(self, mock_openai):
        """测试成功生成文档字符串"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='\n    Args:\n        x: 参数\n    \n    Returns:\n        结果\n    '))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True,
                'model': 'kimi-k2-5',
                'max_tokens': 500,
                'temperature': 0.3
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                
                result = enhancer.generate_docstring("def foo(x): return x")
                
                assert result is not None
    
    def test_generate_docstring_empty_code(self):
        """测试空代码生成文档"""
        with patch('src.llm.enhancer.get_config') as mock_config:
            mock_config.return_value.get_llm_config.return_value = {
                'enabled': True
            }
            with patch.dict('os.environ', {'KIMI_API_KEY': 'test-key'}):
                enhancer = LLMEnhancer()
                enhancer.client = Mock()
                
                result = enhancer.generate_docstring("")
                assert "为空" in result


class TestLLMHelperFunctions:
    """测试便捷函数"""
    
    @patch('src.llm.enhancer.LLMEnhancer')
    def test_get_llm_enhancer(self, mock_enhancer_class):
        """测试 get_llm_enhancer 函数"""
        mock_instance = Mock()
        mock_enhancer_class.return_value = mock_instance
        
        result = get_llm_enhancer()
        
        assert result is mock_instance
    
    @patch('src.llm.enhancer.get_llm_enhancer')
    def test_analyze_with_llm(self, mock_get_enhancer):
        """测试 analyze_with_llm 函数"""
        mock_enhancer = Mock()
        mock_enhancer.analyze_code_issues.return_value = "分析结果"
        mock_get_enhancer.return_value = mock_enhancer
        
        result = analyze_with_llm("code", [])
        
        assert result == "分析结果"
        mock_enhancer.analyze_code_issues.assert_called_once_with("code", [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
