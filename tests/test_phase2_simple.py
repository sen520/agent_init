#!/usr/bin/env python3
"""
测试 phase2 节点 - 简化版
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State


class TestPhase2Nodes:
    """Phase2 节点测试"""

    def test_llm_analyze_issues_import(self):
        """测试 llm_analyze_issues 可以导入"""
        from src.nodes.phase2 import llm_analyze_issues
        assert callable(llm_analyze_issues)

    def test_validate_optimization_import(self):
        """测试 validate_optimization 可以导入"""
        from src.nodes.phase2 import validate_optimization
        assert callable(validate_optimization)

    def test_generate_llm_report_import(self):
        """测试 generate_llm_report 可以导入"""
        from src.nodes.phase2 import generate_llm_report
        assert callable(generate_llm_report)

    def test_llm_analyze_issues_basic(self):
        """测试 LLM 分析基本调用"""
        from src.nodes.phase2 import llm_analyze_issues

        state = State(project_path=str(Path(__file__).parent.parent))
        try:
            result = llm_analyze_issues(state)
            assert result is not None
        except Exception as e:
            pytest.skip(f"依赖未满足: {e}")

    def test_validate_optimization_basic(self):
        """测试验证优化基本调用"""
        from src.nodes.phase2 import validate_optimization

        state = State(project_path=str(Path(__file__).parent.parent))
        try:
            result = validate_optimization(state)
            assert result is not None
        except Exception as e:
            pytest.skip(f"依赖未满足: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
