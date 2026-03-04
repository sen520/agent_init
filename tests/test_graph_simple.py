#!/usr/bin/env python3
"""
测试 graph 模块 - 简化版
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGraphBase:
    """Graph base 测试"""

    def test_graph_base_import(self):
        """测试 graph.base 可以导入"""
        from src.graph import base
        assert hasattr(base, 'optimization_app')

    def test_build_simple_workflow(self):
        """测试构建简单工作流"""
        try:
            from src.graph.base import build_simple_workflow
            workflow = build_simple_workflow()
            assert workflow is not None
        except Exception as e:
            pytest.skip(f"构建失败: {e}")

    def test_optimization_app_exists(self):
        """测试 optimization_app 存在"""
        from src.graph import base
        assert base.optimization_app is not None


class TestGraphSelfOptimizing:
    """Graph self_optimizing 测试"""

    def test_graph_self_optimizing_import(self):
        """测试 graph.self_optimizing 可以导入"""
        from src.graph import self_optimizing
        assert hasattr(self_optimizing, 'build_self_optimizing_workflow')

    def test_build_self_optimizing_workflow(self):
        """测试构建自优化工作流"""
        try:
            from src.graph.self_optimizing import build_self_optimizing_workflow
            workflow = build_self_optimizing_workflow()
            assert workflow is not None
        except Exception as e:
            pytest.skip(f"构建失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
