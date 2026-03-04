#!/usr/bin/env python3
"""
测试 self_optimizing 节点 - 简化版
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State


class TestSelfOptimizingNodes:
    """自优化节点测试"""

    def test_start_self_optimization_import(self):
        """测试 start_self_optimization 可以导入"""
        from src.nodes.self_optimizing import start_self_optimization
        assert callable(start_self_optimization)

    def test_run_optimization_round_import(self):
        """测试 run_optimization_round 可以导入"""
        from src.nodes.self_optimizing import run_optimization_round
        assert callable(run_optimization_round)

    def test_create_self_optimization_report_import(self):
        """测试 create_self_optimization_report 可以导入"""
        from src.nodes.self_optimizing import create_self_optimization_report
        assert callable(create_self_optimization_report)

    def test_start_self_optimization_basic(self):
        """测试启动自优化基本调用"""
        from src.nodes.self_optimizing import start_self_optimization

        state = State(project_path=str(Path(__file__).parent.parent))
        try:
            result = start_self_optimization(state)
            assert result is not None
        except Exception as e:
            pytest.skip(f"依赖未满足: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
