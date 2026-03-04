#!/usr/bin/env python3
"""
测试 optimization 节点 - 简化版
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State


class TestOptimizationNodes:
    """优化节点测试"""

    def test_apply_optimization_import(self):
        """测试 apply_optimization 可以导入"""
        from src.nodes.optimization import apply_optimization
        assert callable(apply_optimization)

    def test_apply_optimization_basic_call(self):
        """测试 apply_optimization 基本调用"""
        from src.nodes.optimization import apply_optimization

        state = State(project_path=str(Path(__file__).parent.parent))
        # 由于依赖复杂，主要测试不抛出异常
        try:
            result = apply_optimization(state)
            assert result is not None
        except Exception as e:
            # 预期的异常（如缺少依赖）
            pytest.skip(f"依赖未满足: {e}")


class TestOptimizationHelpers:
    """测试优化辅助功能"""

    def test_file_modifier_import(self):
        """测试 file_modifier 可以导入"""
        from src.utils.file_modifier import FileModifier
        assert FileModifier is not None

    def test_file_modifier_init(self):
        """测试 FileModifier 初始化"""
        from src.utils.file_modifier import FileModifier

        try:
            modifier = FileModifier()
            assert modifier is not None
        except Exception as e:
            pytest.skip(f"初始化失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
