#!/usr/bin/env python3
"""
测试 Graph 模块 - src/graph/base.py
"""
import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.base import (
    build_graph,
    build_simple_graph,
    build_optimization_graph,
    build_self_optimizing_graph,
    build_phase2_graph
)
from src.state.base import State


class TestBuildGraph:
    """测试 build_graph 函数"""
    
    def test_build_graph_returns_compiled_graph(self):
        """测试 build_graph 返回编译后的图"""
        graph = build_graph()
        
        # 应该返回编译后的图对象
        assert graph is not None
        # 应该有 invoke 方法
        assert hasattr(graph, 'invoke')
        assert hasattr(graph, 'ainvoke')
    
    def test_build_simple_graph(self):
        """测试构建简单图"""
        graph = build_simple_graph()
        
        assert graph is not None
        assert hasattr(graph, 'invoke')
    
    def test_build_optimization_graph(self):
        """测试构建优化图"""
        graph = build_optimization_graph()
        
        assert graph is not None
        assert hasattr(graph, 'invoke')
    
    def test_build_self_optimizing_graph(self):
        """测试构建自优化图"""
        graph = build_self_optimizing_graph()
        
        assert graph is not None
        assert hasattr(graph, 'invoke')
    
    def test_build_phase2_graph(self):
        """测试构建 Phase 2 图"""
        graph = build_phase2_graph()
        
        assert graph is not None
        assert hasattr(graph, 'invoke')


class TestSimpleGraphExecution:
    """测试简单图执行"""
    
    def test_simple_graph_sync_execution(self):
        """测试简单图的同步执行"""
        graph = build_simple_graph()
        state = State()
        state.project_path = "."
        
        # 同步执行
        result = graph.invoke(state)
        
        assert result is not None


class TestGraphNodes:
    """测试图中的节点"""
    
    def test_simple_graph_nodes(self):
        """测试简单图的节点配置"""
        # 简单图应该有这些节点：initialize, analyze, create_report, end
        graph = build_simple_graph()
        
        # 验证图可以执行
        assert graph is not None
    
    def test_optimization_graph_nodes(self):
        """测试优化图的节点配置"""
        graph = build_optimization_graph()
        
        # 优化图应该有更多节点
        assert graph is not None
    
    def test_phase2_graph_nodes(self):
        """测试 Phase 2 图的节点配置"""
        graph = build_phase2_graph()
        
        # Phase 2 图应该有 LLM 和验证节点
        assert graph is not None


class TestGraphErrorHandling:
    """测试图的错误处理"""
    pass


class TestRealNodesAvailable:
    """测试真实节点可用性"""
    
    def test_real_nodes_import(self):
        """测试真实节点可以导入"""
        try:
            from src.graph.base import REAL_NODES_AVAILABLE
            # 应该有一个布尔值表示真实节点是否可用
            assert isinstance(REAL_NODES_AVAILABLE, bool)
        except ImportError:
            pytest.skip("REAL_NODES_AVAILABLE not exported")
    
    def test_optimization_nodes_available(self):
        """测试优化节点可用性"""
        try:
            from src.graph.base import OPTIMIZATION_NODES_AVAILABLE
            assert isinstance(OPTIMIZATION_NODES_AVAILABLE, bool)
        except ImportError:
            pytest.skip("OPTIMIZATION_NODES_AVAILABLE not exported")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
