#!/usr/bin/env python3
"""
测试 performance - src/utils/performance.py
"""
import pytest
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.performance import PerformanceTracker, PerformanceMetrics


class TestPerformanceTracker:
    """PerformanceTracker 测试类"""
    
    def test_init(self):
        """测试初始化"""
        tracker = PerformanceTracker()
        assert tracker is not None
        assert tracker.metrics == {}
    
    def test_track_function(self):
        """测试跟踪函数"""
        tracker = PerformanceTracker()
        
        @tracker.track_function("test_func")
        def test_func():
            time.sleep(0.01)
            return 42
        
        result = test_func()
        
        assert result == 42
        assert "test_func" in tracker.metrics
    
    def test_get_metrics(self):
        """测试获取指标"""
        tracker = PerformanceTracker()
        
        @tracker.track_function("test_func")
        def test_func():
            return 1
        
        test_func()
        
        metrics = tracker.get_metrics("test_func")
        
        assert metrics is not None
        assert metrics.function_name == "test_func"
    
    def test_get_all_metrics(self):
        """测试获取所有指标"""
        tracker = PerformanceTracker()
        
        @tracker.track_function("func1")
        def func1():
            return 1
        
        @tracker.track_function("func2")
        def func2():
            return 2
        
        func1()
        func2()
        
        all_metrics = tracker.get_all_metrics()
        
        assert len(all_metrics) == 2
    
    def test_reset(self):
        """测试重置"""
        tracker = PerformanceTracker()
        
        @tracker.track_function("test_func")
        def test_func():
            return 1
        
        test_func()
        tracker.reset()
        
        assert tracker.metrics == {}


class TestPerformanceMetrics:
    """PerformanceMetrics 测试"""
    
    def test_create_metrics(self):
        """测试创建指标"""
        from datetime import datetime
        
        metrics = PerformanceMetrics(
            function_name="test",
            duration=0.1,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert metrics.function_name == "test"
        assert metrics.duration == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
