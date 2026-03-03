"""
性能分析工具
"""
import time
import functools
from typing import Dict, Any, Callable, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """性能指标"""
    function_name: str
    duration: float
    start_time: datetime
    end_time: datetime
    call_count: int = 1
    memory_usage: Optional[Dict[str, int]] = None


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
    
    def track_function(self, name: Optional[str] = None):
        """装饰器：跟踪函数性能"""
        def decorator(func: Callable):
            func_name = name or f"{func.__module__}.{func.__func__.name}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._track_execution(func_name, func, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def _track_execution(self, name: str, func: Callable, *args, **kwargs):
        """跟踪执行过程"""
        start_time = datetime.now()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = datetime.now()
            end_memory = self._get_memory_usage()
            
            duration = (end_time - start_time).total_seconds()
            memory_delta = self._calculate_memory_delta(start_memory, end_memory)
            
            # 更新指标
            if name in self.metrics:
                existing = self.metrics[name]
                existing.call_count += 1
                existing.duration = (existing.duration * (existing.call_count - 1) + duration) / existing.call_count
            else:
                self.metrics[name] = PerformanceMetrics(
                    function_name=name,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                    call_count=1,
                    memory_usage=memory_delta
                )
        
        return result
    
    @ contextmanager
    def track_block(self, name: str):
        """上下文管理器：跟踪代码块性能"""
        start_time = datetime.now()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = datetime.now()
            end_duration = self._get_memory_usage()
            
            duration = (end_time - start_time).total_seconds()
            memory_delta = self._calculate_memory_delta(start_memory, end_duration)
            
            # 记录块性能
            block_name = f"block:{name}"
            if block_name in self.metrics:
                existing = self.metrics[block_name]
                existing.call_count += 1
                existing.duration = (existing.duration * (existing.call_count - 1) + duration) / existing.call_count
            else:
                self.metrics[block_name] = PerformanceMetrics(
                    function_name=block_name,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                    call_count=1,
                    memory_usage=memory_delta
                )
    
    def get_report(self, sort_by: str = "duration") -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics:
            return {"message": "没有性能数据"}
        
        metrics_list = list(self.metrics.values())
        
        if sort_by == "duration":
            metrics_list.sort(key=lambda x: x.duration, reverse=True)
        elif sort_by == "call_count":
            metrics_list.sort(key=lambda x: x.call_count, reverse=True)
        
        total_duration = sum(m.duration * m.call_count for m in self.metrics.values())
        
        return {
            "total_functions": len(self.metrics),
            "total_duration": total_duration,
            "average_duration": total_duration / len(self.metrics) if self.metrics else 0,
            "top_functions": [
                {
                    "name": m.function_name,
                    "average_duration": m.duration,
                    "total_duration": m.duration * m.call_count,
                    "call_count": m.call_count,
                    "memory_usage": m.memory_usage
                }
                for m in metrics_list[:10]
            ]
        }
    
    def clear(self):
        """清除所有指标"""
        self.metrics.clear()
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """获取当前内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,  # 物理内存
                "vms": memory_info.vms,  # 虚拟内存
            }
        except ImportError:
            return {}
    
    def _calculate_memory_delta(self, start: Dict[str, int], end: Dict[str, int]) -> Optional[Dict[str, int]]:
        """计算内存使用变化"""
        if not start or not end:
            return None
        
        return {
            key: end.get(key, 0) - start.get(key, 0)
            for key in ["rss", "vms"]
        }


# 全局性能跟踪器实例
global_tracker = PerformanceTracker()


# 便捷装饰器
def track_performance(name: Optional[str] = None):
    """性能跟踪装饰器"""
    return global_tracker.track_function(name)


@contextmanager
def track_performance_block(name: str):
    """性能跟踪上下文管理器"""
    with global_tracker.track_block(name):
        yield


def get_performance_report() -> Dict[str, Any]:
    """获取全局性能报告"""
    return global_tracker.get_report()


def clear_performance_metrics():
    """清除全局性能指标"""
    global_tracker.clear()