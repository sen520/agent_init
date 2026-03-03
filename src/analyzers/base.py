"""
代码分析器基类

该模块定义了所有代码分析器的统一接口和基础功能。
所有具体的分析器实现都应该继承自这个基类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from src.state.base import CodeIssue, Severity


@dataclass
class AnalysisResult:
    """代码分析结果"""
    file_path: str
    issues: List[CodeIssue]
    metrics: Dict[str, Any]
    analysis_time: datetime
    success: bool
    error_message: Optional[str] = None


class BaseAnalyzer(ABC):
    """
    代码分析器基类
    
    所有代码分析器都应该继承这个基类，并实现analyze方法。
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ) -> None:
        """
        初始化分析器
        
        Args:
            config: 分析器配置
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or self._get_default_logger()
        self._metrics_cache: Dict[str, AnalysisResult] = {}
    
    @abstractmethod
    def analyze(self, file_path: Union[str, Path]) -> AnalysisResult:
        """
        分析单个文件
        
        Args:
            file_path: 要分析的文件路径
            
        Returns:
            AnalysisResult: 分析结果
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现analyze方法")
    
    def analyze_directory(
        self,
        directory_path: Union[str, Path],
        file_pattern: str = "*.py",
        recursive: bool = True
    ) -> List[AnalysisResult]:
        """
        分析整个目录
        
        Args:
            directory_path: 要分析的目录路径
            file_pattern: 文件匹配模式
            recursive: 是否递归分析
            
        Returns:
            List[AnalysisResult]: 所有文件的分析结果
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        # 查找匹配的文件
        if recursive:
            files = list(directory_path.rglob(file_pattern))
        else:
            files = list(directory_path.glob(file_pattern))
        
        self.logger.info(f"开始分析 {len(files)} 个文件")
        
        results = []
        for file_path in files:
            try:
                result = self.analyze(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"分析文件 {file_path} 时出错: {e}")
                results.append(AnalysisResult(
                    file_path=str(file_path),
                    issues=[],
                    metrics={},
                    analysis_time=datetime.now(),
                    success=False,
                    error_message=str(e)
                ))
        
        self.logger.info(f"分析完成，成功: {sum(1 for r in results if r.success)}，失败: {sum(1 for r in results if not r.success)}")
        return results
    
    def get_summary(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        获取分析结果摘要
        
        Args:
            results: 分析结果列表
            
        Returns:
            Dict[str, Any]: 摘要信息
        """
        total_files = len(results)
        successful_files = sum(1 for r in results if r.success)
        total_issues = sum(len(r.issues) for r in results)
        
        # 按严重程度统计问题
        severity_counts = {}
        for result in results:
            for issue in result.issues:
                severity = issue.severity.name
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_files": total_files,
            "successful_files": successful_files,
            "failure_rate": 1.0 - (successful_files / total_files) if total_files > 0 else 0,
            "total_issues": total_issues,
            "severity_distribution": severity_counts,
            "average_issues_per_file": total_issues / total_files if total_files > 0 else 0
        }
    
    def _get_default_logger(self) -> Any:
        """获取默认日志记录器"""
        try:
            from ..utils.logger import default_logger
            return default_logger
        except ImportError:
            import logging
            return logging.getLogger(self.__class__.__name__)
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._metrics_cache.clear()