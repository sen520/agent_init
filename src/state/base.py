from typing import Dict, List, Any, Optional
import logging
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class CodeFileMetrics(BaseModel):
    """单个代码文件的度量"""
    file_path: str
    lines_of_code: int = 0
    functions_count: int = 0
    classes_count: int = 0
    cyclomatic_complexity: float = 0.0
    docstring_present: bool = False
    imports: List[str] = Field(default_factory=list)
    

class CodeIssue(BaseModel):
    """代码问题"""
    file_path: str
    line_number: int
    issue_type: str  # "documentation", "naming", "complexity", "duplication", "bug_risk"
    description: str
    severity: str = "medium"  # "low", "medium", "high"
    suggestion: str = ""


class OptimizationSuggestion(BaseModel):
    """优化建议"""
    suggestion_id: str
    description: str
    priority: int = 3  # 1=high, 2=medium, 3=low
    estimated_effort: int = 1  # 1-5 难度等级
    expected_impact: Dict[str, float] = Field(default_factory=dict)
    files_affected: List[str] = Field(default_factory=list)
    status: str = "pending"  # "pending", "approved", "implemented", "rejected"


class CodeAnalysis(BaseModel):
    """代码分析结果"""
    timestamp: datetime = Field(default_factory=datetime.now)
    project_root: str = ""
    total_files_analyzed: int = 0
    total_lines_of_code: int = 0
    average_complexity: float = 0.0
    
    # 详细数据
    file_metrics: List[CodeFileMetrics] = Field(default_factory=list)
    issues: List[CodeIssue] = Field(default_factory=list)
    
    # 摘要统计
    issue_summary: Dict[str, int] = Field(default_factory=dict)
    language_distribution: Dict[str, int] = Field(default_factory=dict)


class ImplementationResult(BaseModel):
    """实施结果"""
    suggestion_id: str
    implemented_at: datetime
    changed_files: List[str] = Field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    tests_passed: bool = True
    before_metrics: Dict[str, Any] = Field(default_factory=dict)
    after_metrics: Dict[str, Any] = Field(default_factory=dict)


class OptimizationPlan(BaseModel):
    """优化计划"""
    suggestions: List[OptimizationSuggestion] = Field(default_factory=list)
    selected_suggestion: Optional[OptimizationSuggestion] = None
    implementation_order: List[str] = Field(default_factory=list)


class State(BaseModel):
    """工作流状态"""
    # 项目信息
    project_path: str = ""
    project_name: str = ""
    project_type: str = ""  # "python", "javascript", "mixed"
    
    # 分析阶段
    analysis: CodeAnalysis = CodeAnalysis()
    plan: OptimizationPlan = OptimizationPlan()
    analysis_reports: List[Dict[str, Any]] = Field(default_factory=list)
    
    # 实施阶段
    current_implementation: Optional[ImplementationResult] = None
    implementations: List[ImplementationResult] = Field(default_factory=list)
    applied_changes: List[str] = Field(default_factory=list)  # 应用的具体变更
    
    # 评估阶段
    baseline_metrics: Dict[str, Any] = Field(default_factory=dict)
    current_metrics: Dict[str, Any] = Field(default_factory=dict)
    improvement_summary: Dict[str, float] = Field(default_factory=dict)
    
    # Phase 2 新增字段
    llm_suggestions: List[Dict[str, Any]] = Field(default_factory=list)  # LLM 建议
    validation_result: Optional[Dict[str, Any]] = None  # 验证结果
    report_files: List[str] = Field(default_factory=list)  # 生成的报告文件
    
    # 循环控制
    iteration_count: int = 0
    max_iterations: int = 5
    should_continue: bool = True
    stop_reason: Optional[str] = None
    
    # 日志和错误处理
    logs: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    def add_log(self, message: str):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        # 保持日志大小合理（原地删除，不创建新列表）
        if len(self.logs) > 100:
            del self.logs[:-50]
    
    def add_error(self, error: str):
        """添加错误消息"""
        self.errors.append(error)
        logger.info(f"❌ 错误: {error}")
