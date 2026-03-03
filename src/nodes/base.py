from typing import Dict, Any
from src.state.base import State


# ========================
# 节点1: 初始化项目
# ========================
def initialize_project(state: State) -> State:
    """初始化项目分析"""
    print("🔄 [节点1] 初始化项目")
    state.project_path = "/root/.openclaw/workspace/code"
    state.iteration_count += 1
    print(f"   项目路径: {state.project_path}")
    print(f"   当前迭代: {state.iteration_count}")
    return state


# ========================
# 节点2: 分析代码
# ========================
def analyze_code(state: State) -> State:
    """分析代码结构和质量"""
    print("🔍 [节点2] 分析代码")
    
    # 模拟分析结果 - 修正字段名
    state.analysis.total_files_analyzed = 25  # 模拟文件数量
    state.analysis.total_lines_of_code = 1000  # 模拟总行数
    state.analysis.average_complexity = 0.7  # 模拟复杂度
    
    # 模拟识别的问题
    state.analysis.issues = [
        "部分函数缺少文档字符串",
        "一些变量命名不够清晰",
        "存在重复代码片段",
        "某些模块耦合度较高"
    ]
    
    print(f"   发现 {len(state.analysis.issues)} 个潜在问题")
    for i, issue in enumerate(state.analysis.issues, 1):
        print(f"     {i}. {issue}")
    
    return state


# ========================
# 节点3: 生成优化计划
# ========================
def plan_optimizations(state: State) -> State:
    """基于分析结果生成优化计划"""
    print("📋 [节点3] 生成优化计划")
    
    # 基于问题生成建议
    suggestions = []
    priorities = []
    
    if "部分函数缺少文档字符串" in state.analysis.issues:
        suggestions.append("为关键函数添加文档字符串")
        priorities.append(1)  # 高优先级
        
    if "一些变量命名不够清晰" in state.analysis.issues:
        suggestions.append("改进不清晰的变量命名")
        priorities.append(2)  # 中等优先级
        
    if "存在重复代码片段" in state.analysis.issues:
        suggestions.append("提取重复代码为函数")
        priorities.append(3)  # 低优先级
    
    state.plan.suggestions = suggestions
    state.plan.priorities = priorities
    state.plan.estimated_impact = {
        "readability": 0.8,  # 可读性提升预期
        "maintainability": 0.6,  # 可维护性提升预期
        "complexity": -0.2  # 复杂度降低预期
    }
    
    print(f"   生成 {len(suggestions)} 个优化建议：")
    for i, (suggestion, priority) in enumerate(zip(suggestions, priorities), 1):
        print(f"     {i}. [{priority}] {suggestion}")
    
    return state


# ========================
# 节点4: 应用变更
# ========================
def apply_changes(state: State) -> State:
    """应用优化变更"""
    print("🛠️  [节点4] 应用变更")
    
    applied = []
    if state.plan.suggestions:
        # 模拟应用第一个高优先级优化
        suggestion = state.plan.suggestions[0]
        applied.append(f"应用优化: {suggestion}")
        state.analysis.issues = [issue for issue in state.analysis.issues 
                               if suggestion not in issue]
        
        # 更新指标
        state.improvement_metrics = {
            "lines_of_code": -10,  # 减少10行代码
            "complexity": -0.1,  # 复杂度降低0.1
            "readability": 0.2  # 可读性提升0.2
        }
    
    state.applied_changes = applied
    
    if applied:
        print(f"   已应用变更: {applied[0]}")
        print(f"   剩余问题: {len(state.analysis.issues)} 个")
    else:
        print("   没有需要应用的变更")
    
    return state


# ========================
# 节点5: 验证变更
# ========================
def verify_changes(state: State) -> State:
    """验证应用的变更"""
    print("✅ [节点5] 验证变更")
    
    if state.applied_changes:
        # 模拟验证结果
        state.verification_results = {
            "tests_passed": True,
            "functionality_preserved": True,
            "no_syntax_errors": True
        }
        print("   验证结果: 所有测试通过，功能保持完整")
    else:
        state.verification_results = {
            "tests_passed": True,  # 没有变更，测试依然通过
            "functionality_preserved": True,
            "no_syntax_errors": True
        }
        print("   验证结果: 没有变更需要验证")
    
    return state


# ========================
# 节点6: 评估结果
# ========================
def evaluate_results(state: State) -> State:
    """评估优化效果"""
    print="📊 [节点6] 评估结果"
    
    improvements = []
    if state.improvement_metrics.get("lines_of_code", 0) < 0:
        improvements.append(f"代码行数减少: {abs(state.improvement_metrics['lines_of_code'])} 行")
    
    if state.improvement_metrics.get("complexity", 0) < 0:
        improvements.append("代码复杂度降低")
    
    if state.improvement_metrics.get("readability", 0) > 0:
        improvements.append("代码可读性提升")
    
    # 决定是否继续优化
    remaining_issues = len(state.analysis.issues)
    if remaining_issues > 0 and state.iteration_count < 3:  # 最多迭代3次
        state.should_continue = True
        print(f"   仍有 {remaining_issues} 个问题待处理，继续优化")
    else:
        state.should_continue = False
        print(f"   优化完成，共 {state.iteration_count} 次迭代")
    
    if improvements:
        print(f"   改进成果: {', '.join(improvements)}")
    
    return state


# ========================
# 节点7: 结束流程
# ========================
def end_optimization(state: State) -> State:
    """结束优化流程"""
    print("🏁 [节点7] 结束优化")
    
    total_improvements = len(state.applied_changes)
    remaining_issues = len(state.analysis.issues)
    
    print("=" * 50)
    print("优化总结报告:")
    print(f"  迭代次数: {state.iteration_count}")
    print(f"  应用优化: {total_improvements} 项")
    print(f"  剩余问题: {remaining_issues} 个")
    
    if total_improvements > 0:
        print("应用的具体优化:")
        for i, change in enumerate(state.applied_changes, 1):
            print(f"  {i}. {change}")
    
    if remaining_issues > 0:
        print("未解决的问题:")
        for i, issue in enumerate(state.analysis.issues, 1):
            print(f"  {i}. {issue}")
    
    print("=" * 50)
    return state
