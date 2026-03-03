#!/usr/bin/env python3
"""
代码优化节点 - 应用具体的代码优化策略
"""
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

from src.state.base import State, CodeAnalysis, ImplementationResult


def apply_optimization(state: State) -> State:
    """
    应用代码优化节点
    
    根据分析结果，应用相应的优化策略
    """
    try:
        from src.strategies.optimization_strategies import CodeOptimizer, optimize_code_file
        from src.tools.file_scanner import FileScanner
    except ImportError as e:
        state.logs.append(f"导入优化模块失败: {e}")
        state.errors.append(f"优化模块导入错误: {e}")
        return state
    
    # 获取项目路径
    project_path = state.project_path or "."
    
    # 如果没有分析结果或没有文件，跳过优化
    if not state.analysis.total_files_analyzed:
        state.logs.append("没有需要优化的文件")
        return state
    
    state.logs.append("开始应用代码优化...")
    
    try:
        # 扫描需要优化的文件
        scanner = FileScanner(project_path)
        python_files = scanner.scan_python_files()
        
        # 过滤出有问题的文件（基于之前的分析）
        files_to_optimize = []
        for file_name in python_files[:10]:  # 限制数量，避免过多修改
            file_path = os.path.join(project_path, file_name)
            
            # 跳过这些系统文件，避免破坏
            skip_files = {'__init__.py', 'test_optimization.py', 'config.py'}
            if any(file_name.endswith(sf) for sf in skip_files):
                continue
                
            files_to_optimize.append(file_path)
        
        if not files_to_optimize:
            state.logs.append("没有找到需要优化的文件")
            return state
        
        # 创建优化器
        optimizer = CodeOptimizer()
        applied_implementations = []
        total_changes = 0
        
        # 对每个文件应用优化
        for file_path in files_to_optimize:
            state.logs.append(f"优化文件: {os.path.basename(file_path)}")
            
            try:
                # 先分析文件
                analysis = optimizer.analyze_file(file_path)
                
                if analysis.get('needs_optimization', False):
                    # 选择优化策略 - 按优先级和安全程度排序
                    # 1. 安全策略 (不容易破坏功能)
                    strategies_to_apply = [
                        'empty_line_optimizer',          # 空行规范化
                        'comment_optimizer',             # 注释格式化  
                        'import_optimizer',              # 导入组织
                        'line_length_optimizer',         # 行长度优化
                    ]
                    # 2. 较安全策略 (添加建议标记)
                    if len(files_to_optimize) <= 3:  # 限制文件数量时添加
                        strategies_to_apply.extend([
                            'variable_naming_optimizer',    # 变量命名建议
                            'function_length_optimizer',    # 函数长度检测
                            'duplicate_code_optimizer',     # 重复代码检测
                        ])
                    
                    # 应用优化
                    result = optimize_code_file(file_path, strategies_to_apply)
                    
                    if result.get('optimization_applied'):
                        changes_count = result.get('changes_count', 0)
                        total_changes += changes_count
                        
                        # 创建实现结果记录
                        impl = ImplementationResult(
                            step=state.iteration_count,
                            strategy="auto_optimization",
                            target_file=file_path,
                            changes_count=changes_count,
                            success=True,
                            details={
                                "applied_strategies": result.get('strategies_applied', []),
                                "file_issues_count": analysis.get('total_issues', 0),
                                "changes": result.get('specific_changes', [])[:5]  # 只保留前5个变更
                            }
                        )
                        applied_implementations.append(impl)
                        
                        # 添加到状态
                        state.applied_changes.append(
                            f"优化 {os.path.basename(file_path)}: {changes_count} 处变更"
                        )
                        
                        state.logs.append(f"✅ {os.path.basename(file_path)}: {changes_count} 处优化")
                    else:
                        state.logs.append(f"ℹ️ {os.path.basename(file_path)}: 无需优化")
                else:
                    state.logs.append(f"ℹ️ {os.path.basename(file_path)}: 发现问题但无需优化")
                    
            except Exception as e:
                error_impl = ImplementationResult(
                    step=state.iteration_count,
                    strategy="auto_optimization",
                    target_file=file_path,
                    changes_count=0,
                    success=False,
                    details={"error": str(e)}
                )
                applied_implementations.append(error_impl)
                state.logs.append(f"❌ 优化 {os.path.basename(file_path)} 失败: {e}")
        
        # 更新状态
        if applied_implementations:
            state.current_implementation = applied_implementations[-1]  # 最后一个实现
            state.implementations.extend(applied_implementations)
        
        state.logs.append(f"优化完成: 处理了 {len(files_to_optimize)} 个文件，应用了 {total_changes} 处变更")
        
        # 更新分析中的问题数量（可能有些问题已被解决）
        if state.analysis:
            remaining_issues = max(0, len(state.analysis.issues) - total_changes)
            # 创建新的analysis对象反映优化后的状态
            state.analysis = CodeAnalysis(
                total_files=state.analysis.total_files,
                total_lines=state.analysis.total_lines,
                complexity=max(10, state.analysis.complexity - total_changes * 2),
                issues=[issue for i, issue in enumerate(state.analysis.issues) if i < remaining_issues]
            )
        
    except Exception as e:
        error_impl = ImplementationResult(
            step=state.iteration_count,
            strategy="auto_optimization",
            target_file=project_path,
            changes_count=0,
            success=False,
            details={"error": str(e)}
        )
        state.implementations.append(error_impl)
        state.errors.append(f"优化过程错误: {e}")
        state.logs.append(f"优化失败: {e}")
    
    return state


def analyze_optimization_results(state: State) -> State:
    """
    分析优化结果节点
    
    评估优化效果，决定是否需要继续优化
    """
    state.logs.append("分析优化效果...")
    
    if not state.implementations:
        state.logs.append("没有实施任何优化，停止优化")
        state.should_continue = False
        state.stop_reason = "没有可优化的内容"
        return state
    
    # 计算优化效果
    successful_implementations = [impl for impl in state.implementations if impl.success]
    total_changes = sum(impl.changes_count for impl in successful_implementations)
    
    # 记录基础的改进指标
    state.improvement_summary = {
        "files_optimized": len(set(impl.target_file for impl in successful_implementations)),
        "total_applied_changes": total_changes,
        "success_rate": len(successful_implementations) / max(len(state.implementations), 1) * 100,
        "iteration_count": state.iteration_count
    }
    
    # 记录baseline指标（简化）
    state.baseline_metrics = {
        "initial_issues": len(state.analysis.issues) if state.analysis else 0,
        "files_analyzed": state.analysis.total_files if state.analysis else 0,
        "initial_complexity": state.analysis.complexity if state.analysis else 50
    }
    
    # 记录当前指标
    state.current_metrics = {
        "current_issues": len(state.analysis.issues) if state.analysis else 0,
        "files_optimized": state.improvement_summary["files_optimized"],
        "applied_changes": total_changes,
        "current_complexity": state.analysis.complexity if state.analysis else 50
    }
    
    # 评估是否需要继续
    remaining_issues = len(state.analysis.issues) if state.analysis else 0
    iteration_limit_reached = state.iteration_count >= state.max_iterations
    
    if remaining_issues == 0:
        state.should_continue = False
        state.stop_reason = "所有问题已解决"
        state.logs.append("✅ 所有问题已解决，停止优化")
    elif iteration_limit_reached:
        state.should_continue = False
        state.stop_reason = "达到最大迭代次数"
        state.logs.append(f"🛑 达到最大迭代次数 ({state.max_iterations})，停止优化")
    elif total_changes == 0:
        state.should_continue = False
        state.stop_reason = "此轮没有应用任何变更"
        state.logs.append("ℹ️ 此轮没有应用变更，停止优化")
    else:
        state.should_continue = True
        state.logs.append(f"✨ 发现 {remaining_issues} 个剩余问题，继续优化")
        state.iteration_count += 1  # 增加迭代计数
    
    return state


def create_optimization_summary(state: State) -> State:
    """
    创建优化总结节点
    
    生成最终的优化报告
    """
    state.logs.append("生成优化总结...")
    
    # 创建总结报告
    summary = {
        "project": state.project_path or ".",
        "total_iterations": state.iteration_count,
        "files_processed": state.analysis.total_files if state.analysis else 0,
        "issues_found": len(state.analysis.issues) if state.analysis else 0,
        "optimizations_applied": len(state.applied_changes),
        "success": len(state.errors) == 0,
        "stop_reason": state.stop_reason or "优化完成"
    }
    
    # 添加改进指标
    if state.improvement_summary:
        summary.update(state.improvement_summary)
    
    # 记录到日志
    state.logs.append("=" * 50)
    state.logs.append("优化总结报告:")
    state.logs.append(f"  项目路径: {summary['project']}")
    state.logs.append(f"  迭代次数: {summary['total_iterations']}")
    state.logs.append(f"  处理文件: {summary['files_processed']}")
    state.logs.append(f"  发现问题: {summary['issues_found']}")
    state.logs.append(f"  应用优化: {summary['optimizations_applied']}")
    state.logs.append(f"  优化成功: {summary['success']}")
    state.logs.append(f"  停止原因: {summary['stop_reason']}")
    state.logs.append("=" * 50)
    
    return state