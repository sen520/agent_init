#!/usr/bin/env python3
"""
代码优化节点 - 应用具体的代码优化策略
"""
from typing import Dict, List, Any, Optional
import os
from pathlib import Path
from datetime import datetime
import uuid

from src.state.base import State, CodeAnalysis, ImplementationResult


def apply_optimization(state: State) -> State:
    """
    应用代码优化节点 - 使用真实的文件修改
    """
    try:
        from src.strategies.optimization_strategies import CodeOptimizer, optimize_code_file
        from src.tools.file_scanner import FileScanner
        from src.utils.file_modifier import FileModifier, apply_optimization_safely
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
    
    # 创建文件修改器
    modifier = FileModifier()
    
    try:
        # 扫描需要优化的文件
        scanner = FileScanner(project_path)
        python_files = scanner.scan_python_files()
        
        # 基于分析结果，找出有问题的文件
        files_with_issues = set()
        for issue in state.analysis.issues:
            files_with_issues.add(issue.file_path)
        
        # 只优化有问题的文件
        files_to_optimize = []
        for file_name in python_files[:15]:  # 限制数量
            if file_name in files_with_issues:
                file_path = os.path.join(project_path, file_name)
                
                # 跳过敏感文件
                skip_files = {'__init__.py', 'test_optimization.py', 'conftest.py'}
                if any(sf in file_name for sf in skip_files):
                    continue
                    
                files_to_optimize.append(file_path)
        
        if not files_to_optimize:
            state.logs.append("没有找到需要优化的文件")
            state.stop_reason = "没有可优化的文件"
            state.should_continue = False
            return state
        
        print(f"   🎯 确定 {len(files_to_optimize)} 个文件需要优化")
        
        # 创建优化器
        optimizer = CodeOptimizer()
        applied_implementations = []
        total_changes = 0
        
        # 对每个文件应用优化
        for file_path in files_to_optimize:
            file_name = os.path.basename(file_path)
            state.logs.append(f"优化文件: {file_name}")
            
            try:
                # 先分析文件
                analysis = optimizer.analyze_file(file_path)
                
                if analysis.get('needs_optimization', False):
                    # 选择优化策略 - 只使用安全的策略
                    strategies_to_apply = [
                        'empty_line_optimizer',    # 空行规范化
                        'comment_optimizer',       # 注释格式化
                        'import_optimizer',        # 导入组织
                    ]
                    
                    # 应用优化（获取优化后的内容）
                    result = optimize_code_file(file_path, strategies_to_apply)
                    
                    if result.get('optimization_applied') and result.get('optimized_content'):
                        optimized_content = result['optimized_content']
                        changes_count = result.get('changes_count', 0)
                        
                        # 安全地写入文件
                        apply_result = apply_optimization_safely(
                            file_path, 
                            optimized_content, 
                            modifier
                        )
                        
                        if apply_result['success'] and apply_result['changes_applied']:
                            total_changes += changes_count
                            
                            # 创建实现结果记录
                            impl = ImplementationResult(
                                suggestion_id=str(uuid.uuid4()),
                                implemented_at=datetime.now(),
                                changed_files=[file_path],
                                lines_added=changes_count,
                                lines_removed=0,
                                tests_passed=True,
                                before_metrics={},
                                after_metrics={
                                    "strategies_applied": result.get('strategies_applied', []),
                                    "changes": result.get('specific_changes', [])[:5],
                                    "backup_path": apply_result.get('backup_path')
                                }
                            )
                            applied_implementations.append(impl)
                            
                            # 添加到状态
                            state.applied_changes.append(
                                f"优化 {file_name}: {changes_count} 处变更"
                            )
                            
                            print(f"      ✅ {file_name}: {changes_count} 处优化已应用")
                            state.logs.append(f"✅ {file_name}: {changes_count} 处优化")
                        else:
                            print(f"      ⚠️ {file_name}: {apply_result.get('message', '未应用')}")
                    else:
                        print(f"      ℹ️ {file_name}: 无需优化")
                else:
                    print(f"      ℹ️ {file_name}: 无需优化")
                    
            except Exception as e:
                print(f"      ❌ {file_name}: 优化失败 - {e}")
                error_impl = ImplementationResult(
                    suggestion_id=str(uuid.uuid4()),
                    implemented_at=datetime.now(),
                    changed_files=[file_path],
                    lines_added=0,
                    lines_removed=0,
                    tests_passed=False,
                    before_metrics={},
                    after_metrics={"error": str(e)}
                )
                applied_implementations.append(error_impl)
        
        # 更新状态
        if applied_implementations:
            successful_impls = [
                impl for impl in applied_implementations 
                if impl.lines_added > 0
            ]
            if successful_impls:
                state.current_implementation = successful_impls[-1]
            state.implementations.extend(applied_implementations)
        
        state.logs.append(
            f"优化完成: 处理了 {len(files_to_optimize)} 个文件，"
            f"应用了 {total_changes} 处变更"
        )
        print(f"\n   📊 优化完成: {total_changes} 处变更已应用到 {len(modifier.modified_files)} 个文件")
        
        # 如果没有任何变更，停止优化
        if total_changes == 0:
            state.should_continue = False
            state.stop_reason = "没有可以应用的优化"
        
    except Exception as e:
        print(f"   ❌ 优化过程出错: {e}")
        error_impl = ImplementationResult(
            suggestion_id=str(uuid.uuid4()),
            implemented_at=datetime.now(),
            changed_files=[project_path],
            lines_added=0,
            lines_removed=0,
            tests_passed=False,
            before_metrics={},
            after_metrics={"error": str(e)}
        )
        state.implementations.append(error_impl)
        state.errors.append(f"优化过程错误: {e}")
        state.logs.append(f"优化失败: {e}")
        state.should_continue = False
        state.stop_reason = f"优化失败: {e}"
    
    return state
    
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
    # 判断成功的标准: after_metrics 中没有 error 字段，且 lines_added > 0
    successful_implementations = [
        impl for impl in state.implementations 
        if not impl.after_metrics.get("error") and impl.lines_added > 0
    ]
    total_changes = sum(impl.lines_added for impl in successful_implementations)
    
    # 记录基础的改进指标
    state.improvement_summary = {
        "files_optimized": len(set(impl.changed_files[0] for impl in successful_implementations if impl.changed_files)),
        "total_applied_changes": total_changes,
        "success_rate": len(successful_implementations) / max(len(state.implementations), 1) * 100,
        "iteration_count": state.iteration_count
    }
    
    # 记录baseline指标（简化）
    state.baseline_metrics = {
        "initial_issues": len(state.analysis.issues) if state.analysis else 0,
        "files_analyzed": state.analysis.total_files_analyzed if state.analysis else 0,
        "initial_complexity": state.analysis.average_complexity if state.analysis else 50
    }
    
    # 记录当前指标
    state.current_metrics = {
        "current_issues": len(state.analysis.issues) if state.analysis else 0,
        "files_optimized": state.improvement_summary["files_optimized"],
        "applied_changes": total_changes,
        "current_complexity": state.analysis.average_complexity if state.analysis else 50
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
        "files_processed": state.analysis.total_files_analyzed if state.analysis else 0,
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