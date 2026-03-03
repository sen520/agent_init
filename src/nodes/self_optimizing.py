#!/usr/bin/env python3
"""
自优化节点 - 实现系统自优化能力的核心节点
"""
from typing import Dict, List, Any
import os

from src.state.base import State


def start_self_optimization(state: State) -> State:
    """
    开始自优化节点
    
    识别需要优化的自己的代码文件，并开始自优化循环
    """
    try:
        from src.self_optimizing.orchestrator import SelfOptimizingOrchestrator
    except ImportError as e:
        state.logs.append(f"导入自优化编排器失败: {e}")
        state.errors.append(f"自优化组件不可用: {e}")
        state.should_continue = False
        state.stop_reason = "自优化组件缺失"
        return state
    
    # 确定项目路径
    project_path = state.project_path or "."
    
    state.logs.append("🚀 开始系统自优化...")
    
    try:
        # 创建自优化编排器
        orchestrator = SelfOptimizingOrchestrator(project_path)
        
        # 检查是否有目标文件
        if not orchestrator.target_files:
            state.logs.append("⚠️ 没有找到需要自优化的目标文件")
            state.should_continue = False
            state.stop_reason = "无自优化目标"
            return state
        
        state.logs.append(f"📁 确定了 {len(orchestrator.target_files)} 个自优化目标")
        
        # 记录目标文件
        for file_path in orchestrator.target_files:
            file_name = os.path.basename(file_path)
            state.logs.append(f"   📄 {file_name}")
        
        # 将编排器存储在状态中供后续节点使用
        state.orchestrator = orchestrator
        
        # 设置继续优化
        state.should_continue = True
        state.logs.append("✅ 自优化准备完成")
        
    except Exception as e:
        state.errors.append(f"自优化初始化失败: {e}")
        state.should_continue = False
        state.stop_reason = f"自优化失败: {e}"
        state.logs.append(f"❌ 自优化初始化失败: {e}")
    
    return state


def run_optimization_round(state: State) -> State:
    """
    执行一轮自优化
    
    分析问题并应用优化策略
    """
    if not hasattr(state, 'orchestrator') or state.orchestrator is None:
        state.should_continue = False
        state.stop_reason = "自优化编排器不可用"
        state.errors.append("自优化编排器状态异常")
        return state
    
    orchestrator = state.orchestrator
    state.logs.append(f"🔄 执行第 {state.iteration_count + 1} 轮自优化...")
    
    try:
        # 执行一轮优化
        round_result = orchestrator.run_self_optimization_round()
        
        # 记录结果
        state.optimization_rounds.append(round_result)
        
        # 更新统计
        state.total_files_analyzed += round_result["files_analyzed"]
        state.total_issues_found += round_result["issues_found"]
        state.total_optimizations_applied += round_result["optimizations_applied"]
        
        # 记录应用的具体变更
        if round_result["optimizations_applied"] > 0:
            strategies_used = round_result["strategies_used"]
            for strategy in strategies_used:
                if strategy not in state.strategies_used:
                    state.strategies_used.append(strategy)
            
            # 记录具体变更示例
            applied_changes_strategies = round_result["strategies_used"]
            state.applied_changes.extend([
                f"自优化第{state.iteration_count + 1}轮: 应用{strategy}策略" 
                for strategy in applied_changes_strategies
            ])
        
        # 记录日志
        state.logs.append(f"   📊 本轮结果: 问题{round_result['issues_found']} → 优化{round_result['optimizations_applied']}")
        
        if round_result["success"] and round_result["optimizations_applied"] > 0:
            state.logs.append(f"   ✅ 第 {state.iteration_count + 1} 轮自优化成功")
            state.iteration_count += 1
            state.should_continue = True
        else:
            # 优化收敛或失败，停止
            state.should_continue = False
            if not round_result["success"]:
                state.stop_reason = "优化策略已收敛或无效果"
            else:
                state.stop_reason = "代码质量达到最优"
            state.logs.append(f"   🛑 停止自优化: {state.stop_reason}")
        
    except Exception as e:
        state.errors.append(f"自优化轮次执行失败: {e}")
        state.should_continue = False
        state.stop_reason = f"自优化异常: {e}"
        state.logs.append(f"❌ 自优化轮次失败: {e}")
    
    return state


def validate_optimization(state: State) -> State:
    """
    验证优化结果
    
    确保优化后的系统仍然正常工作
    """
    if not hasattr(state, 'orchestrator') or state.orchestrator is None:
        state.errors.append("无法验证优化：编排器不可用")
        return state
    
    state.logs.append("🧪 验证自优化结果...")
    
    try:
        # 运行自验证
        validation_result = state.orchestrator.self_validate()
        
        # 记录验证结果
        state.validation_result = validation_result
        
        if validation_result["success"]:
            state.logs.append(f"✅ 自验证通过: {validation_result['tests_passed']} 项测试")
            state.optimization_success = True
        else:
            state.logs.append(f"⚠️ 自验证部分失败: {validation_result['tests_failed']} 项测试")
            state.optimization_success = False
            state.errors.append("自验证发现异常")
            
            # 显示失败的测试
            for test_result in validation_result["test_results"]:
                if test_result["status"] != "PASSED":
                    state.logs.append(f"   ❌ {test_result['name']}: {test_result.get('error', 'failed')}")
        
    except Exception as e:
        state.errors.append(f"自验证过程失败: {e}")
        state.optimization_success = False
        state.logs.append(f"❌ 自验证失败: {e}")
    
    return state


def create_self_optimization_report(state: State) -> State:
    """
    生成自优化报告
    
    总结整个自优化过程的结果
    """
    state.logs.append("📋 生成自优化报告...")
    
    # 收集统计信息
    total_rounds = len(state.optimization_rounds)
    total_optimizations = state.total_optimizations_applied
    total_issues = state.total_issues_found
    total_files = state.total_files_analyzed
    
    # 生成报告
    report_sections = [
        "# 🤖 系统自优化报告",
        "",
        f"## 📊 优化统计",
        f"- **优化轮数**: {total_rounds}",
        f"- **分析文件数**: {total_files}",
        f"- **发现问题数**: {total_issues}",
        f"- **应用优化数**: {total_optimizations}",
        f"- **使用策略**: {', '.join(state.strategies_used) if state.strategies_used else '无'}",
        f"- **优化成功**: {'✅ 是' if state.optimization_success else '❌ 否'}",
    ]
    
    # 添加各轮详情
    if state.optimization_rounds:
        report_sections.extend([
            "",
            "## 🔄 各轮详情"
        ])
        
        for i, round_result in enumerate(state.optimization_rounds, 1):
            status = "✅ 成功" if round_result["success"] else "⚠️ 收敛"
            report_sections.append(
                f"### 第 {i} 轮 - {status}"
            )
            report_sections.append(f"- 发现问题: {round_result['issues_found']}")
            report_sections.append(f"- 应用优化: {round_result['optimizations_applied']}")
            
            if round_result["strategies_used"]:
                strategies_str = ', '.join(round_result["strategies_used"])
                report_sections.append(f"- 使用策略: {strategies_str}")
            
            if round_result.get("errors"):
                report_sections.append(f"- 错误: {len(round_result['errors'])} 个")
    
    # 添加验证结果
    if hasattr(state, 'validation_result') and state.validation_result:
        val_result = state.validation_result
        report_sections.extend([
            "",
            "## 🧪 自验证结果",
            f"- **测试通过**: {val_result['tests_passed']}",
            f"- **测试失败**: {val_result['tests_failed']}",
            f"- **验证成功**: {'✅ 是' if val_result['success'] else '❌ 否'}"
        ])
        
        if val_result.get("test_results"):
            report_sections.append("### 测试详情")
            for test in val_result["test_results"]:
                status_emoji = "✅" if test["status"] == "PASSED" else "❌"
                report_sections.append(f"- {status_emoji} {test['name']}: {test['status'].lower()}")
    
    # 添加结论
    report_sections.extend([
        "",
        "## 🎯 结论"
    ])
    
    if state.optimization_success and total_optimizations > 0:
        report_sections.extend([
            "🎉 自优化循环成功完成！",
            "- 系统成功优化了自己的代码",
            "- 优化后系统功能保持正常",
            "- 代码质量得到显著提升"
        ])
    elif total_optimizations == 0:
        report_sections.extend([
            "💡 代码质量已经很好",
            "- 发现的问题不足以触发优化",
            "- 系统不需要进一步的优化"
        ])
    else:
        report_sections.extend([
            "⚠️ 自优化过程中遇到困难",
            "- 需要进一步检查和调整优化策略",
            "- 建议手动验证系统功能"
        ])
    
    # 生成报告内容
    report_content = '\n'.join(report_sections)
    
    # 保存报告
    try:
        prompt_dir = os.path.join(state.project_path or ".", "prompt")
        os.makedirs(prompt_dir, exist_ok=True)
        report_file = os.path.join(prompt_dir, "self_optimization_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        state.logs.append(f"📄 自优化报告已保存: {report_file}")
        state.report_file = report_file
        
    except Exception as e:
        state.errors.append(f"保存报告失败: {e}")
        state.logs.append(f"❌ 保存报告失败: {e}")
    
    # 添加简短的摘要到日志
    state.logs.append("=" * 60)
    state.logs.append("🏆 自优化总结:")
    state.logs.append(f"   执行轮数: {total_rounds}")
    state.logs.append(f"   优化数量: {total_optimizations}")
    state.logs.append(f"   验证结果: {'通过' if state.optimization_success else '失败'}")
    state.logs.append(f"   成功状态: {'✅ 成功' if state.optimization_success else '⚠️ 部分'}")
    state.logs.append("=" * 60)
    
    return state