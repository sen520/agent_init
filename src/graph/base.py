#!/usr/bin/env python3
"""
LangGraph工作流定义 - 代码优化助手的核心引擎
"""
import logging
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# 导入节点函数
try:
    # 优先使用真实分析节点
    from src.nodes.real import (
        initialize_project,
        analyze_code,
        create_analysis_report,
        end_optimization
    )
    REAL_NODES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"真实分析节点不可用，使用模拟节点: {e}")
    from src.nodes.base import (
        initialize_project,
        analyze_code,
        create_analysis_report,
        end_optimization
    )
    REAL_NODES_AVAILABLE = False

# 导入优化节点
try:
    from src.nodes.optimization import (
        apply_optimization,
        analyze_optimization_results,
        create_optimization_summary
    )
    OPTIMIZATION_NODES_AVAILABLE = True
except ImportError as e:
    logger.info(f"导入优化节点失败: {e}")
    OPTIMIZATION_NODES_AVAILABLE = False

# 导入状态管理
from src.state.base import State


def build_graph():
    """构建默认的工作流图（完整优化流程）"""
    return build_self_optimizing_graph()


def build_simple_graph():
    """构建简单的测试图"""
    # 创建状态图
    builder = StateGraph(State)
    
    # 添加节点
    builder.add_node("initialize", initialize_project)
    builder.add_node("analyze", analyze_code)
    builder.add_node("create_report", create_analysis_report)
    builder.add_node("end", end_optimization)
    
    # 设置入口点
    builder.set_entry_point("initialize")
    
    # 添加边
    builder.add_edge("initialize", "analyze")
    builder.add_edge("analyze", "create_report")
    builder.add_edge("create_report", "end")
    
    # 设置结束点
    builder.set_finish_point("end")
    
    return builder.compile()


def build_optimization_graph():
    """构建完整的优化工作流"""
    # 创建状态图
    builder = StateGraph(State)
    
    # 添加基本节点
    builder.add_node("initialize", initialize_project)
    builder.add_node("analyze", analyze_code)
    builder.add_node("create_report", create_analysis_report)
    builder.add_node("end", end_optimization)
    
    if OPTIMIZATION_NODES_AVAILABLE:
        # 添加优化节点
        builder.add_node("apply_optimization", apply_optimization)
        builder.add_node("analyze_results", analyze_optimization_results)
        builder.add_node("create_summary", create_optimization_summary)
        
        # 设置入口点
        builder.set_entry_point("initialize")
        
        # 添加边 - 完整优化流程
        builder.add_edge("initialize", "analyze")
        builder.add_edge("analyze", "apply_optimization")
        builder.add_edge("apply_optimization", "analyze_results")
        
        # 条件边：根据是否继续决定下一步
        builder.add_conditional_edges(
            "analyze_results",
            lambda state: "continue" if state.should_continue else "end",
            {
                "continue": "analyze",      # 继续下一次迭代
                "end": "create_summary"     # 结束并生成总结
            }
        )
        
        builder.add_edge("create_summary", "end")
    else:
        # 如果优化节点不可用，使用简化流程
        builder.set_entry_point("initialize")
        builder.add_edge("initialize", "analyze")
        builder.add_edge("analyze", "create_report")
        builder.add_edge("create_report", "end")
    
    # 设置结束点
    builder.set_finish_point("end")
    
    return builder.compile()


def build_self_optimizing_graph():
    """构建自优化图 - 让系统能够优化自己的代码"""
    # 创建状态图
    builder = StateGraph(State)
    
    # 添加节点
    builder.add_node("initialize", initialize_project)
    builder.add_node("analyze", analyze_code)
    builder.add_node("apply_optimization", apply_optimization)
    builder.add_node("analyze_results", analyze_optimization_results)
    builder.add_node("create_summary", create_optimization_summary)
    builder.add_node("end", end_optimization)
    
    # 添加 HTML 报告生成节点（如果可用）
    try:
        from src.utils.report_generator import create_report_node
        builder.add_node("generate_report", create_report_node)
        has_report_node = True
    except ImportError:
        has_report_node = False
    
    # 设置入口点
    builder.set_entry_point("initialize")
    
    # 自优化流程
    builder.add_edge("initialize", "analyze")
    builder.add_edge("analyze", "apply_optimization")
    builder.add_edge("apply_optimization", "analyze_results")
    
    # 条件边：循环优化直到满足条件
    if has_report_node:
        builder.add_conditional_edges(
            "analyze_results",
            lambda state: "continue" if state.should_continue and state.iteration_count < 2 else "end",
            {
                "continue": "analyze",         # 继续分析（可能发现新的优化点）
                "end": "create_summary"        # 结束优化
            }
        )
        builder.add_edge("create_summary", "generate_report")
        builder.add_edge("generate_report", "end")
    else:
        builder.add_conditional_edges(
            "analyze_results",
            lambda state: "continue" if state.should_continue and state.iteration_count < 2 else "end",
            {
                "continue": "analyze",         # 继续分析（可能发现新的优化点）
                "end": "create_summary"        # 结束优化
            }
        )
        builder.add_edge("create_summary", "end")
    
    # 设置结束点
    builder.set_finish_point("end")
    
    return builder.compile()


def build_phase2_graph():
    """
    构建 Phase 2 工作流 - 集成 LLM 和测试验证
    """
    builder = StateGraph(State)
    
    # 添加基础节点
    builder.add_node("initialize", initialize_project)
    builder.add_node("analyze", analyze_code)
    builder.add_node("apply_optimization", apply_optimization)
    builder.add_node("analyze_results", analyze_optimization_results)
    builder.add_node("create_summary", create_optimization_summary)
    builder.add_node("end", end_optimization)
    
    # 尝试添加 Phase 2 节点
    try:
        from src.nodes.phase2 import llm_analyze_issues, validate_optimization, generate_llm_report
        builder.add_node("llm_analyze", llm_analyze_issues)
        builder.add_node("validate", validate_optimization)
        builder.add_node("llm_report", generate_llm_report)
        has_phase2 = True
    except ImportError as e:
        logger.info(f"Phase 2 节点不可用: {e}")
        has_phase2 = False
    
    # 设置入口点
    builder.set_entry_point("initialize")
    
    if has_phase2:
        # Phase 2 完整流程
        builder.add_edge("initialize", "analyze")
        builder.add_edge("analyze", "llm_analyze")  # LLM 分析
        builder.add_edge("llm_analyze", "apply_optimization")
        builder.add_edge("apply_optimization", "validate")  # 测试验证
        builder.add_edge("validate", "analyze_results")
        
        builder.add_conditional_edges(
            "analyze_results",
            lambda state: "continue" if state.should_continue and state.iteration_count < 2 else "end",
            {
                "continue": "analyze",
                "end": "llm_report"  # 生成 LLM 报告
            }
        )
        builder.add_edge("llm_report", "create_summary")
        builder.add_edge("create_summary", "end")
    else:
        # 回退到 Phase 1 流程
        builder.add_edge("initialize", "analyze")
        builder.add_edge("analyze", "apply_optimization")
        builder.add_edge("apply_optimization", "analyze_results")
        builder.add_conditional_edges(
            "analyze_results",
            lambda state: "continue" if state.should_continue else "end",
            {
                "continue": "analyze",
                "end": "create_summary"
            }
        )
        builder.add_edge("create_summary", "end")
    
    builder.set_finish_point("end")
    return builder.compile()


# 创建可用的图实例
simple_app = build_simple_graph()
if OPTIMIZATION_NODES_AVAILABLE:
    optimization_app = build_optimization_graph()
    self_optimizing_app = build_self_optimizing_graph()
else:
    optimization_app = simple_app
    self_optimizing_app = simple_app


if __name__ == "__main__":
    """测试工作流"""
    import asyncio
    from src.state.base import State
    
    # 测试简单工作流
    logger.info("🧪 测试LangGraph工作流")
    logger.info("=" * 60)
    
    async def test_workflow(workflow_name, app, initial_state):
        logger.info(f"\n🔄 测试 {workflow_name}:")
        try:
            result_dict = await app.ainvoke(initial_state)
            final_state = State(**result_dict)
            
            logger.info(f"✅ {workflow_name} 完成")
            logger.info(f"   迭代次数: {final_state.iteration_count}")
            logger.info(f"   日志数量: {len(final_state.logs)}")
            logger.info(f"   错误数量: {len(final_state.errors)}")
            
            # 显示最后几条日志
            if final_state.logs:
                logger.info("   最近日志:")
                for log in final_state.logs[-3:]:
                    logger.info(f"     {log}")
                    
            return final_state
            
        except Exception as e:
            logger.info(f"❌ {workflow_name} 失败: {e}")
            return None
    
    # 运行测试
    async def run_tests():
        initial_state = State(project_path=".")
        
        # 测试1: 简单工作流
        result1 = await test_workflow("简单工作流", simple_app, initial_state)
        
        # 测试2: 优化工作流（如果可用）
        if OPTIMIZATION_NODES_AVAILABLE:
            initial_state2 = State(project_path=".")
            result2 = await test_workflow("优化工作流", optimization_app, initial_state2)
            
            # 测试3: 自优化工作流
            initial_state3 = State(project_path=".")
            result3 = await test_workflow("自优化工作流", self_optimizing_app, initial_state3)
        else:
            logger.info("⚠️ 优化节点不可用，跳过优化工作流测试")
    
    asyncio.run(run_tests())
    
    logger.info("\n✅ 工作流测试完成")
    logger.info(f"📊 可用的工作流:")
    logger.info(f"   - 简单工作流 (�)")
    if OPTIMIZATION_NODES_AVAILABLE:
        logger.info(f"   - 优化工作流 (�)")
        logger.info(f"   - 自优化工作流 (�)")
    else:
        logger.info(f"   - 优化工作流 (❌ - 节点不可用)")
        logger.info(f"   - 自优化工作流 (❌ - 节点不可用)")