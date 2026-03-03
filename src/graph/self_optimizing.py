#!/usr/bin/env python3
"""
自优化LangGraph工作流 - 实现系统自优化闭环的核心工作流
"""
from langgraph.graph import StateGraph, END

# 导入节点
from src.nodes.base import end_optimization
from src.nodes.self_optimizing import (
    start_self_optimization,
    run_optimization_round,
    validate_optimization,
    create_self_optimization_report
)

# 导入状态管理
from src.state.base import State


def build_self_optimizing_workflow():
    """构建自优化工作流"""
    
    # 创建状态图
    builder = StateGraph(State)
    
    # 添加节点
    builder.add_node("start", start_self_optimization)
    builder.add_node("optimize", run_optimization_round)
    builder.add_node("validate", validate_optimization)
    builder.add_node("report", create_self_optimization_report)
    builder.add_node("end", end_optimization)
    
    # 设置入口点
    builder.set_entry_point("start")
    
    # 添加边
    builder.add_edge("start", "optimize")
    
    # 条件边：决定是否继续优化
    builder.add_conditional_edges(
        "optimize",
        lambda state: "continue" if state.should_continue else "validate",
        {
            "continue": "optimize",      # 继续下一轮优化
            "validate": "validate"       # 结束优化，进行验证
        }
    )
    
    builder.add_edge("validate", "report")
    builder.add_edge("report", "end")
    
    # 设置结束点
    builder.set_finish_point("end")
    
    return builder.compile()


# 创建工作流实例
self_optimizing_app = build_self_optimizing_workflow()


if __name__ == "__main__":
    """测试自优化工作流"""
    import asyncio
    from src.state.base import State
    
    async def test_self_optimizing_workflow():
        print("🤖 测试自优化工作流")
        print("=" * 60)
        
        try:
            # 创建初始状态 - 设置最大迭代次数以防无限循环
            initial_state = State(
                project_path=".",
                max_iterations=3  # 限制自优化轮数
            )
            
            print("🚀 启动自优化工作流...")
            
            # 运行工作流
            result_dict = await self_optimizing_app.ainvoke(initial_state)
            final_state = State(**result_dict)
            
            # 显示结果
            print("✅ 自优化工作流完成")
            print(f"   总轮数: {len(final_state.optimization_rounds)}")
            print(f"   分析文件: {final_state.total_files_analyzed}")
            print(f"   发现问题: {final_state.total_issues_found}")
            print(f"   应用优化: {final_state.total_optimizations_applied}")
            print(f"   使用策略: {', '.join(final_state.strategies_used) if final_state.strategies_used else '无'}")
            print(f"   优化成功: {'✅' if final_state.optimization_success else '❌'}")
            
            # 显示关键日志
            print("\n📋 执行摘要:")
            for log in final_state.logs[-8:]:
                print(f"   • {log}")
            
            # 显示报告文件
            if hasattr(final_state, 'report_file'):
                print(f"\n📄 详细报告: {final_state.report_file}")
            
            return final_state
            
        except Exception as e:
            print(f"❌ 自优化工作流失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 运行测试
    result = asyncio.run(test_self_optimizing_workflow())
    
    if result and result.optimization_success:
        print("\n🎉 自优化系统成功运行！")
    elif result:
        print("\n🤔 自优化系统运行完成，但有一些问题")
    else:
        print("\n😞 自优化系统遇到问题")