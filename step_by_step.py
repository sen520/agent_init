#!/usr/bin/env python3
"""
分步执行工作流 - 每一步都可视化展示
"""
import asyncio
import sys
import os

# 确保路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.state.base import State
from src.nodes.base import (
    initialize_project, analyze_code, plan_optimizations,
    apply_changes, verify_changes, evaluate_results, end_optimization
)
from src.graph.base import build_graph


def print_step_header(step_name):
    """打印步骤标题"""
    print(f"\n{'='*60}")
    print(f"📌 步骤: {step_name}")
    print('='*60)


def print_state_summary(state):
    """打印状态摘要"""
    print(f"  迭代次数: {state.iteration_count}")
    print(f"  项目路径: {state.project_path}")
    print(f"  发现问题: {len(state.analysis.issues)} 个")
    if state.plan.suggestions:
        print(f"  优化建议: {len(state.plan.suggestions)} 条")
    if state.applied_changes:
        print(f"  已应用变更: {len(state.applied_changes)} 项")
    print(f"  是否继续: {state.should_continue}")


async def manual_step_execution():
    """手动分步执行每个节点"""
    print("🤖 代码自我优化助手 - 分步执行")
    print("="*60)
    
    # 初始化状态
    state = State()
    print("💭 初始状态创建完成")
    
    # 步骤1: 初始化项目
    print_step_header("1. 初始化项目")
    state = initialize_project(state)
    print_state_summary(state)
    
    # 步骤2: 分析代码
    print_step_header("2. 分析代码")
    state = analyze_code(state)
    print_state_summary(state)
    print(f"  具体问题:")
    for i, issue in enumerate(state.analysis.issues, 1):
        print(f"    {i}. {issue}")
    
    await asyncio.sleep(0.5)  # 小延迟，让输出更清晰
    
    # 步骤3: 生成优化计划
    print_step_header("3. 生成优化计划")
    state = plan_optimizations(state)
    print_state_summary(state)
    print(f"  具体建议:")
    for i, (suggestion, priority) in enumerate(zip(state.plan.suggestions, state.plan.priorities), 1):
        print(f"    {i}. [优先级:{priority}] {suggestion}")
    
    # 步骤4: 应用变更
    print_step_header("4. 应用变更")
    state = apply_changes(state)
    print_state_summary(state)
    if state.applied_changes:
        print(f"  应用的具体变更:")
        for i, change in enumerate(state.applied_changes, 1):
            print(f"    {i}. {change}")
    
    # 步骤5: 验证变更
    print_step_header("5. 验证变更")
    state = verify_changes(state)
    print_state_summary(state)
    print(f"  验证结果:")
    for key, value in state.verification_results.items():
        print(f"    ✅ {key}: {value}")
    
    # 步骤6: 评估结果
    print_step_header("6. 评估结果")
    state = evaluate_results(state)
    print_state_summary(state)
    
    # 步骤7: 结束优化
    print_step_header("7. 结束优化")
    state = end_optimization(state)
    
    return state


async def single_iteration_test():
    """测试单次完整迭代"""
    print("\n🔁 现在测试完整的单次迭代...")
    print("="*60)
    
    # 使用完整的图
    app = build_graph()
    state = State()
    
    # 运行单次迭代
    result = await app.ainvoke(state, {"configurable": {"thread_id": "test1"}})
    
    print("\n✅ 单次迭代完成!")
    print("="*60)
    print(f"最终状态:")
    print(f"  迭代次数: {result.iteration_count}")
    print(f"  应用优化: {len(result.applied_changes)} 项")
    print(f"  剩余问题: {len(result.analysis.issues)} 个")
    print(f"  是否继续: {result.should_continue}")
    
    return result


async def main():
    """主函数"""
    print("选择执行模式:")
    print("1. 手动分步执行")
    print("2. 运行完整工作流")
    print("3. 测试循环迭代")
    
    try:
        choice = int(input("请输入选择 (1-3): "))
    except:
        choice = 1
    
    if choice == 1:
        # 模式1: 手动分步
        await manual_step_execution()
    elif choice == 2:
        # 模式2: 完整工作流
        await single_iteration_test()
    elif choice == 3:
        # 模式3: 测试循环
        print("\n🔄 测试循环迭代功能...")
        app = build_graph()
        state = State()
        
        # 运行多次迭代
        max_iterations = 3
        for i in range(max_iterations):
            print(f"\n🌀 迭代 {i+1}:")
            state = await app.ainvoke(state, {"configurable": {"thread_id": f"loop_{i}"}})
            print(f"  迭代完成: 应用了 {len(state.applied_changes)} 个变更")
            print(f"  剩余问题: {len(state.analysis.issues)} 个")
            
            if not state.should_continue:
                print("  工作流决定终止优化")
                break
    else:
        await manual_step_execution()


if __name__ == "__main__":
    asyncio.run(main())