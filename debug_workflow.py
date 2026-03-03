#!/usr/bin/env python3
"""
工作流调试脚本 - 逐步定位问题
"""
import sys
import os

# 设置路径
sys.path.insert(0, os.getcwd())

print("🔍 工作流调试")
print("=" * 60)

def step_test(step_name, test_func):
    """逐步测试函数"""
    try:
        result = test_func()
        print(f"✅ {step_name}: 成功")
        return result
    except Exception as e:
        print(f"❌ {step_name}: 失败 - {e}")
        import traceback
        traceback.print_exc()
        return None

# 测试步骤
print("1. 测试基本Python环境...")
def test_basic():
    import ast
    import os
    return True

step_test("基本环境", test_basic)

print()

print("2. 测试核心模块导入...")
def test_imports():
    from src.state.base import State
    from src.nodes.base import initialize_project, analyze_code
    from langgraph.graph import StateGraph
    return True

step_test("模块导入", test_imports)

print()

print("3. 测试状态创建...")
def test_state():
    from src.state.base import State
    state = State()
    print(f"   State创建: {state}")
    return state

state_obj = step_test("状态创建", test_state)

print()

print("4. 测试单个节点函数...")
def test_single_node():
    if state_obj is None:
        return None
    from src.nodes.base import initialize_project
    result = initialize_project(state_obj)
    print(f"   初始化结果: logs={len(result.logs)}")
    return result

node_result = step_test("节点函数", test_single_node)

print()

print("5. 测试工作流构建...")
def test_workflow_build():
    if state_obj is None:
        return None
    from src.nodes.base import initialize_project, analyze_code, create_analysis_report, end_optimization
    from src.state.base import State
    from langgraph.graph import StateGraph, END
    
    # 构建工作流
    builder = StateGraph(State)
    builder.add_node("initialize", initialize_project)
    builder.add_node("analyze", analyze_code)
    builder.add_node("create_report", create_analysis_report)
    builder.add_node("end", end_optimization)
    
    builder.set_entry_point("initialize")
    builder.add_edge("initialize", "analyze")
    builder.add_edge("analyze", "create_report")
    builder.add_edge("create_report", "end")
    builder.set_finish_point("end")
    
    app = builder.compile()
    return app

workflow_app = step_test("工作流构建", test_workflow_build)

print()

print("6. 测试工作流执行...")
async def test_workflow_run():
    if workflow_app is None:
        return None
    
    from src.state.base import State
    
    # 创建初始状态
    initial_state = State(project_path=".")
    
    # 运行工作流
    print("   开始执行工作流...")
    result_dict = await workflow_app.ainvoke(initial_state)
    final_state = State(**result_dict)
    
    print(f"   执行成功 - 迭代: {final_state.iteration_count}, 日志: {len(final_state.logs)}")
    return final_state

import asyncio
if workflow_app is not None:
    final_result = asyncio.run(test_workflow_run())
    step_test("工作流执行", lambda: final_result)
else:
    print("❌ 跳过工作流执行测试 - 构建失败")

print()

print("🏁 调试完成")
if workflow_app is not None:
    print("✅ 工作流基本可用")
else:
    print("❌ 工作流构建存在问题")