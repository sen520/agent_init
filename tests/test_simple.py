#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pydantic
    print("✅ pydantic 已安装")
except ImportError:
    print("❌ pydantic 未安装")
    
try:
    import langgraph
    print("✅ langgraph 已安装")
except ImportError:
    print("❌ langgraph 未安装")

try:
    from src.state.base import State
    from src.nodes.base import (
        initialize_project, analyze_code, plan_optimizations,
        apply_changes, verify_changes, evaluate_results, end_optimization
    )
    print("✅ 所有本地模块导入成功")
    
    # 测试State
    state = State()
    print(f"State测试: {state}")
    
    # 测试节点函数
    state = initialize_project(state)
    print(f"初始化后: {state}")
    
    print("✅ 所有基本功能正常")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()