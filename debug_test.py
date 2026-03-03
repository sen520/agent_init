#!/usr/bin/env python3
"""调试测试脚本"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def debug_main():
    from src.graph.base import build_simple_graph
    from src.state.base import State
    
    print("🔧 调试测试...")
    print("=" * 60)
    
    app = build_simple_graph()
    initial_state = State()
    
    print(f"初始状态类型: {type(initial_state)}")
    print(f"初始状态迭代: {initial_state.iteration_count}")
    
    # 检查app的类型
    print(f"app类型: {type(app)}")
    
    # 运行工作流
    result = await app.ainvoke(initial_state)
    
    print(f"\n结果类型: {type(result)}")
    print(f"结果内容: {result}")
    
    # 检查结果属性
    if isinstance(result, dict):
        print("\n⚠️ 结果是字典类型")
        print(f"迭代次数: {result.get('iteration_count', 'Not found')}")
        print(f"应用变更: {result.get('applied_changes', 'Not found')}")
    else:
        print(f"\n✅ 结果是State类型: {type(result)}")
        print(f"迭代次数: {result.iteration_count}")
        print(f"应用变更: {result.applied_changes}")
        
    return result

if __name__ == '__main__':
    import asyncio
    asyncio.run(debug_main())