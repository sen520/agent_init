#!/usr/bin/env python3
"""
测试脚本 - 使用绝对路径
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_main():
    from src.graph.base import build_simple_graph
    from src.state.base import State
    
    print("🧪 开始测试简化工作流...")
    print("=" * 60)
    
    app = build_simple_graph()
    initial_state = State()
    
    result = await app.ainvoke(initial_state)
    
    print("-" * 60)
    print("✅ 简化测试通过!")
    print(f"最终状态迭代次数: {result.iteration_count}")
    print(f"应用优化数量: {len(result.applied_changes)}")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())