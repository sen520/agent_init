#!/usr/bin/env python3
"""测试修复后的代码"""

import asyncio
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import create_logger
from src.graph.base import build_simple_graph
from src.state.base import State

load_dotenv()


async def test_fix():
    """测试修复"""
    print("🧪 测试节点修复...")
    print("=" * 60)
    
    # 创建简化工作流
    app = build_simple_graph()
    initial_state = State(project_path=os.path.dirname(__file__))
    
    print("✅ 工作流构建成功")
    print(f"   初始状态类型: {type(initial_state)}")
    print(f"   项目路径: {initial_state.project_path}")
    
    # 只运行前两个节点测试
    try:
        result = await app.ainvoke(initial_state)
        print("✅ 工作流执行成功!")
        print(f"   最终状态: {result.model_dump_json(indent=2)[:300]}...")
        
        print("\n📊 分析结果:")
        print(f"   总文件数: {result.analysis.total_files_analyzed}")
        print(f"   总代码行数: {result.analysis.total_lines_of_code}")
        print(f"   平均复杂度: {result.analysis.average_complexity}")
        
        return True
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = asyncio.run(test_fix())
    if success:
        print("\n🎉 修复测试通过!")
    else:
        print("\n🚨 修复测试失败!")
        sys.exit(1)