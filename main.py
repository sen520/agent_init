import asyncio
import sys
import os
from dotenv import load_dotenv

# 确保能够正确导入src模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import create_logger
from src.graph.base import build_graph, build_simple_graph
from src.config.settings import settings
from src.state.base import State

load_dotenv()


async def run_full_workflow():
    """运行完整的工作流"""
    print("🚀 启动自我优化工作流...")
    print("=" * 60)
    
    # 构建图
    agent = build_graph()
    
    # 初始化状态
    initial_state = State()
    
    # 执行工作流
    print(f"开始执行，初始状态: {initial_state.model_dump_json(indent=2)}")
    print("-" * 60)
    
    result = await agent.ainvoke(initial_state)
    
    print("-" * 60)
    print(f"工作流执行完成!")
    print(f"最终状态: {result.model_dump_json(indent=2)}")
    
    return result


async def run_simple_test():
    """运行简化测试"""
    print("🧪 运行简化测试...")
    print("=" * 60)
    
    agent = build_simple_graph()
    initial_state = State()
    
    result = await agent.ainvoke(initial_state)
    
    print("-" * 60)
    print("✅ 简化测试通过!")
    if isinstance(result, dict):
        print("✅ 工作流执行完成")
        print(f"结果类型: {type(result)}")
        if isinstance(result, dict) and 'iteration_count' in result:
            print(f"迭代次数: {result['iteration_count']}")
    else:
        print(f"最终状态: {result.model_dump_json(indent=2)[:200]}...")
    
    return result


async def main():
    """主函数"""
    print("🤖 代码自我优化助手 v0.1")
    print("=" * 60)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "test"  # 默认运行测试
    
    if mode == "test":
        # 运行简化测试
        await run_simple_test()
    elif mode == "full":
        # 运行完整工作流
        await run_full_workflow()
    elif mode == "help":
        print("使用说明:")
        print("  python main.py test      # 运行简化测试（默认）")
        print("  python main.py full      # 运行完整工作流")
        print("  python main.py help      # 显示帮助信息")
    else:
        print(f"未知模式: {mode}")
        print("使用 'python main.py help' 查看帮助")

    # 创建日志记录器
    logger = create_logger()
    logger.info(f"工作流执行完成，模式: {mode}")


if __name__ == '__main__':
    asyncio.run(main())
