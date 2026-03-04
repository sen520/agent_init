
from dotenv import load_dotenv
from src.config.settings import settings
from src.graph.base import build_graph, build_simple_graph, build_phase2_graph
from src.state.base import State
from src.utils.logger import create_logger
from src.utils.logging_config import setup_colored_logging, get_logger
import asyncio
import logging
import os
import sys

# 确保能够正确导入src模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


load_dotenv()

# 设置统一日志
setup_colored_logging('INFO')
logger = get_logger(__name__)


async def run_full_workflow():
    """运行完整的工作流"""
    logger.info("🚀 启动自我优化工作流...")
    print("=" * 60)
    
    # 构建图
    agent = build_graph()
    
    # 初始化状态
    initial_state = State()
    
    # 执行工作流
    logger.debug(f"初始状态: {initial_state.model_dump_json(indent=2)[:500]}...")
    print("-" * 60)
    
    result = await agent.ainvoke(initial_state)
    
    print("-" * 60)
    logger.info("工作流执行完成!")
    if isinstance(result, dict):
        logger.info(f"最终状态: {result}")
    else:
        logger.info(f"最终状态: {result.model_dump_json(indent=2)[:500]}...")
    
    return result


async def run_simple_test():
    """运行简化测试"""
    logger.info("🧪 运行简化测试...")
    print("=" * 60)
    
    agent = build_simple_graph()
    initial_state = State()
    
    result = await agent.ainvoke(initial_state)
    
    print("-" * 60)
    logger.info("✅ 简化测试通过!")
    if isinstance(result, dict):
        logger.info("✅ 工作流执行完成")
        logger.info(f"结果类型: {type(result)}")
        if isinstance(result, dict) and 'iteration_count' in result:
            logger.info(f"迭代次数: {result['iteration_count']}")
    else:
        logger.info(f"最终状态: {result.model_dump_json(indent=2)[:200]}...")
    
    return result


async def run_phase2_workflow():
    """运行 Phase 2 工作流（集成 LLM 和测试验证）"""
    logger.info("🚀 启动 Phase 2 智能优化工作流...")
    logger.info("🧠 集成 LLM 智能分析和测试验证")
    print("=" * 60)
    
    agent = build_phase2_graph()
    initial_state = State()
    
    logger.info("开始执行...")
    print("-" * 60)
    
    result = await agent.ainvoke(initial_state)
    
    print("-" * 60)
    logger.info("✅ Phase 2 工作流执行完成!")
    
    if isinstance(result, dict):
        # 显示 LLM 建议
        if 'llm_suggestions' in result and result['llm_suggestions']:
            logger.info("📝 LLM 智能建议:")
            for suggestion in result['llm_suggestions'][:3]:
                logger.info(f"  📄 {suggestion['file']}")
        
        # 显示验证结果
        if 'validation_result' in result and result['validation_result']:
            validation = result['validation_result']
            if validation.get('success'):
                logger.info("✅ 验证通过")
            else:
                logger.warning("⚠️ 验证未完全通过")
    
    return result


async def main():
    """主函数"""
    print("🤖 代码自我优化助手 v0.2")
    print("=" * 60)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "test"  # 默认运行测试
    
    logger.info(f"启动模式: {mode}")
    
    if mode == "test":
        # 运行简化测试
        await run_simple_test()
    elif mode == "full":
        # 运行完整工作流
        await run_full_workflow()
    elif mode == "phase2":
        # 运行 Phase 2 工作流
        await run_phase2_workflow()
    elif mode == "help":
        print("使用说明:")
        print("  python main.py test      # 运行简化测试（默认）")
        print("  python main.py full      # 运行完整工作流")
        print("  python main.py phase2    # 运行 Phase 2 智能优化")
        print("  python main.py help      # 显示帮助信息")
    else:
        logger.error(f"未知模式: {mode}")
        print("使用 'python main.py help' 查看帮助")

    # 创建日志记录器
    app_logger = create_logger("self_optimizing_assistant")
    app_logger.info(f"工作流执行完成，模式: {mode}")


if __name__ == '__main__':
    asyncio.run(main())