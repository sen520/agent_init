#!/usr/bin/env python3
"""
自优化代码助手 - 命令行界面
"""
import argparse
import asyncio
import sys
import os
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.base import State
from src.graph.base import optimization_app, create_optimization_workflow
from src.self_optimizing.orchestrator import run_self_optimization
from src.strategies.optimization_strategies import CodeOptimizer

logger = logging.getLogger(__name__)


def analyze_command(args):
    """分析命令"""
    logger.info(f"🔍 分析项目: {args.path}")
    
    async def run_analysis():
        try:
            state = State(project_path=args.path)
            result = await optimization_app.ainvoke(state)
            
            logger.info("📊 分析结果:")
            logger.info(f"   📁 分析文件数: {result['total_files_analyzed']}")
            logger.info(f"   🔍 发现问题数: {result['total_issues_found']}")
            logger.info(f"   📝 变更记录数: {len(result['applied_changes'])}")
            
            if result['errors']:
                logger.warning(f"\n⚠️  错误信息:")
                for error in result['errors'][:5]:
                    logger.warning(f"   • {error}")
            
            logger.info("🏆 分析完成！")
            
        except Exception as e:
            logger.error(f"❌ 分析失败: {e}")
            return 1
        return 0
    
    return asyncio.run(run_analysis())


def optimize_command(args):
    """优化命令"""
    logger.info(f"🔧 优化项目: {args.path}")
    
    async def run_optimization():
        try:
            app = create_optimization_workflow()
            state = State(
                project_path=args.path,
                strategies_to_apply=args.strategies if args.strategies else None
            )
            
            result = await app.ainvoke(state)
            
            logger.info("📊 优化结果:")
            logger.info(f"   📁 分析文件数: {result['total_files_analyzed']}")
            logger.info(f"   🔍 发现问题数: {result['total_issues_found']}")
            logger.info(f"   🔧 应用优化数: {result['total_optimizations_applied']}")
            logger.info(f"   📝 变更记录数: {len(result['applied_changes'])}")
            
            if result['applied_changes']:
                logger.info("✨ 应用的主要优化:")
                for i, change in enumerate(result['applied_changes'][:10], 1):
                    logger.info(f"   {i}. {change}")
            
            logger.info("🏆 优化完成！")
            
        except Exception as e:
            logger.error(f"❌ 优化失败: {e}")
            return 1
        return 0
    
    return asyncio.run(run_optimization())


def self_opt_command(args):
    """自优化命令"""
    logger.info(f"🤖 自优化项目: {args.path}")
    
    try:
        result = run_self_optimization(args.path)
        
        opt_result = result["optimization"]
        val_result = result["validation"]
        
        logger.info("📊 自优化结果:")
        logger.info(f"   🔄 优化轮数: {opt_result['total_rounds']}")
        logger.info(f"   📁 分析文件数: {opt_result['total_files_analyzed']}")
        logger.info(f"   🔍 发现问题数: {opt_result['total_issues_found']}")
        logger.info(f"   🔧 应用优化数: {opt_result['total_optimizations_applied']}")
        
        logger.info("📊 验证结果:")
        logger.info(f"   ✅ 测试通过: {val_result['tests_passed']}")
        logger.info(f"   ❌ 测试失败: {val_result['tests_failed']}")
        logger.info(f"   🎯 验证成功: {'是' if val_result['success'] else '否'}")
        
        # 显示报告文件
        if hasattr(val_result, 'report_file'):
            logger.info(f"📄 详细报告: {val_result.get('report_file', 'self_optimization_report.md')}")
        
        logger.info("🏆 自优化完成！")
        
    except Exception as e:
        logger.error(f"❌ 自优化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


def strategies_command(args):
    """策略列表命令"""
    logger.info("🎛️  可用优化策略:")
    
    try:
        optimizer = CodeOptimizer()
        strategies = optimizer.strategies
        
        for i, strategy in enumerate(strategies, 1):
            logger.info(f"{i}. **{strategy.name}**")
            logger.info(f"   描述: {strategy.description}")
            logger.info(f"   类型: {type(strategy).__name__}")
        
        logger.info(f"📊 总计: {len(strategies)} 种优化策略")
        
        logger.info("💡 使用建议:")
        logger.info("   • 安全策略: comment_optimizer, empty_line_optimizer")
        logger.info("   • 标准策略: import_optimizer, line_length_optimizer")  
        logger.info("   • 高级策略: function_length_optimizer, variable_naming_optimizer")
        logger.info("   • 检测策略: duplicate_code_optimizer")
        
    except Exception as e:
        logger.error(f"❌ 获取策略列表失败: {e}")
        return 1
    
    return 0


def demo_command(args):
    """演示命令"""
    logger.info("🎬 运行演示...")
    
    try:
        # 清理可能的演示文件
        demo_files = ["src/self_optimizing/demo_target.py", "demo_target.py.backup"]
        for file in demo_files:
            if os.path.exists(file):
                os.remove(file)
        
        # 运行自优化演示
        from test_self_optimization_demo import self_optimization_demo
        self_optimization_demo()
        
    except Exception as e:
        logger.error(f"❌ 演示失败: {e}")
        return 1
    
    return 0


def main():
    """主命令行入口"""
    parser = argparse.ArgumentParser(
        prog="soa", 
        description="🤖 自优化代码助手 - 智能代码优化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  soa analyze .                    # 分析当前项目
  soa optimize --strategies comment_optimizer,line_length_optimizer .
  soa self-opt .                    # 运行自优化
  soa strategies                    # 查看可用策略
  soa demo                          # 运行演示
        """
    )
    
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 分析命令
    analyze_parser = subparsers.add_parser("analyze", help="分析项目代码质量")
    analyze_parser.add_argument("path", default=".", nargs="?", help="项目路径 (默认: 当前目录)")
    analyze_parser.set_defaults(func=analyze_command)
    
    # 优化命令
    optimize_parser = subparsers.add_parser("optimize", help="优化项目代码")
    optimize_parser.add_argument("path", default=".", nargs="?", help="项目路径 (默认: 当前目录)")
    optimize_parser.add_argument(
        "--strategies", 
        help="指定优化策略 (逗号分隔)",
        choices=[
            "comment_optimizer", "empty_line_optimizer", "import_optimizer",
            "line_length_optimizer", "function_length_optimizer", 
            "variable_naming_optimizer", "duplicate_code_optimizer"
        ]
    )
    optimize_parser.set_defaults(func=optimize_command)
    
    # 自优化命令
    self_opt_parser = subparsers.add_parser("self-opt", help="运行自优化循环")
    self_opt_parser.add_argument("path", default=".", nargs="?", help="项目路径 (默认: 当前目录)")
    self_opt_parser.set_defaults(func=self_opt_command)
    
    # 策略列表命令
    strategies_parser = subparsers.add_parser("strategies", help="显示可用优化策略")
    strategies_parser.set_defaults(func=strategies_command)
    
    # 演示命令
    demo_parser = subparsers.add_parser("demo", help="运行功能演示")
    demo_parser.set_defaults(func=demo_command)
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有命令，显示帮助
    if args.command is None:
        parser.print_help()
        return 0
    
    # 执行命令
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())