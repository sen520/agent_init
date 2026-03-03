#!/usr/bin/env python3
"""
演示优化策略的具体效果
"""
import sys
import os

# 设置路径
sys.path.insert(0, os.getcwd())

def create_test_file():
    """创建一个有各种问题的测试文件"""
    test_content = '''"""测试文件 - 包含各种需要优化的问题"""

import os,sys,re
from typing import Dict, List
import math

#TODO: This is a todo
def very_long_function_that_should_be_split():#FIXME: add docstring
    x=1
    if x>0:
        print("test")
    if x<10:
        print("test2")    
    if x>0:
        print("test")
    if x<10:
        print("test2")    
    if x>0:
        print("test")
    if x<10:
        print("test2")    
    if x>0:
        print("test")
    if x<10:
        print("test2")    
    if x>0:
        print("test")
    if x<10:
        print("test2")    
    return x

class GoodClass:
    
    def method1(self):
        a=5
        return a
        
#
#
#
    def method2(self):
        return 2

def another_function():
    pass
'''
    
    test_file = 'temp_test_file.py'
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    return test_file

def demo_optimization():
    """演示优化效果"""
    print("🎬 代码优化效果演示")
    print("=" * 60)
    
    # 创建测试文件
    test_file = create_test_file()
    print(f"📝 创建测试文件: {test_file}")
    
    # 显示原始内容
    with open(test_file, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    print(f"\n📋 原始内容 ({len(original_content.splitlines())} 行):")
    print("-" * 40)
    for i, line in enumerate(original_content.splitlines(), 1):
        print(f"{i:2d}: {line}")
    
    # 导入优化器
    from src.strategies.optimization_strategies import CodeOptimizer
    
    optimizer = CodeOptimizer()
    print(f"\n🔧 可用策略数: {len(optimizer.strategies)}")
    
    # 分析问题
    analysis = optimizer.analyze_file(test_file)
    issues_found = analysis.get('total_issues', 0)
    
    print(f"\n🔍 发现 {issues_found} 个问题:")
    strategy_results = analysis.get('strategy_results', {})
    
    for strategy_name, result in strategy_results.items():
        if result.get('can_optimize', False):
            issue_count = result.get('issues_found', 0)
            print(f"   • {strategy_name}: {issue_count} 个问题")
    
    # 应用优化（选择几种安全的策略）
    strategies_to_apply = [
        'comment_optimizer',
        'empty_line_optimizer', 
        'import_optimizer',
        'line_length_optimizer'
    ]
    
    print(f"\n🔄 应用优化策略: {', '.join(strategies_to_apply)}")
    
    result = optimizer.optimize_file(test_file, strategies_to_apply)
    
    if result.get('optimization_applied'):
        changes_count = result.get('changes_count', 0)
        print(f"✅ 应用成功: {changes_count} 处变更")
        
        # 显示优化后的内容
        with open(test_file, 'r', encoding='utf-8') as f:
            optimized_content = f.read()
        
        print(f"\n✨ 优化后内容 ({len(optimized_content.splitlines())} 行):")
        print("-" * 40)
        for i, line in enumerate(optimized_content.splitlines(), 1):
            print(f"{i:2d}: {line}")
        
        # 具体变更详述
        changed_results = result.get('optimization_results', [])
        print(f"\n📊 变更详情:")
        for strategy_result in changed_results:
            if 'changes_count' in strategy_result:
                strategy_name = strategy_result.get('strategy', 'unknown')
                count = strategy_result.get('changes_count', 0)
                print(f"   • {strategy_name}: {count} 处变更")
        
        print(f"\n🎯 优化效果:")
        print(f"   • 格式化了TODO注释")
        print(f"   • 规范化了空行使用") 
        print(f"   • 重组了import语句")
        print(f"   • 优化了代码行长度")
        
    else:
        print(f"❌ 优化失败: {result.get('error')}")
    
    # 清理测试文件
    try:
        os.remove(test_file)
        if os.path.exists(test_file + '.backup'):
            os.remove(test_file + '.backup')
        print(f"\n🧹 清理测试文件: {test_file}")
    except:
        pass
    
    print(f"\n🎉 演示完成！")
    print(f"💡 系统现在拥有 {len(optimizer.strategies)} 种实用优化策略")

if __name__ == "__main__":
    demo_optimization()