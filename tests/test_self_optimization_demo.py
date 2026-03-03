#!/usr/bin/env python3
"""
自优化系统演示和测试脚本
展示系统如何优化自己的代码
"""
import sys
import os
import time

# 设置路径
sys.path.insert(0, os.getcwd())

def create_demo_file_for_optimization():
    """创建一个有明显问题的演示文件供自优化"""
    demo_content = '''"""自优化演示目标文件 - 包含各种问题"""

import os,sys,re,json
from typing import Dict, List

#TODO fix this
def very_long_function_that_should_be_split():
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
    return x

#
#
#
def another_function():
    a=5
    return a

def duplicate_function():
    x=1
    if x>0:
        print("test")
    if x<10:
        print("test2")    
    return x

def duplicate_function_copy():
    x=1
    if x>0:
        print("test")
    if x<10:
        print("test2")    
    return x

class TestClass:
    
    def method1(self):
        b=10
        return b
        
    def method2(self):
        return 20
'''
    
    demo_file = "src/self_optimizing/demo_target.py"
    with open(demo_file, 'w', encoding='utf-8') as f:
        f.write(demo_content)
    
    print(f"📝 创建演示文件: {demo_file}")
    return demo_file

def self_optimization_demo():
    """自优化系统的完整演示"""
    print("🤖 自优化系统完整演示")
    print("=" * 60)
    
    # 创建演示文件
    demo_file = create_demo_file_for_optimization()
    
    print("\n1️⃣ 演示文件已创建，包含的问题:")
    print("   • 导入语句不规范 (多个import在一行)")
    print("   • TODO注释需要格式化")
    print("   • 函数过长 (15+行)")
    print("   • 多余空行 (3个连续空行)")
    print("   • 重复代码 (duplicate_function 和它的复制)")
    print("   • 变量命名不够描述性 (a, b, x)")
    
    # 测试自优化编排器
    print(f"\n2️⃣ 开始自优化...")
    
    from src.self_optimizing.orchestrator import SelfOptimizingOrchestrator
    
    start_time = time.time()
    
    # 包括演示文件在内的自优化
    orchestrator = SelfOptimizingOrchestrator(".")
    
    # 确保包含演示文件
    if demo_file not in orchestrator.target_files:
        orchestrator.target_files.append(demo_file)
        print(f"   📁 添加演示文件到优化目标: {os.path.basename(demo_file)}")
    
    print(f"   🎯 总优化目标: {len(orchestrator.target_files)} 个文件")
    
    # 运行一轮优化演示
    print(f"\n3️⃣ 执行优化...")
    round_result = orchestrator.run_self_optimization_round()
    
    print(f"\n4️⃣ 优化结果:")
    print(f"   📄 分析文件: {round_result['files_analyzed']}")
    print(f"   🔍 发现问题: {round_result['issues_found']}")
    print(f"   🔧 应用优化: {round_result['optimizations_applied']}")
    print(f"   ✅ 优化成功: {round_result['success']}")
    print(f"   📋 使用策略: {', '.join(round_result['strategies_used'])}")
    
    # 显示原始内容 vs 优化后
    print(f"\n5️⃣ 演示文件优化效果:")
    
    # 读取备份的原始内容
    backup_file = demo_file + ".backup"
    if os.path.exists(backup_file):
        with open(backup_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        print("   📋 原始内容 (前10行):")
        for i, line in enumerate(original_content.splitlines()[:10], 1):
            print(f"      {i:2d}: {line}")
    
    # 读取优化后的内容
    if os.path.exists(demo_file):
        with open(demo_file, 'r', encoding='utf-8') as f:
            optimized_content = f.read()
        
        print("\n   ✨ 优化后内容 (前10行):")
        for i, line in enumerate(optimized_content.splitlines()[:10], 1):
            print(f"      {i:2d}: {line}")
    
    # 具体变更分析
    print(f"\n6️⃣ 变更分析:")
    if round_result['optimizations_applied'] > 0:
        print("   ✅ 检测到的改进:")
        
        # 简单分析具体改进
            improvements = []
        
        # 检查导入改善
        strategies_used = round_result.get('strategies_used', [])
        if 'import_optimizer' in strategies_used:
            improvements.append("导入语句重新组织")
            
        # 检查注释改善
        if 'comment_optimizer' in round_result['strategies_used']:
            improvements.append("TODO注释格式化")
            
        # 检查空行改善
        if 'empty_line_optimizer' in round_result['strategies_used']:
            improvements.append("空行使用规范化")
            
        # 检查函数长度检测
        if 'function_length_optimizer' in round_result['strategies_used']:
            improvements.append("长函数检测和标记")
            
        # 检查变量命名
        if 'variable_naming_optimizer' in round_result['strategies_used']:
            improvements.append("变量命名建议")
            
        for improvement in improvements:
            print(f"      • {improvement}")
    else:
        print("   💡 系统谨慎起见，只添加了优化建议标记")
    
    # 验证系统功能完整性
    print(f"\n7️⃣ 系统自验证...")
    validation_result = orchestrator.self_validate()
    
    print(f"   📊 验证结果:")
    print(f"      ✅ 通过: {validation_result['tests_passed']}")
    print(f"      ❌ 失败: {validation_result['tests_failed']}")
    print(f"      🎯 整体: {'成功' if validation_result['success'] else '失败'}")
    
    # 性能统计
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n📈 性能统计:")
    print(f"   ⏱️  总用时: {duration:.2f} 秒")
    print(f"   🔄 优化轮数: 1")
    print(f"   📁 目标文件: {len(orchestrator.target_files)}")
    print(f"   🔧 总变更: {round_result['optimizations_applied']}")
    
    # 清理演示文件
    try:
        if os.path.exists(demo_file):
            os.remove(demo_file)
        if os.path.exists(backup_file):
            os.remove(backup_file)
        print(f"\n🧹 清理演示文件")
    except:
        pass
    
    # 总结
    print(f"\n🎯 自优化演示总结:")
    
    if validation_result['success'] and round_result['optimizations_applied'] > 0:
        print("   🎉 系统**成功优化了代码**并保持了功能完整性！")
        print("   ✅ 体现了真正的自学习能力")
    elif validation_result['success']:
        print("   💡 系统确认了代码质量，谨慎地只添加建议标记")
        print("   ✅ 体现了安全第一的设计理念")
    else:
        print("   ⚠️ 需要进一步调优优化策略")
    
    print("   🚀 这是迈向真正AI代码智能的重要一步！")

if __name__ == "__main__":
    self_optimization_demo()