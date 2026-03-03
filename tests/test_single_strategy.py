#!/usr/bin/env python3
"""
测试单个文件的策略应用
"""
import sys
import os

# 设置路径
sys.path.insert(0, os.getcwd())

def test_strategy_on_file(file_path, strategy_name):
    """测试特定文件和策略"""
    from src.strategies.optimization_strategies import CodeOptimizer
    
    print(f"🎯 测试 '{strategy_name}' on '{file_path}'")
    print("-" * 50)
    
    optimizer = CodeOptimizer()
    
    # 查找策略
    strategy = None
    for s in optimizer.strategies:
        if s.name == strategy_name:
            strategy = s
            break
    
    if not strategy:
        print(f"❌ 策略 '{strategy_name}' 不存在")
        return
    
    print(f"📝 策略描述: {strategy.description}")
    
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📄 文件大小: {len(content)} 字符, {len(content.splitlines())} 行")
        
        # 分析
        analysis = strategy.analyze(file_path, content)
        issues = analysis.get('issues_found', 0)
        can_optimize = analysis.get('can_optimize', False)
        
        print(f"🔍 分析结果: {issues} 个问题, 可优化: {can_optimize}")
        
        # 显示问题详情
        if strategy_name == 'line_length_optimizer':
            long_lines = analysis.get('long_lines', [])
            print(f"   长行数: {len(long_lines)}")
            for i, line in enumerate(long_lines[:3]):
                print(f"      第{line['line_number']}行: {line['length']}字符")
                print(f"         {line['content'][:60]}...")
        
        elif strategy_name == 'empty_line_optimizer':
            issues = analysis.get('empty_line_issues', [])
            print(f"   空行问题: {len(issues)}")
            for issue in issues[:3]:
                print(f"      第{issue['line_number']}行: {issue['type']}")
        
        elif strategy_name == 'comment_optimizer':
            issues = analysis.get('comment_issues', [])
            print(f"   注释问题: {len(issues)}")
            for issue in issues[:3]:
                print(f"      第{issue['line_number']}行: {issue['type']}")
        
        elif strategy_name == 'function_length_optimizer':
            functions = analysis.get('long_functions', [])
            print(f"   长函数: {len(functions)}")
            for func in functions[:3]:
                print(f"      {func['name']}(): {func['length']}行 (第{func['line_number']}行)")
        
        elif strategy_name == 'variable_naming_optimizer':
            issues = analysis.get('naming_issues', [])
            print(f"   命名问题: {len(issues)}")
            for issue in issues[:3]:
                print(f"      第{issue['line_number']}行: '{issue['variable']}' - {issue['issue']}")
        
        elif strategy_name == 'import_optimizer':
            issues = analysis.get('import_issues', [])
            imports = analysis.get('imports_found', 0)
            print(f"   导入数: {imports}, 问题: {len(issues)}")
            for issue in issues[:3]:
                print(f"      第{issue['line_number']}行: {issue['type']}")
        
        elif strategy_name == 'duplicate_code_optimizer':
            duplicates = analysis.get('duplicate_groups', [])
            print(f"   重复块: {len(duplicates)}")
            for dup in duplicates[:3]:
                print(f"      出现于第{dup['duplicate_line']}行, 相似度{dup['similarity']}")
        
        if can_optimize:
            print(f"\n🔧 尝试应用优化...")
            try:
                # 应用优化
                optimized_content, result = strategy.apply(file_path, content)
                changes = result.get('changes_count', 0)
                
                print(f"✅ 应用成功: {changes} 处变更")
                
                # 显示变更
                change_list = result.get('changes', [])
                for i, change in enumerate(change_list[:3]):
                    change_type = change.get('type', 'unknown')
                    change_desc = change.get('description', change_type)
                    print(f"   变更{i+1}: {change_desc}")
                
            except Exception as e:
                print(f"❌ 应用失败: {e}")
        
    except Exception as e:
        print(f"❌ 文件操作失败: {e}")

if __name__ == "__main__":
    # 测试几个策略
    test_files = [
        ('code/src/strategies/optimization_strategies.py', 'function_length_optimizer'),
        ('code/src/strategies/optimization_strategies.py', 'line_length_optimizer'),
        ('code/src/strategies/optimization_strategies.py', 'empty_line_optimizer'),
        ('code/src/nodes/base.py', 'variable_naming_optimizer'),
    ]
    
    for file_path, strategy in test_files:
        if os.path.exists(file_path):
            test_strategy_on_file(file_path, strategy)
            print("\n" + "="*60)
        else:
            print(f"⚠️ 文件不存在: {file_path}")