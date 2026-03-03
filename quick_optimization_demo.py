#!/usr/bin/env python3
"""
快速优化演示 - 展示系统优化建议的实际应用
"""
import sys
import os

# 设置路径
sys.path.insert(0, os.getcwd())

def demonstrate_optimization_improvements():
    """演示具体的优化改进"""
    print("🎯 系统优化改进演示")
    print("=" * 60)
    
    # 1. 配置系统演示
    print("\n1️⃣ 📋 配置系统演示:")
    try:
        from src.utils.config import get_config, load_config
        
        # 加载默认配置
        config = get_config()
        print("   ✅ 配置系统正常工作")
        print(f"   📝 最大行长度: {config.get_max_line_length()}")
        print(f"   🧩 默认策略: {', '.join(config.get_strategy_group('default')[:3])}")
        print(f"   🛡️ 备份功能: {'启用' if config.is_backup_enabled() else '禁用'}")
        
    except Exception as e:
        print(f"   ❌ 配置系统演示失败: {e}")
    
    # 2. 性能缓存演示
    print("\n2️⃣ ⚡ 性能缓存演示:")
    try:
        import time
        from src.strategies.optimization_strategies import CodeOptimizer
        
        optimizer = CodeOptimizer()
        test_file = "src/state/base.py"
        
        if os.path.exists(test_file):
            # 第一次分析
            start_time = time.time()
            analysis1 = optimizer.analyze_file(test_file)
            first_duration = time.time() - start_time
            
            # 第二次分析（模拟缓存效果）
            start_time = time.time()
            analysis2 = optimizer.analyze_file(test_file)
            second_duration = time.time() - start_time
            
            print(f"   📊 首次分析: {first_duration:.3f}秒")
            print(f"   📊 重复分析: {second_duration:.3f}秒")
            print(f"   🎯 结果一致: {analysis1['total_issues'] == analysis2['total_issues']}")
            
    except Exception as e:
        print(f"   ❌ 性能演示失败: {e}")
    
    # 3. 改进的策略选择演示
    print("\n3️⃣ 🎛️ 智能策略选择演示:")
    try:
        strategies = [
            ("safe", ["comment_optimizer", "empty_line_optimizer"]),
            ("standard", ["comment_optimizer", "empty_line_optimizer", "import_optimizer", "line_length_optimizer"]),
            ("aggressive", ["function_length_optimizer", "variable_naming_optimizer"])
        ]
        
        for level, strategy_list in strategies:
            print(f"   📝 {level.capitalize()} 级别: {', '.join(strategy_list[:2])}{'...' if len(strategy_list) > 2 else ''}")
    
    except Exception as e:
        print(f"   ❌ 策略演示失败: {e}")
    
    # 4. 错误处理增强演示
    print("\n4️⃣ 🛡️ 错误处理增强演示:")
    
    # 创建有问题的测试文件
    test_files = {
        "empty.py": "# 空文件\n",
        "syntax_error.py": "def broken(\n    pass\n",
        "normal.py": "def normal_function():\n    return 'ok'\n"
    }
    
    from src.strategies.optimization_strategies import CodeOptimizer
    optimizer = CodeOptimizer()
    
    for filename, content in test_files.items():
        try:
            with open(filename, 'w') as f:
                f.write(content)
            
            analysis = optimizer.analyze_file(filename)
            if 'error' in analysis:
                print(f"   ✅ {filename}: 正确捕获错误")
            else:
                print(f"   ✅ {filename}: 正常分析 ({analysis['total_issues']} 问题)")
            
        except Exception as e:
            print(f"   ❌ {filename}: 处理异常 - {e}")
        finally:
            # 清理文件
            if os.path.exists(filename):
                os.remove(filename)
            backup_file = filename + '.backup'
            if os.path.exists(backup_file):
                os.remove(backup_file)
    
    # 5. 类型安全改进演示
    print("\n5️⃣ 📝 类型安全改进演示:")
    try:
        from src.state.base import State
        
        # 创建带类型的状态
        state = State(project_path=".")
        
        # 类型安全的属性访问
        typed_attributes = [
            ('iteration_count', 'int'),
            ('total_files_analyzed', 'int'), 
            ('optimization_success', 'bool'),
            ('strategies_used', 'List[str]'),
            ('logs', 'List[str]')
        ]
        
        for attr, type_info in typed_attributes:
            if hasattr(state, attr):
                value = getattr(state, attr)
                print(f"   ✅ {attr}: {type_info} = {value}")
        
    except Exception as e:
        print(f"   ❌ 类型演示失败: {e}")
    
    print("\n🎉 优化改进演示完成！")
    print("\n💡 改进总结:")
    print("   📋 配置系统: 支持灵活的策略配置")
    print("   ⚡ 性能优化: 建议添加缓存机制")
    print("   🎛️ 策略分级: 安全/标准/激进三个级别")
    print("   🛡️ 错误处理: 优雅处理各种异常情况")
    print("   📝 类型安全: 渐进式提升类型注解覆盖率")

if __name__ == "__main__":
    demonstrate_optimization_improvements()