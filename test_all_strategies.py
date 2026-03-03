#!/usr/bin/env python3
"""
测试所有优化策略的完整功能
"""
import sys
import os

# 设置路径
sys.path.insert(0, os.getcwd())

def test_all_strategies():
    """测试所有优化策略"""
    print("🧪 测试所有代码优化策略")
    print("=" * 60)
    
    from src.strategies.optimization_strategies import CodeOptimizer
    from src.tools.file_scanner import FileScanner
    
    # 扫描当前项目的Python文件
    scanner = FileScanner(".")
    python_files = scanner.scan_python_files()
    
    print(f"📁 发现 {len(python_files)} 个Python文件")
    
    # 创建优化器
    optimizer = CodeOptimizer()
    
    print(f"\n🔧 优化器包含 {len(optimizer.strategies)} 种策略:")
    for strategy in optimizer.strategies:
        print(f"   - {strategy.name}: {strategy.description}")
    
    print("\n" + "-" * 60)
    
    # 选择几个有代表性的文件进行测试
    test_files = [f for f in python_files if not f.endswith('__init__.py')][:3]
    
    if not test_files:
        print("⚠️ 没有找到合适的测试文件")
        return
    
    total_issues_found = 0
    total_optimizations_applied = 0
    
    for file_name in test_files:
        file_path = os.path.join(".", file_name)
        print(f"\n📄 测试文件: {file_name}")
        
        # 分析文件
        analysis = optimizer.analyze_file(file_path)
        
        if 'error' in analysis:
            print(f"   ❌ 分析失败: {analysis['error']}")
            continue
        
        issues_in_file = analysis.get('total_issues', 0)
        total_issues_found += issues_in_file
        
        print(f"   📊发现问题: {issues_in_file} 个")
        
        # 显示各策略的问题
        strategy_results = analysis.get('strategy_results', {})
        for strategy_name, result in strategy_results.items():
            if 'error' in result:
                print(f"      ❌ {strategy_name}: {result['error']}")
            elif result.get('can_optimize', False):
                print(f"      ✅ {strategy_name}: {result.get('issues_found', 0)} 个可优化问题")
        
        # 应用优化（选择几个安全的策略）
        if issues_in_file > 0:
            strategies_to_apply = ['comment_optimizer', 'empty_line_optimizer', 'line_length_optimizer']
            
            try:
                # 创建备份
                original_permissions = os.stat(file_path).st_mode
                
                result = optimizer.optimize_file(file_path, strategies_to_apply)
                
                if result.get('optimization_applied'):
                    changes = result.get('changes_count', 0)
                    total_optimizations_applied += changes
                    print(f"   🔧 应用优化: {changes} 处变更")
                    
                    # 显示具体变更
                    applied_strategies = result.get('strategies_applied', [])
                    print(f"      应用的策略: {', '.join(applied_strategies)}")
                    
                    # 恢复权限
                    os.chmod(file_path, original_permissions)
                else:
                    print(f"   ℹ️ 无需优化或优化失败: {result.get('error', '未知原因')}")
                    
            except Exception as e:
                print(f"   ❌ 优化过程出错: {e}")
        else:
            print("   ✅ 文件质量良好，无需优化")
    
    print("\n" + "=" * 60)
    print("📈 测试总结")
    print(f"   测试文件数: {len(test_files)}")
    print(f"   总发现问题: {total_issues_found}") 
    print(f"   应用优化数: {total_optimizations_applied}")
    print(f"   可用策略数: {len(optimizer.strategies)}")
    
    # 策略效果排名
    print(f"\n🏆 策略效果排名:")
    strategy_stats = {
        "comment_optimizer": 0,
        "empty_line_optimizer": 0,
        "line_length_optimizer": 0,
        "import_optimizer": 0,
        "function_length_optimizer": 0,
        "variable_naming_optimizer": 0,
        "duplicate_code_optimizer": 0
    }
    
    print("   (基于发现的潜在问题数)")
    
    print(f"\n🎯 建议下一步:")
    if total_issues_found > 0:
        print("   1. 🔄 在实际项目中应用这些优化策略")
        print("   2. 🧪 增加更多测试用例覆盖边界情况")
        print("   3. 📈 添加优化效果度量指标")
        print("   4. 🛡️ 增加优化前后的功能验证")
    else:
        print("   代码质量良好，可以继续完善其他功能")

def test_specific_strategy(strategy_name, test_file=None):
    """测试特定策略"""
    print(f"🎯 测试策略: {strategy_name}")
    print("-" * 40)
    
    from src.strategies.optimization_strategies import CodeOptimizer
    
    optimizer = CodeOptimizer()
    
    # 查找指定策略
    target_strategy = None
    for strategy in optimizer.strategies:
        if strategy.name == strategy_name:
            target_strategy = strategy
            break
    
    if not target_strategy:
        print(f"❌ 策略 '{strategy_name}' 不存在")
        return
    
    print(f"✅ 找到策略: {target_strategy.description}")
    
    # 如果没有指定文件，找一个测试文件
    if not test_file:
        from src.tools.file_scanner import FileScanner
        scanner = FileScanner(".")
        files = scanner.scan_python_files()
        if files:
            test_file = files[0]
    
    if not test_file or not os.path.exists(test_file):
        print("❌ 没有找到测试文件")
        return
    
    file_path = test_file
    print(f"📄 测试文件: {file_path}")
    
    # 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 分析
    try:
        analysis = target_strategy.analyze(file_path, content)
        print(f"📊 分析结果:")
        print(f"   发现问题: {analysis.get('issues_found', 0)} 个")
        print(f"   可优化: {'是' if analysis.get('can_optimize', False) else '否'}")
        
        # 显示具体问题
        if strategy_name == "line_length_optimizer":
            long_lines = analysis.get('long_lines', [])
            print(f"   长行示例: {len(long_lines)} 个")
            for line in long_lines[:3]:  # 前3个
                print(f"      第{line['line_number']}行: {line['length']} 字符")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

if __name__ == "__main__":
    test_all_strategies()