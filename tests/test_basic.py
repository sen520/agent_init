#!/usr/bin/env python3
"""基本功能测试"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.state.base import State, CodeAnalysis, OptimizationPlan


async def test_basic():
    """测试基本模型功能"""
    print("🧪 测试基本模型功能...")
    print("=" * 60)
    
    # 创建基本状态
    state = State(
        project_path="/test/path",
        project_name="TestProj",
        project_type="python"
    )
    
    print("✅ 状态创建成功")
    print(f"   项目: {state.project_name}")
    print(f"   路径: {state.project_path}")
    print(f"   类型: {state.project_type}")
    
    # 测试分析数据设置
    state.analysis.total_files_analyzed = 10
    state.analysis.total_lines_of_code = 500
    state.analysis.average_complexity = 0.6
    
    print("\n📊 分析数据:")
    print(f"   文件数: {state.analysis.total_files_analyzed}")
    print(f"   代码行: {state.analysis.total_lines_of_code}")
    print(f"   复杂度: {state.analysis.average_complexity}")
    
    # 测试JSON序列化
    json_str = state.model_dump_json(indent=2)
    print(f"\n📄 JSON 大小: {len(json_str)} 字符")
    
    # 测试日志功能
    state.add_log("这是一个测试日志")
    print(f"\n📝 日志: {state.logs}")
    
    return True


def test_node_fixes():
    """测试节点修复"""
    print("\n🔧 测试节点修复...")
    print("=" * 60)
    
    from src.nodes.base import initialize_project, analyze_code
    
    # 创建状态
    state = State(project_path="/test/path")
    
    try:
        # 测试初始化节点
        state1 = initialize_project(state)
        print("✅ 初始化节点通过")
        
        # 测试分析节点
        state2 = analyze_code(state1)
        print("✅ 分析节点通过")
        
        print(f"\n📊 分析结果:")
        print(f"   总文件数: {state2.analysis.total_files_analyzed}")
        print(f"   总代码行: {state2.analysis.total_lines_of_code}")
        
        return True
    except Exception as e:
        print(f"❌ 节点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_python_environment():
    """测试Python环境"""
    print("\n🐍 测试Python环境...")
    print("=" * 60)
    
    # 检查关键依赖
    required_modules = [
        "langgraph", 
        "pydantic", 
        "dotenv",
        "yaml",
        "typing_extensions"
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} 未安装")
            return False
    
    return True


if __name__ == '__main__':
    print("🚀 开始代码测试...")
    
    tests = [
        ("Python环境", test_python_environment),
        ("基本模型", lambda: asyncio.run(test_basic())),
        ("节点修复", test_node_fixes)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📋 测试结果:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
    else:
        print("\n🚨 有测试失败!")
        sys.exit(1)