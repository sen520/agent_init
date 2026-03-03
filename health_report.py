#!/usr/bin/env python3
"""代码项目健康检查报告"""

import os
import sys
import subprocess
import json
from datetime import datetime

def check_system_status():
    """检查系统状态"""
    print("🔍 系统状态检查...")
    print("-" * 60)
    
    checks = []
    
    # 检查Python版本
    python_found = False
    for cmd in ['python3', 'python']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                python_version = result.stdout.strip()
                checks.append(("Python版本", f"{cmd}: {python_version}", "✅"))
                python_found = True
                break
        except:
            continue
    
    if not python_found:
        checks.append(("Python版本", "未找到", "❌"))
    
    # 检查工作空间
    try:
        cwd = os.getcwd()
        checks.append(("工作目录", cwd, "✅"))
    except:
        checks.append(("工作目录", "无法获取", "❌"))
    
    # 检查OpenClaw状态
    try:
        result = subprocess.run(['openclaw', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(("OpenClaw状态", "运行中", "✅"))
        else:
            checks.append(("OpenClaw状态", "未运行", "⚠️"))
    except FileNotFoundError:
        checks.append(("OpenClaw状态", "未安装", "❌"))
    
    return checks

def check_project_structure():
    """检查项目结构"""
    print("\n📁 项目结构检查...")
    print("-" * 60)
    
    base_path = os.getcwd()
    checks = []
    
    essential_files = [
        ("README.md", "项目说明"),
        ("main.py", "主程序"),
        ("src/", "源代码目录"),
        ("venv/", "虚拟环境"),
        (".env", "环境配置"),
        ("requirement.txt", "依赖文件"),
    ]
    
    for file_path, description in essential_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            size = ""
            if os.path.isfile(full_path):
                size = f"({os.path.getsize(full_path)} bytes)"
            checks.append((description, file_path, f"✅ {size}"))
        else:
            checks.append((description, file_path, "❌ 缺失"))
    
    return checks

def check_python_environment():
    """检查Python环境"""
    print("\n🐍 Python环境检查...")
    print("-" * 60)
    
    checks = []
    
    # 检查关键依赖
    required_packages = [
        ("langgraph", "工作流框架"),
        ("pydantic", "数据验证"),
        ("dotenv", "环境变量"),
        ("yaml", "配置文件"),
    ]
    
    for package, description in required_packages:
        try:
            __import__(package)
            version = ""
            try:
                import importlib.metadata
                version = importlib.metadata.version(package)
                checks.append((description, f"{package}=={version}", "✅"))
            except:
                checks.append((description, package, "✅"))
        except ImportError:
            checks.append((description, package, "❌ 未安装"))
    
    # 检查虚拟环境
    if os.path.exists("venv/bin/python"):
        checks.append(("虚拟环境", "venv/", "✅"))
    else:
        checks.append(("虚拟环境", "venv/", "⚠️ 未找到"))
    
    return checks

def check_code_tests():
    """检查代码测试"""
    print("\n🧪 代码测试检查...")
    print("-" * 60)
    
    checks = []
    
    test_files = [
        ("test_basic.py", "基本功能测试"),
        ("run_test.py", "完整流程测试"),
        ("test_fix.py", "修复测试"),
        ("test_simple.py", "简单测试"),
    ]
    
    for file_name, description in test_files:
        if os.path.exists(file_name):
            try:
                # 尝试运行测试
                result = subprocess.run(
                    ['python', file_name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    checks.append((description, file_name, "✅ 通过"))
                else:
                    checks.append((description, file_name, "❌ 失败"))
            except subprocess.TimeoutExpired:
                checks.append((description, file_name, "⚠️ 超时"))
            except Exception as e:
                checks.append((description, file_name, f"⚠️ 错误: {str(e)[:50]}"))
        else:
            checks.append((description, file_name, "⚠️ 不存在"))
    
    return checks

def calculate_health_score(all_checks):
    """计算健康分数"""
    total = len(all_checks)
    passed = 0
    warnings = 0
    failed = 0
    
    for desc, path, result in all_checks:
        if result.startswith("✅"):
            passed += 1
        elif result.startswith("⚠️"):
            warnings += 1
        else:
            failed += 1
    
    if total == 0:
        return 0, 0, 0, 0
    
    score = (passed * 100 + warnings * 50) // total
    
    return score, passed, warnings, failed

def main():
    """主函数"""
    print("🚀 代码项目健康检查报告")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {os.getcwd()}")
    print("=" * 80)
    
    all_checks = []
    
    # 执行各项检查
    system_checks = check_system_status()
    all_checks.extend(system_checks)
    
    project_checks = check_project_structure()
    all_checks.extend(project_checks)
    
    python_checks = check_python_environment()
    all_checks.extend(python_checks)
    
    test_checks = check_code_tests()
    all_checks.extend(test_checks)
    
    # 计算分数
    score, passed, warnings, failed = calculate_health_score(all_checks)
    
    # 显示总结
    print("\n📊 检查结果总结:")
    print("-" * 60)
    
    for desc, path, result in all_checks:
        print(f"  {desc:20} {path:30} {result}")
    
    print("\n" + "=" * 80)
    print(f"🏆 健康分数: {score}%")
    print(f"  通过: {passed} 项")
    print(f"  警告: {warnings} 项")
    print(f"  失败: {failed} 项")
    
    # 建议
    print("\n💡 建议:")
    if failed > 0:
        print("  ⚠️  有失败的检查项需要修复")
    if warnings > 0:
        print("  ℹ️  有一些警告项需要注意")
    if score >= 80:
        print("  ✅ 项目健康状况良好")
    elif score >= 60:
        print("  ⚠️  项目健康状况一般，需要维护")
    else:
        print("  🚨 项目健康状况较差，建议进行全面维护")
    
    # 生成报告文件
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "working_directory": os.getcwd(),
        "health_score": score,
        "stats": {
            "passed": passed,
            "warnings": warnings,
            "failed": failed
        },
        "checks": [
            {
                "description": desc,
                "path": path,
                "result": result
            }
            for desc, path, result in all_checks
        ]
    }
    
    report_file = "health_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细报告已保存到: {report_file}")
    
    return score >= 70  # 返回是否健康

if __name__ == "__main__":
    try:
        healthy = main()
        print("\n" + "=" * 80)
        if healthy:
            print("✅ 总体健康状况: 良好")
            sys.exit(0)
        else:
            print("⚠️  总体健康状况: 需要关注")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 检查过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)