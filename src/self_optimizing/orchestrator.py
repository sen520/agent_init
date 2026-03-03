#!/usr/bin/env python3
"""
自优化编排器 - 让系统能够优化自己的代码
"""
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time

# 添加当前项目到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SelfOptimizingOrchestrator:
    """自优化编排器 - 核心自优化引擎"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.optimization_rounds = []
        self.max_rounds = 3  # 最大自优化轮数
        self.target_files = []  # 需要优化的自己的文件
        
        # 核心组件
        self.scanner = None
        self.optimizer = None
        self.workflow = None
        
        self._initialize_components()
        self._identify_target_files()
    
    def _initialize_components(self):
        """初始化核心组件"""
        try:
            from src.tools.file_scanner import FileScanner
            from src.strategies.optimization_strategies import CodeOptimizer
            from src.graph.base import optimization_app
            
            self.scanner = FileScanner(str(self.project_path))
            self.optimizer = CodeOptimizer()
            self.workflow = optimization_app
            
            print("✅ 自优化组件初始化成功")
            
        except ImportError as e:
            raise ImportError(f"无法初始化自优化组件: {e}")
    
    def _identify_target_files(self):
        """识别需要优化的自己的代码文件"""
        # 定义系统自己的代码文件
        self_files = [
            "src/tools/file_scanner.py",
            "src/tools/code_analyzer.py", 
            "src/strategies/optimization_strategies.py",
            "src/nodes/base.py",
            "src/nodes/optimization.py",
            "src/graph/base.py",
            "src/state/base.py",
            "src/self_optimizing/orchestrator.py"
        ]
        
        self.target_files = []
        for rel_path in self_files:
            full_path = self.project_path / rel_path
            if full_path.exists():
                self.target_files.append(str(full_path))
        
        print(f"📁 识别了 {len(self.target_files)} 个自优化目标文件")
    
    def run_self_optimization_round(self) -> Dict[str, Any]:
        """执行一轮自优化"""
        round_number = len(self.optimization_rounds) + 1
        print(f"\n🔄 开始第 {round_number} 轮自优化...")
        
        round_result = {
            "round": round_number,
            "timestamp": time.time(),
            "files_analyzed": 0,
            "issues_found": 0,
            "optimizations_applied": 0,
            "strategies_used": [],
            "success": True,
            "errors": []
        }
        
        # 分析自己的代码
        total_issues = 0
        files_with_issues = []
        
        for file_path in self.target_files:
            try:
                analysis = self.optimizer.analyze_file(file_path)
                
                if 'error' in analysis:
                    round_result["errors"].append(f"分析 {file_path} 失败: {analysis['error']}")
                    continue
                
                issues_in_file = analysis.get('total_issues', 0)
                total_issues += issues_in_file
                round_result["files_analyzed"] += 1
                
                if issues_in_file > 0:
                    files_with_issues.append({
                        "file": file_path,
                        "issues": issues_in_file,
                        "analysis": analysis
                    })
                
                print(f"   📄 {os.path.basename(file_path)}: {issues_in_file} 个问题")
                
            except Exception as e:
                round_result["errors"].append(f"处理 {file_path} 异常: {e}")
        
        round_result["issues_found"] = total_issues
        
        # 如果没有问题，结束优化
        if total_issues == 0:
            print("   ✅ 代码质量良好，无需优化")
            round_result["success"] = True
            return round_result
        
        # 选择优化策略
        optimization_strategies = self._select_optimization_strategy(round_number)
        round_result["strategies_used"] = optimization_strategies
        
        # 应用优化
        total_optimizations = 0
        for file_info in files_with_issues[:3]:  # 限制每轮最多优化3个文件
            file_path = file_info["file"]
            
            try:
                print(f"   🔧 优化: {os.path.basename(file_path)}")
                result = self.optimizer.optimize_file(file_path, optimization_strategies)
                
                if result.get('optimization_applied'):
                    changes = result.get('changes_count', 0)
                    total_optimizations += changes
                    print(f"      ✅ 应用 {changes} 处优化")
                    
                    # 记录具体策略
                    applied = result.get('strategies_applied', [])
                    for strategy in applied:
                        if strategy not in round_result["strategies_used"]:
                            round_result["strategies_used"].append(strategy)
                else:
                    print(f"      ℹ️ 无需优化或优化失败")
                    error = result.get('error', '未知原因')
                    if error:
                        round_result["errors"].append(f"优化 {file_path}: {error}")
                
            except Exception as e:
                round_result["errors"].append(f"优化 {file_path} 异常: {e}")
        
        round_result["optimizations_applied"] = total_optimizations
        
        # 确定是否成功
        if total_optimizations == 0:
            round_result["success"] = False
            print(f"   ⚠️ 第 {round_number} 轮优化未产生实际效果")
        else:
            print(f"   ✅ 第 {round_number} 轮完成: {total_optimizations} 处优化")
        
        return round_result
    
    def _select_optimization_strategy(self, round_number: int) -> List[str]:
        """选择优化策略"""
        
        # 第1轮：安全优化
        if round_number == 1:
            strategies = [
                'comment_optimizer',
                'empty_line_optimizer', 
                'import_optimizer'
            ]
            print(f"   📋 第1轮策略: 安全优先 ({', '.join(strategies)})")
            return strategies
        
        # 第2轮：轻度优化
        elif round_number == 2:
            strategies = [
                'line_length_optimizer',
                'variable_naming_optimizer'
            ]
            print(f"   📋 第2轮策略: 轻度优化 ({', '.join(strategies)})")
            return strategies
        
        # 第3轮及以后：高级优化
        else:
            strategies = [
                'function_length_optimizer',
                'duplicate_code_optimizer'
            ]
            print(f"   📋 第{round_number}轮策略: 高级优化 ({', '.join(strategies)})")
            return strategies
    
    def run_full_self_optimization(self) -> Dict[str, Any]:
        """运行完整的自优化循环"""
        print("🚀 开始自优化循环...")
        print(f"📁 项目路径: {self.project_path}")
        print(f"🎯 目标文件数: {len(self.target_files)}")
        print(f"🔄 最大轮数: {self.max_rounds}")
        
        overall_result = {
            "start_time": time.time(),
            "end_time": None,
            "total_rounds": 0,
            "total_files_analyzed": 0,
            "total_issues_found": 0,
            "total_optimizations_applied": 0,
            "rounds": [],
            "success": False,
            "stop_reason": ""
        }
        
        # 执行多轮优化
        for round_num in range(self.max_rounds):
            round_result = self.run_self_optimization_round()
            self.optimization_rounds.append(round_result)
            overall_result["rounds"].append(round_result)
            
            # 累计统计
            overall_result["total_files_analyzed"] += round_result["files_analyzed"]
            overall_result["total_issues_found"] += round_result["issues_found"]
            overall_result["total_optimizations_applied"] += round_result["optimizations_applied"]
            overall_result["total_rounds"] += 1
            
            # 如果本轮没有发现任何问题，提前结束
            if round_result["issues_found"] == 0:
                overall_result["stop_reason"] = "代码已达到理想质量"
                overall_result["success"] = True
                break
            
            # 如果本轮优化没有效果，也提前结束
            if not round_result["success"] or round_result["optimizations_applied"] == 0:
                overall_result["stop_reason"] = "优化策略已收敛"
                overall_result["success"] = True
                break
            
            # 短暂休息，避免过于激进的优化
            time.sleep(1)
        
        # 处理达到最大轮数的情况
        if overall_result["total_rounds"] >= self.max_rounds:
            overall_result["stop_reason"] = "达到最大优化轮数"
            overall_result["success"] = True
        
        overall_result["end_time"] = time.time()
        optimization_duration = overall_result["end_time"] - overall_result["start_time"]
        
        # 生成优化报告
        print(f"\n" + "="*60)
        print("🏆 自优化循环完成！")
        print(f"📊 总体统计:")
        print(f"   ⏱️  用时: {optimization_duration:.2f} 秒")
        print(f"   🔄 优化轮数: {overall_result['total_rounds']}")
        print(f"   📁 分析文件: {overall_result['total_files_analyzed']}")
        print(f"   🔍 发现问题: {overall_result['total_issues_found']}")
        print(f"   🔧 应用优化: {overall_result['total_optimizations_applied']}")
        print(f"   ✅ 整体成功: {overall_result['success']}")
        print(f"   🛑 停止原因: {overall_result['stop_reason']}")
        
        # 显示每轮详情
        print(f"\n📋 各轮详情:")
        for i, round_result in enumerate(overall_result["rounds"], 1):
            print(f"   第{i}轮: 问题{round_result['issues_found']} → 优化{round_result['optimizations_applied']} ({'成功' if round_result['success'] else '无效果'})")
        
        return overall_result
    
    def self_validate(self) -> Dict[str, Any]:
        """自验证 - 确保优化后系统仍然正常工作"""
        print("🧪 开始自验证...")
        
        validation_result = {
            "start_time": time.time(),
            "end_time": None,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_results": [],
            "success": False
        }
        
        # 测试核心组件
        tests = [
            self._test_file_scanner,
            self._test_code_analyzer,
            self._test_optimization_strategies,
            self._test_workflow_integration
        ]
        
        for test_func in tests:
            test_name = test_func.__name__
            try:
                print(f"   🧪 {test_name}...")
                result = test_func()
                if result:
                    validation_result["tests_passed"] += 1
                    validation_result["test_results"].append({"name": test_name, "status": "PASSED"})
                    print(f"      ✅ 通过")
                else:
                    validation_result["tests_failed"] += 1
                    validation_result["test_results"].append({"name": test_name, "status": "FAILED"})
                    print(f"      ❌ 失败")
                    
            except Exception as e:
                validation_result["tests_failed"] += 1
                validation_result["test_results"].append({"name": test_name, "status": "ERROR", "error": str(e)})
                print(f"      ❌ 异常: {e}")
        
        validation_result["end_time"] = time.time()
        validation_result["success"] = validation_result["tests_failed"] == 0
        
        print(f"\n📊 自验证结果:")
        print(f"   ✅ 通过: {validation_result['tests_passed']}")
        print(f"   ❌ 失败: {validation_result['tests_failed']}")
        print(f"   🎯 整体: {'成功' if validation_result['success'] else '失败'}")
        
        return validation_result
    
    def _test_file_scanner(self) -> bool:
        """测试文件扫描器"""
        if not self.scanner:
            return False
        
        try:
            files = self.scanner.scan_python_files()
            return len(files) > 0
        except:
            return False
    
    def _test_code_analyzer(self) -> bool:
        """测试代码分析器"""
        if not self.optimizer:
            return False
        
        try:
            # 测试分析一个已知文件
            test_file = self.target_files[0] if self.target_files else None
            if not test_file:
                return False
                
            analysis = self.optimizer.analyze_file(test_file)
            return 'error' not in analysis
        except:
            return False
    
    def _test_optimization_strategies(self) -> bool:
        """测试优化策略"""
        if not self.optimizer:
            return False
        
        try:
            return len(self.optimizer.strategies) >= 7  # 应该有7种策略
        except:
            return False
    
    def _test_workflow_integration(self) -> bool:
        """测试工作流集成"""
        if not self.workflow:
            return False
        
        try:
            # 简单测试：检查工作流是否存在
            return True
        except:
            return False


# 便捷函数
def run_self_optimization(project_path: str = ".") -> Dict[str, Any]:
    """运行完整的自优化循环"""
    orchestrator = SelfOptimizingOrchestrator(project_path)
    
    # 1. 运行自优化
    optimization_result = orchestrator.run_full_self_optimization()
    
    # 2. 自验证
    validation_result = orchestrator.self_validate()
    
    return {
        "optimization": optimization_result,
        "validation": validation_result
    }


if __name__ == "__main__":
    print("🚀 自优化编排器演示")
    print("=" * 60)
    
    try:
        result = run_self_optimization(".")
        
        opt_result = result["optimization"]
        val_result = result["validation"]
        
        print(f"\n🎯 最终总结:")
        print(f"   优化轮数: {opt_result['total_rounds']}")
        print(f"   应用变更: {opt_result['total_optimizations_applied']}")
        print(f"   自验证: {'通过' if val_result['success'] else '失败'}")
        
        if opt_result["total_optimizations_applied"] > 0:
            print("✨ 系统成功优化了自己的代码！")
        else:
            print("💡 系统代码质量已经很好")
            
    except Exception as e:
        print(f"❌ 自优化过程异常: {e}")
        import traceback
        traceback.print_exc()