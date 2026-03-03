#!/usr/bin/env python3
"""
真实的代码分析节点 - 替换模拟数据
"""
from typing import Dict, Any
from pathlib import Path
import os

from src.state.base import State, CodeIssue
from src.tools.file_scanner import FileScanner
from src.tools.code_analyzer import CodeAnalyzer


def initialize_project(state: State) -> State:
    """初始化项目分析 - 使用实际项目路径"""
    print("🔄 [节点1] 初始化项目")
    
    # 使用当前目录或传入的路径
    if not state.project_path:
        state.project_path = os.getcwd()
    
    # 确保路径存在
    project_path = Path(state.project_path)
    if not project_path.exists():
        # 尝试使用 agent_init 项目本身
        agent_init_path = Path(__file__).parent.parent.parent.parent
        if agent_init_path.exists():
            state.project_path = str(agent_init_path)
            print(f"   ⚠️  指定路径不存在，使用默认路径: {state.project_path}")
        else:
            state.project_path = os.getcwd()
    
    state.iteration_count += 1
    print(f"   项目路径: {state.project_path}")
    print(f"   当前迭代: {state.iteration_count}")
    return state


def analyze_code(state: State) -> State:
    """真实代码分析 - 使用 CodeAnalyzer 分析实际文件"""
    print("🔍 [节点2] 分析代码")
    
    try:
        # 初始化扫描器和分析器
        scanner = FileScanner(state.project_path)
        analyzer = CodeAnalyzer()
        
        # 扫描 Python 文件
        python_files = scanner.scan_python_files()
        
        if not python_files:
            print("   ⚠️  未找到 Python 文件")
            state.analysis.total_files_analyzed = 0
            return state
        
        print(f"   发现 {len(python_files)} 个 Python 文件")
        
        # 分析每个文件
        all_issues = []
        total_lines = 0
        total_complexity = 0
        files_with_issues = 0
        
        # 限制分析文件数量，避免太慢
        files_to_analyze = python_files[:20]  # 最多分析20个文件
        
        for file_path in files_to_analyze:
            full_path = Path(state.project_path) / file_path
            
            try:
                result = analyzer.analyze_file(str(full_path))
                
                # 累加统计
                total_lines += result.get('total_lines', 0)
                total_complexity += result.get('stats', {}).get('complexity', 0)
                
                # 收集问题
                file_issues = result.get('issues', [])
                if file_issues:
                    files_with_issues += 1
                    for issue in file_issues:
                        code_issue = CodeIssue(
                            file_path=file_path,
                            line_number=issue.get('line', 0),
                            issue_type=issue.get('type', 'unknown'),
                            description=issue.get('description', ''),
                            severity=issue.get('severity', 'medium'),
                            suggestion=issue.get('suggestion', '')
                        )
                        all_issues.append(code_issue)
                
                # 显示进度
                if len(all_issues) <= 5:
                    print(f"   📄 {file_path}: {len(file_issues)} 个问题")
                    
            except Exception as e:
                print(f"   ❌ 分析失败 {file_path}: {e}")
        
        # 更新状态
        state.analysis.total_files_analyzed = len(files_to_analyze)
        state.analysis.total_lines_of_code = total_lines
        state.analysis.average_complexity = total_complexity / max(len(files_to_analyze), 1)
        state.analysis.issues = all_issues
        state.analysis.project_root = state.project_path
        
        # 问题摘要
        issue_summary = {}
        for issue in all_issues:
            issue_type = issue.issue_type
            issue_summary[issue_type] = issue_summary.get(issue_type, 0) + 1
        state.analysis.issue_summary = issue_summary
        
        print(f"\n   📊 分析完成:")
        print(f"      文件数: {len(files_to_analyze)}")
        print(f"      总行数: {total_lines}")
        print(f"      发现 {len(all_issues)} 个问题（{files_with_issues} 个文件）")
        
        if issue_summary:
            print(f"\n   📋 问题分类:")
            for issue_type, count in sorted(issue_summary.items(), key=lambda x: -x[1])[:5]:
                print(f"      - {issue_type}: {count}")
        
    except Exception as e:
        print(f"   ❌ 分析过程出错: {e}")
        state.errors.append(f"代码分析失败: {e}")
    
    return state


def create_analysis_report(state: State) -> State:
    """创建真实的分析报告"""
    print("📋 [节点3] 创建分析报告")
    
    report = {
        "project_path": state.project_path,
        "files_analyzed": state.analysis.total_files_analyzed,
        "lines_of_code": state.analysis.total_lines_of_code,
        "average_complexity": round(state.analysis.average_complexity, 2),
        "issues_found": len(state.analysis.issues),
        "issue_summary": state.analysis.issue_summary,
        "iteration_count": state.iteration_count,
        "top_issues": []
    }
    
    # 收集最严重的问题
    severity_order = {"high": 0, "medium": 1, "low": 2}
    sorted_issues = sorted(
        state.analysis.issues,
        key=lambda x: severity_order.get(x.severity, 3)
    )[:10]  # 只显示前10个
    
    for issue in sorted_issues:
        report["top_issues"].append({
            "file": issue.file_path,
            "line": issue.line_number,
            "type": issue.issue_type,
            "description": issue.description,
            "severity": issue.severity
        })
    
    state.analysis_reports.append(report)
    
    print(f"   分析报告已生成:")
    print(f"      - 分析文件: {report['files_analyzed']}")
    print(f"      - 发现问题: {report['issues_found']}")
    print(f"      - 平均复杂度: {report['average_complexity']}")
    
    if report["top_issues"]:
        print(f"\n   🔴 前 {len(report['top_issues'])} 个问题:")
        for i, issue in enumerate(report["top_issues"][:5], 1):
            print(f"      {i}. [{issue['severity'].upper()}] {issue['file']}:{issue['line']}")
            print(f"         {issue['description']}")
    
    return state


def end_optimization(state: State) -> State:
    """结束优化流程 - 生成最终总结"""
    print("🏁 [节点7] 结束优化")
    
    total_issues = len(state.analysis.issues)
    fixed_issues = len(state.applied_changes)
    
    print("=" * 60)
    print("优化总结报告:")
    print(f"  迭代次数: {state.iteration_count}")
    print(f"  分析文件: {state.analysis.total_files_analyzed}")
    print(f"  代码行数: {state.analysis.total_lines_of_code}")
    print(f"  发现问题: {total_issues}")
    print(f"  应用优化: {fixed_issues}")
    
    if state.stop_reason:
        print(f"  停止原因: {state.stop_reason}")
    
    if state.analysis.issue_summary:
        print("\n  问题统计:")
        for issue_type, count in sorted(state.analysis.issue_summary.items(), key=lambda x: -x[1]):
            print(f"    - {issue_type}: {count}")
    
    if total_issues > 0 and fixed_issues == 0:
        print("\n  💡 提示: 发现问题但未应用优化")
        print("     运行 'python main.py full' 进行完整优化")
    
    print("=" * 60)
    return state
