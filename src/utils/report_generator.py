#!/usr/bin/env python3
"""
HTML 报告生成器 - 生成可视化的代码分析报告
"""
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from src.state.base import State


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>代码分析报告 - {{ project_name }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        header .meta {
            opacity: 0.9;
            font-size: 0.95em;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            text-align: center;
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card .number {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .card.good .number { color: #10b981; }
        .card.warning .number { color: #f59e0b; }
        .card.danger .number { color: #ef4444; }
        
        .card .label {
            color: #6b7280;
            font-size: 0.9em;
        }
        
        .section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .section h2 {
            color: #374151;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }
        
        .issue-list {
            list-style: none;
        }
        
        .issue-item {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .issue-item.high {
            background: #fef2f2;
            border-left-color: #ef4444;
        }
        
        .issue-item.medium {
            background: #fffbeb;
            border-left-color: #f59e0b;
        }
        
        .issue-item.low {
            background: #eff6ff;
            border-left-color: #3b82f6;
        }
        
        .issue-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .issue-type {
            font-weight: bold;
            color: #374151;
        }
        
        .issue-severity {
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .issue-severity.high {
            background: #ef4444;
            color: white;
        }
        
        .issue-severity.medium {
            background: #f59e0b;
            color: white;
        }
        
        .issue-severity.low {
            background: #3b82f6;
            color: white;
        }
        
        .issue-location {
            color: #6b7280;
            font-size: 0.85em;
            font-family: monospace;
        }
        
        .issue-description {
            margin-top: 8px;
            color: #4b5563;
        }
        
        .issue-suggestion {
            margin-top: 8px;
            padding: 10px;
            background: #f9fafb;
            border-radius: 6px;
            font-size: 0.9em;
            color: #059669;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 12px;
            background: #f9fafb;
            border-radius: 6px;
        }
        
        .stat-label {
            color: #6b7280;
        }
        
        .stat-value {
            font-weight: bold;
            color: #374151;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #9ca3af;
        }
        
        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.5;
        }
        
        footer {
            text-align: center;
            padding: 30px;
            color: #9ca3af;
            font-size: 0.85em;
        }
        
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 5px;
        }
        
        .badge-success {
            background: #d1fae5;
            color: #065f46;
        }
        
        .badge-warning {
            background: #fef3c7;
            color: #92400e;
        }
        
        .badge-danger {
            background: #fee2e2;
            color: #991b1b;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 代码分析报告</h1>
            <p class="meta">
                项目: {{ project_path }} | 
                生成时间: {{ timestamp }} |
                迭代: {{ iteration_count }}
            </p>
        </header>
        
        <div class="summary-cards">
            <div class="card {{ files_class }}">
                <div class="label">分析文件</div>
                <div class="number">{{ total_files }}</div>
            </div>
            <div class="card {{ lines_class }}">
                <div class="label">代码行数</div>
                <div class="number">{{ total_lines }}</div>
            </div>
            <div class="card {{ issues_class }}">
                <div class="label">发现问题</div>
                <div class="number">{{ total_issues }}</div>
            </div>
            <div class="card {{ complexity_class }}">
                <div class="label">平均复杂度</div>
                <div class="number">{{ avg_complexity }}</div>
            </div>
        </div>
        
        {% if issue_summary %}
        <div class="section">
            <h2>📊 问题统计</h2>
            <div class="stats-grid">
                {% for issue_type, count in issue_summary.items() %}
                <div class="stat-row">
                    <span class="stat-label">{{ issue_type }}</span>
                    <span class="stat-value">{{ count }}</span>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <div class="section">
            <h2>🔍 发现的问题</h2>
            {% if issues %}
            <ul class="issue-list">
                {% for issue in issues %}
                <li class="issue-item {{ issue.severity }}">
                    <div class="issue-header">
                        <span class="issue-type">{{ issue.type }}</span>
                        <span class="issue-severity {{ issue.severity }}">{{ issue.severity }}</span>
                    </div>
                    <div class="issue-location">📄 {{ issue.file }}:{{ issue.line }}</div>
                    <div class="issue-description">{{ issue.description }}</div>
                    {% if issue.suggestion %}
                    <div class="issue-suggestion">💡 {{ issue.suggestion }}</div>
                    {% endif %}
                </li>
                {% endfor %}
            </ul>
            {% else %}
            <div class="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p>🎉 没有发现代码问题！代码质量良好。</p>
            </div>
            {% endif %}
        </div>
        
        {% if applied_changes %}
        <div class="section">
            <h2>✅ 应用的优化</h2>
            <ul>
                {% for change in applied_changes %}
                <li>{{ change }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
    
    <footer>
        <p>由 LangGraph 自我优化代码助手生成 | 🤖 AI 驱动的代码分析</p>
    </footer>
</body>
</html>
"""


def generate_html_report(state: State, output_path: str = None) -> str:
    """
    生成 HTML 格式的分析报告
    
    Args:
        state: 工作流状态
        output_path: 输出路径（可选，从配置读取默认路径）
        
    Returns:
        HTML 文件路径
    """
    from jinja2 import Template
    
    # 加载配置
    config = get_config()
    report_config = config.get_report_config()
    
    # 准备数据
    template_data = {
        "project_name": Path(state.project_path).name or "Unknown",
        "project_path": state.project_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iteration_count": state.iteration_count,
        "total_files": state.analysis.total_files_analyzed,
        "total_lines": state.analysis.total_lines_of_code,
        "total_issues": len(state.analysis.issues),
        "avg_complexity": round(state.analysis.average_complexity, 2),
        "issue_summary": state.analysis.issue_summary,
        "applied_changes": state.applied_changes,
        "issues": []
    }
    
    # 确定卡片样式
    template_data["files_class"] = "good" if template_data["total_files"] > 0 else "warning"
    template_data["lines_class"] = "good" if template_data["total_lines"] > 100 else "warning"
    template_data["issues_class"] = "good" if template_data["total_issues"] == 0 else ("warning" if template_data["total_issues"] < 10 else "danger")
    template_data["complexity_class"] = "good" if template_data["avg_complexity"] < 10 else ("warning" if template_data["avg_complexity"] < 20 else "danger")
    
    # 准备问题列表
    severity_order = {"high": 0, "medium": 1, "low": 2}
    max_issues = report_config.get('max_issues_in_report', 50)
    sorted_issues = sorted(
        state.analysis.issues,
        key=lambda x: severity_order.get(x.severity, 3)
    )[:max_issues]  # 最多显示配置数量的问题
    
    for issue in sorted_issues:
        template_data["issues"].append({
            "type": issue.issue_type,
            "severity": issue.severity,
            "file": issue.file_path,
            "line": issue.line_number,
            "description": issue.description,
            "suggestion": issue.suggestion
        })
    
    # 渲染模板
    template = Template(HTML_TEMPLATE)
    html_content = template.render(**template_data)
    
    # 确定输出路径
    if not output_path:
        reports_dir = Path(report_config.get('output_dir', 'reports'))
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(output_path)


def create_report_node(state: State) -> State:
    """
    创建 HTML 报告的节点函数
    """
    print("📊 [报告] 生成 HTML 分析报告...")
    
    try:
        report_path = generate_html_report(state)
        print(f"   ✅ 报告已生成: {report_path}")
        state.logs.append(f"HTML报告已生成: {report_path}")
        
        # 同时在 summary 中记录
        if not hasattr(state, 'report_files'):
            state.report_files = []
        state.report_files.append(report_path)
        
    except ImportError:
        print("   ⚠️  缺少 jinja2，无法生成 HTML 报告")
        print("   💡 运行: pip install jinja2")
        state.logs.append("缺少 jinja2，无法生成 HTML 报告")
    except Exception as e:
        print(f"   ❌ 生成报告失败: {e}")
        state.errors.append(f"生成 HTML 报告失败: {e}")
    
    return state
