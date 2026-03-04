#!/usr/bin/env python3
"""
Phase 2 节点 - LLM 增强和测试验证
"""
from typing import Dict, Any, List
from pathlib import Path

from src.state.base import State
from src.config.manager import get_config


def llm_analyze_issues(state: State) -> State:
    """
    使用 LLM 分析问题节点
    """
    print("\n🧠 [LLM] 智能分析问题...")
    
    try:
        from src.llm.enhancer import LLMEnhancer
        
        config = get_config()
        llm_config = config.get_llm_config()
        
        if not llm_config.get('enabled', False):
            print("   ℹ️  LLM 未启用，跳过智能分析")
            state.logs.append("LLM 未启用")
            return state
        
        enhancer = LLMEnhancer()
        
        if not enhancer.is_available():
            print("   ⚠️  LLM 客户端不可用（请设置 KIMI_API_KEY）")
            state.logs.append("LLM 客户端不可用")
            return state
        
        # 获取最严重的问题
        severity_order = {"high": 0, "medium": 1, "low": 2}
        top_issues = sorted(
            state.analysis.issues,
            key=lambda x: severity_order.get(x.severity, 3)
        )[:5]  # 只分析前5个
        
        if not top_issues:
            print("   ℹ️  没有问题需要分析")
            return state
        
        # 读取第一个有问题的文件
        if top_issues:
            first_issue = top_issues[0]
            file_path = Path(state.project_path) / first_issue.file_path
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()[:2000]  # 限制长度
            except:
                code = ""
            
            # 使用 LLM 分析
            issues_dict = [{
                "type": i.issue_type,
                "description": i.description,
                "severity": i.severity,
                "line": i.line_number
            } for i in top_issues]
            
            print(f"   📝 正在分析 {first_issue.file_path}...")
            analysis = enhancer.analyze_code_issues(code, issues_dict)
            
            # 保存分析结果
            if not hasattr(state, 'llm_suggestions'):
                state.llm_suggestions = []
            
            state.llm_suggestions.append({
                "file": first_issue.file_path,
                "analysis": analysis
            })
            
            print(f"   ✅ LLM 分析完成")
            state.logs.append(f"LLM 分析了 {first_issue.file_path}")
            
            # 显示部分结果
            preview = analysis[:200] + "..." if len(analysis) > 200 else analysis
            print(f"\n   📋 分析摘要:\n   {preview}")
        
    except ImportError as e:
        print(f"   ⚠️  LLM 模块不可用: {e}")
        state.logs.append(f"LLM 模块导入失败: {e}")
    except Exception as e:
        print(f"   ❌ LLM 分析失败: {e}")
        state.errors.append(f"LLM 分析错误: {e}")
    
    return state


def validate_optimization(state: State) -> State:
    """
    验证优化结果节点
    """
    print("\n🧪 [验证] 测试优化结果...")
    
    try:
        from src.testing.validator import CodeValidator
        from src.utils.file_modifier import FileModifier
        
        # 获取修改过的文件
        modified_files = []
        for impl in state.implementations:
            if impl.lines_added > 0:
                modified_files.extend(impl.changed_files)
        
        if not modified_files:
            print("   ℹ️  没有文件被修改，跳过验证")
            state.logs.append("无修改文件，跳过验证")
            return state
        
        # 运行验证
        validator = CodeValidator(state.project_path)
        result = validator.validate_after_optimization(modified_files)
        
        # 保存验证结果
        state.validation_result = result
        
        if result["success"]:
            print("   ✅ 验证通过！优化成功")
            state.logs.append("优化验证通过")
        else:
            print("   ❌ 验证失败")
            state.logs.append("优化验证失败")
            
            # 如果需要回滚
            if result.get("should_rollback"):
                print("\n   🔄 正在回滚修改...")
                modifier = FileModifier()
                success, failed = modifier.rollback_all()
                print(f"   回滚完成: {success} 成功, {failed} 失败")
                state.logs.append(f"回滚了 {success} 个文件")
                state.applied_changes = []  # 清空已应用的变更
        
    except ImportError as e:
        print(f"   ⚠️  测试验证模块不可用: {e}")
        state.logs.append(f"验证模块导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 验证过程出错: {e}")
        state.errors.append(f"验证错误: {e}")
    
    return state


def generate_llm_report(state: State) -> State:
    """
    生成包含 LLM 建议的报告
    """
    print("\n📊 [报告] 生成智能分析报告...")
    
    # 基础 HTML 报告
    try:
        from src.utils.report_generator import generate_html_report
        report_path = generate_html_report(state)
        print(f"   ✅ HTML 报告: {report_path}")
    except Exception as e:
        print(f"   ⚠️  HTML 报告生成失败: {e}")
    
    # 如果有 LLM 建议，生成文本报告
    if hasattr(state, 'llm_suggestions') and state.llm_suggestions:
        try:
            report_lines = [
                "# 🤖 智能代码分析报告",
                "",
                f"项目: {state.project_path}",
                f"分析时间: {state.analysis.timestamp}",
                f"发现问题: {len(state.analysis.issues)}",
                "",
                "## 🧠 LLM 智能建议",
                ""
            ]
            
            for suggestion in state.llm_suggestions:
                report_lines.extend([
                    f"### 文件: {suggestion['file']}",
                    "",
                    suggestion['analysis'],
                    "",
                    "---",
                    ""
                ])
            
            # 保存报告
            from datetime import datetime
            report_file = Path("reports") / f"llm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report_file.parent.mkdir(exist_ok=True)
            report_file.write_text('\n'.join(report_lines), encoding='utf-8')
            
            print(f"   ✅ LLM 报告: {report_file}")
            state.logs.append(f"LLM 报告已生成: {report_file}")
            
        except Exception as e:
            print(f"   ⚠️  LLM 报告生成失败: {e}")
    
    return state
