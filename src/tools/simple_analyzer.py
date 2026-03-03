#!/usr/bin/env python3
"""
简化版代码分析器 - 先确保基本功能可用
"""
import ast
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


class SimpleCodeAnalyzer:
    """简化的代码分析器 - 避免复杂的AST处理问题"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'max_line_length': 100,
            'max_function_length': 50,
            'check_todos': True,
            'check_security': True,
        }
        self.issues = []
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析单个Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            lines = source.splitlines()
            file_stats = self._get_file_stats(file_path, lines)
            
            # 重置问题列表
            self.issues = []
            
            # 基本分析
            self._check_line_length(lines)
            self._check_todos(lines)
            self._check_imports(source)
            self._check_bare_excepts(source, lines)
            
            # AST解析（可选的）
            try:
                ast_tree = ast.parse(source)
                self._check_ast_issues(ast_tree, lines)
                file_stats['ast_parsed'] = True
                file_stats['ast_metrics'] = self._get_ast_metrics(ast_tree)
            except SyntaxError as e:
                self.issues.append({
                    'type': 'syntax_error',
                    'line': e.lineno,
                    'message': f'语法错误: {str(e)}',
                    'severity': 'error'
                })
                file_stats['ast_parsed'] = False
            except Exception as e:
                # AST解析失败但不影响基本分析
                file_stats['ast_parsed'] = False
            
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'total_lines': len(lines),
                'non_empty_lines': len([l for l in lines if l.strip()]),
                'issues': self.issues,
                'stats': file_stats,
                'issue_summary': self._summarize_issues(),
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': f'文件读取错误: {e}',
                'issues': []
            }
    
    def _get_file_stats(self, file_path: str, lines: List[str]) -> Dict[str, Any]:
        """获取文件基本统计信息"""
        stats = {
            'size_bytes': os.path.getsize(file_path),
            'modification_time': os.path.getmtime(file_path),
            'char_count': sum(len(line) for line in lines),
            'avg_line_length': 0,
            'max_line_length': max((len(line) for line in lines), default=0),
            'empty_lines': len([l for l in lines if not l.strip()]),
            'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
        }
        
        non_empty_lines = stats['total_lines'] - stats['empty_lines']
        stats['avg_line_length'] = stats['char_count'] / max(non_empty_lines, 1)
        
        return stats
    
    def _check_line_length(self, lines: List[str]):
        """检查行长度"""
        max_length = self.config.get('max_line_length', 100)
        for i, line in enumerate(lines):
            if len(line) > max_length:
                self.issues.append({
                    'type': 'long_line',
                    'line': i + 1,
                    'length': len(line),
                    'message': f'行过长 ({len(line)} > {max_length}字符)',
                    'severity': 'info'
                })
    
    def _check_todos(self, lines: List[str]):
        """检查TODO/FIXME注释"""
        if not self.config.get('check_todos', True):
            return
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'todo' in line_lower or 'fixme' in line_lower:
                self.issues.append({
                    'type': 'todo_comment',
                    'line': i + 1,
                    'message': '发现TODO/FIXME注释',
                    'severity': 'info'
                })
    
    def _check_imports(self, source: str):
        """检查导入问题"""
        lines = source.splitlines()
        for i, line in enumerate(lines):
            # 检查通配符导入
            if line.strip().startswith('from ') and ' import *' in line:
                self.issues.append({
                    'type': 'wildcard_import',
                    'line': i + 1,
                    'message': '通配符导入 (*) 可能引入名称冲突',
                    'severity': 'warning'
                })
    
    def _check_bare_excepts(self, source: str, lines: List[str]):
        """检查空的except块"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'except:' or stripped == 'except Exception:':
                self.issues.append({
                    'type': 'bare_except',
                    'line': i + 1,
                    'message': '空的except块，建议指定具体的异常类型',
                    'severity': 'warning'
                })
    
    def _check_ast_issues(self, tree: ast.AST, lines: List[str]):
        """检查AST相关的问题"""
        
        # 检查eval/exec使用
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        line_no = getattr(node, 'lineno', 0)
                        self.issues.append({
                            'type': 'security_eval',
                            'line': line_no,
                            'message': '使用eval/exec可能存在安全风险',
                            'severity': 'warning'
                        })
    
    def _get_ast_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """从AST获取代码指标"""
        metrics = {
            'function_count': 0,
            'class_count': 0,
            'method_count': 0,
            'import_count': 0,
            'function_names': [],
            'class_names': [],
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['function_count'] += 1
                metrics['function_names'].append(node.name)
                
                # 判断是否是方法
                parent = None
                for parent_node in ast.walk(tree):
                    if hasattr(parent_node, 'body'):
                        if node in parent_node.body:
                            if isinstance(parent_node, ast.ClassDef):
                                metrics['method_count'] += 1
                                break
            
            elif isinstance(node, ast.ClassDef):
                metrics['class_count'] += 1
                metrics['class_names'].append(node.name)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['import_count'] += 1
        
        # 调整计数（方法计入总数时去重）
        metrics['function_count'] = max(metrics['function_count'] - metrics['method_count'], 0)
        
        return metrics
    
    def _summarize_issues(self) -> Dict[str, int]:
        """汇总问题"""
        summary = {}
        for issue in self.issues:
            issue_type = issue.get('type', 'unknown')
            summary[issue_type] = summary.get(issue_type, 0) + 1
        return summary
    
    def analyze_directory(self, dir_path: str) -> Dict[str, Any]:
        """分析整个目录"""
        results = []
        total_issues = 0
        
        for root, dirs, files in os.walk(dir_path):
            # 跳过常见不需要的目录
            dirs[:] = [d for d in dirs if d not in ['__pycache__', 'venv', '.git', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    result = self.analyze_file(file_path)
                    results.append(result)
                    total_issues += len(result.get('issues', []))
        
        # 统计
        file_count = len(results)
        files_with_issues = sum(1 for r in results if r.get('issues'))
        
        return {
            'directory': dir_path,
            'file_count': file_count,
            'total_files_with_issues': files_with_issues,
            'total_issues': total_issues,
            'avg_issues_per_file': total_issues / max(file_count, 1),
            'results': results,
            'top_issue_types': self._get_top_issue_types(results)
        }
    
    def _get_top_issue_types(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取最常见的问题类型"""
        issue_counts = {}
        for result in results:
            for issue in result.get('issues', []):
                issue_type = issue.get('type', 'unknown')
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'type': t, 'count': c} for t, c in sorted_issues[:10]]


# 便捷函数
def analyze_file(file_path: str) -> Dict[str, Any]:
    """分析单个文件"""
    analyzer = SimpleCodeAnalyzer()
    return analyzer.analyze_file(file_path)


def analyze_dir(dir_path: str) -> Dict[str, Any]:
    """分析整个目录"""
    analyzer = SimpleCodeAnalyzer()
    return analyzer.analyze_directory(dir_path)


if __name__ == "__main__":
    # 测试代码分析器
    print("🧪 简化代码分析器测试")
    print("=" * 60)
    
    try:
        # 测试当前文件
        result = analyze_file(__file__)
        
        print(f"📄 分析文件: {result['file_name']}")
        print(f"📊 总行数: {result['total_lines']}")
        print(f"📝 非空行: {result['non_empty_lines']}")
        
        issues = result.get('issues', [])
        print(f"🔍 发现问题: {len(issues)} 个")
        
        if issues:
            print("\n具体问题:")
            for issue in issues[:10]:  # 只显示前10个
                line = issue.get('line', '?')
                message = issue.get('message', '')
                severity = issue.get('severity', 'info')
                print(f"  L{line}: {message} [{severity}]")
        else:
            print("\n✅ 没有发现问题")
        
        # 测试目录分析
        print("\n📁 测试目录分析...")
        dir_result = analyze_dir('.')
        print(f"找到 {dir_result['file_count']} 个Python文件")
        print(f"总问题数: {dir_result['total_issues']}")
        
        if dir_result['top_issue_types']:
            print("\n最常见的问题类型:")
            for item in dir_result['top_issue_types'][:5]:
                print(f"  {item['type']}: {item['count']} 次")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()