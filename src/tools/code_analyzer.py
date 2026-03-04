#!/usr/bin/env python3
"""
实用的代码分析器 - 专注于基本但有用的分析功能
"""
import ast
import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.config.manager import get_config

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """实用的代码分析器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            # 从配置文件加载
            cfg = get_config()
            self.config = {
                'max_line_length': cfg.get('optimization.max_line_length', 100),
                'max_function_length': cfg.get('optimization.max_function_length', 50),
                'max_method_length': cfg.get('optimization.max_method_length', 30),
                'check_todos': cfg.get('code_analyzer.check_todos', True),
                'check_security': cfg.get('code_analyzer.check_security', True),
                'check_docs': cfg.get('code_analyzer.check_docs', True),
                'check_imports': cfg.get('code_analyzer.check_imports', True),
            }
        else:
            self.config = config
        self.issues = []
        self.metrics = {}
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析单个Python文件 - 实用版本"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            lines = source.splitlines()
            
            # 重置状态
            self.issues = []
            self.metrics = {}
            
            # 基本统计
            file_stats = self._get_file_stats(file_path, source, lines)
            
            # 运行各种检查
            self._check_syntax(source, file_path, lines)
            self._check_line_length(lines)
            self._check_todo_comments(lines)
            self._check_imports(source, lines)
            self._check_code_quality(source, lines)
            self._check_function_lengths(source, lines)
            
            # 从AST获取结构化信息
            try:
                tree = ast.parse(source)
                ast_metrics = self._get_ast_metrics(tree)
                file_stats.update(ast_metrics)
                
                # AST安全检查
                self._check_ast_security(tree, lines)
                
            except SyntaxError as e:
                self._add_issue('syntax_error', e.lineno, f'语法错误: {str(e)}')
            except Exception as e:
                # AST解析失败，但仍继续其他检查
                import logging
                logging.debug(f"AST解析失败 {file_path}: {e}")
                pass
            
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'total_lines': len(lines),
                'non_empty_lines': file_stats.get('non_empty_lines', 0),
                'issues': self.issues,
                'stats': file_stats,
                'issue_summary': self._summarize_issues(),
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'issues': []
            }
    
    def _get_file_stats(self, file_path: str, source: str, lines: List[str]) -> Dict[str, Any]:
        """获取文件基本统计信息"""
        try:
            size = os.path.getsize(file_path)
        except (OSError, IOError) as e:
            logger.debug(f"无法获取文件大小 {file_path}: {e}")
            size = 0
            
        non_empty_lines = len([l for l in lines if l.strip()])
        comment_lines = len([l for l in lines if l.lstrip().startswith('#')])
        
        return {
            'size_bytes': size,
            'total_chars': len(source),
            'non_empty_lines': non_empty_lines,
            'comment_lines': comment_lines,
            'comment_percentage': round(comment_lines / max(len(lines), 1) * 100, 1),
            'avg_line_length': len(source) / max(non_empty_lines, 1),
            'max_line_length': max((len(l) for l in lines), default=0),
        }
    
    def _check_syntax(self, source: str, file_path: str, lines: List[str]):
        """检查基本语法"""
        try:
            ast.parse(source)
        except SyntaxError as e:
            self._add_issue('syntax_error', e.lineno, f'Python语法错误: {str(e)[:100]}')
    
    def _check_line_length(self, lines: List[str]):
        """检查行长度"""
        max_len = self.config.get('max_line_length', 100)
        for i, line in enumerate(lines):
            if len(line) > max_len:
                self._add_issue('long_line', i+1, f'行过长 ({len(line)} > {max_len}字符)', 'warning')
    
    def _check_todo_comments(self, lines: List[str]):
        """检查TODO/FIXME注释"""
        if not self.config.get('check_todos', True):
            return
            
        keywords = ['todo', 'fixme', 'xxx', 'hack', 'bug', 'note']
        for i, line in enumerate(lines):
            line_lower = line.lower()
            for keyword in keywords:
                if f' {keyword}' in line_lower or line_lower.startswith(keyword):
                    self._add_issue('todo_comment', i+1, f'发现{keyword.upper()}注释', 'info')
                    break
    
    def _check_imports(self, source: str, lines: List[str]):
        """检查导入问题"""
        if not self.config.get('check_imports', True):
            return
            
        line_num = 0
        for match in re.finditer(r'import\s+\*', source, re.MULTILINE):
            # 找到行号
            subtext = source[:match.start()]
            line_num = subtext.count('\n') + 1
            self._add_issue('wildcard_import', line_num, '通配符导入 (*) 应避免使用', 'warning')
        
        # 检查是否缺少__init__.py引用检查（这里简化）
        pass
    
    def _check_code_quality(self, source: str, lines: List[str]):
        """检查代码质量问题"""
        
        # 查找可能的问题模式
        patterns = [
            (r'except\s*:', 'bare_except', '空的except块'),
            (r'except\s+Exception\s*:', 'bare_except', '过于宽泛的Exception捕获'),
            (r'eval\s*\(', 'security_eval', 'eval函数使用（安全风险）'),
            (r'exec\s*\(', 'security_exec', 'exec函数使用（安全风险）'),
            (r'assert\s+\w+\s*==', 'production_assert', '生产代码中的assert可能被优化掉'),
            (r'print\s*\(', 'debug_print', '可能遗留的调试print语句'),
        ]
        
        for pattern, issue_type, description in patterns:
            for match in re.finditer(pattern, source, re.IGNORECASE):
                subtext = source[:match.start()]
                line_num = subtext.count('\n') + 1
                self._add_issue(issue_type, line_num, description, 'info')
    
    def _check_function_lengths(self, source: str, lines: List[str]):
        """检查函数长度 - 简单实现"""
        if not self.config.get('check_function_length', True):
            return
            
        # 简化实现：查找def语句并估算长度
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                func_start = i + 1
                # 估算函数结束（实际应该用AST）
                # 这里只是简单演示
                self._add_issue('function_info', func_start, '函数定义点')
    
    def _get_ast_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """从AST获取代码指标"""
        metrics = {
            'function_count': 0,
            'class_count': 0,
            'method_count': 0,
            'import_count': 0,
            'async_function_count': 0,
            'function_names': [],
            'class_names': [],
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['function_count'] += 1
                metrics['function_names'].append(node.name)
                if node.name.startswith('_'):
                    metrics['private_methods'] = metrics.get('private_methods', 0) + 1
                
                # 检查是否是异步函数
                if any(isinstance(deco, ast.AsyncFunctionDef) for deco in ast.iter_child_nodes(node)) or \
                   isinstance(node, ast.AsyncFunctionDef):
                    metrics['async_function_count'] += 1
                
                # 简单判断是否是方法：如果父节点是ClassDef
                for parent in ast.walk(tree):
                    if hasattr(parent, 'body') and node in parent.body:
                        if isinstance(parent, ast.ClassDef):
                            metrics['method_count'] += 1
                            break
            
            elif isinstance(node, ast.ClassDef):
                metrics['class_count'] += 1
                metrics['class_names'].append(node.name)
                if node.name.startswith('_'):
                    metrics['private_classes'] = metrics.get('private_classes', 0) + 1
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['import_count'] += 1
        
        # 调整计数
        metrics['function_count'] = max(metrics['function_count'] - metrics.get('method_count', 0), 0)
        
        return metrics
    
    def _check_ast_security(self, tree: ast.AST, lines: List[str]):
        """使用AST进行安全检查"""
        
        for node in ast.walk(tree):
            # 检查eval/exec调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        line_no = getattr(node, 'lineno', 0)
                        self._add_issue('security_eval', line_no, f'使用{node.func.id}()可能有安全风险', 'warning')
                
                # 检查pickle/序列化
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['loads', 'load', 'dump', 'dumps']:
                        func_obj = node.func.value
                        if isinstance(func_obj, ast.Name) and func_obj.id in ['pickle', 'marshal']:
                            line_no = getattr(node, 'lineno', 0)
                            self._add_issue('security_serialization', line_no, '不安全的序列化操作', 'warning')
    
    def _add_issue(self, issue_type: str, line: int, message: str, severity: str = 'info'):
        """添加问题记录"""
        self.issues.append({
            'type': issue_type,
            'line': line,
            'message': message,
            'severity': severity
        })
    
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
        
        file_paths = []
        for root, dirs, files in os.walk(dir_path):
            # 跳过常见目录
            skip_dirs = {'__pycache__', 'venv', '.git', '.idea', '.vscode', 'node_modules', 'dist', 'build'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_paths.append(os.path.join(root, file))
        
        # 分析每个文件
        for file_path in file_paths[:50]:  # 限制最大文件数，避免性能问题
            try:
                result = self.analyze_file(file_path)
                results.append(result)
                total_issues += len(result.get('issues', []))
            except Exception as e:
                logger.error(f"分析文件 {file_path} 时出错: {e}")
        
        # 统计
        file_count = len(results)
        files_with_issues = sum(1 for r in results if r.get('issues'))
        
        # 汇总分析
        all_issues = []
        for result in results:
            all_issues.extend(result.get('issues', []))
        
        # 按类型统计
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.get('type', 'unknown')
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'directory': dir_path,
            'files_analyzed': file_count,
            'files_with_issues': files_with_issues,
            'total_issues': total_issues,
            'avg_issues_per_file': total_issues / max(file_count, 1),
            'top_issues': [{'type': t, 'count': c} for t, c in top_issues],
            'file_results': results[:10],  # 只返回前10个结果避免响应过大
        }


# 便捷函数
def analyze_file(file_path: str) -> Dict[str, Any]:
    """分析单个文件"""
    analyzer = CodeAnalyzer()
    return analyzer.analyze_file(file_path)


def analyze_directory(dir_path: str) -> Dict[str, Any]:
    """分析整个目录"""
    analyzer = CodeAnalyzer()
    return analyzer.analyze_directory(dir_path)


def test_analyzer():
    """测试函数"""
    logger.info("🧪 代码分析器测试")
    logger.info("=" * 60)
    
    try:
        # 测试当前文件
        result = analyze_file(__file__)
        
        logger.info(f"📄 分析文件: {result['file_name']}")
        logger.info(f"📊 总行数: {result['total_lines']}")
        logger.info(f"📝 非空行: {result.get('non_empty_lines', 'N/A')}")
        
        stats = result.get('stats', {})
        if stats:
            logger.info(f"📈 字符数: {stats.get('total_chars', 'N/A')}")
            logger.info(f"💬 注释行: {stats.get('comment_lines', 'N/A')} ({stats.get('comment_percentage', 'N/A')}%)")
        
        issues = result.get('issues', [])
        logger.info(f"🔍 发现问题: {len(issues)} 个")
        
        if issues:
            issue_summary = result.get('issue_summary', {})
            logger.info("问题汇总:")
            for issue_type, count in issue_summary.items():
                logger.info(f"  {issue_type}: {count}个")
            
            logger.info("具体问题 (前5个):")
            for issue in issues[:5]:
                line = issue.get('line', '?')
                message = issue.get('message', '')
                severity = issue.get('severity', 'info')
                logger.info(f"  L{line}: {message} [{severity}]")
        else:
            logger.info("✅ 没有发现问题")
        
        # 测试目录分析
        logger.info("📁 测试目录分析...")
        dir_result = analyze_directory('.')
        logger.info(f"分析完成: {dir_result.get('files_analyzed', 0)} 个文件")
        logger.info(f"总问题数: {dir_result.get('total_issues', 0)}")
        
        top_issues = dir_result.get('top_issues', [])
        if top_issues:
            logger.info("最常见的问题类型:")
            for item in top_issues[:5]:
                logger.info(f"  {item['type']}: {item['count']} 次")
        
        logger.info("✅ 代码分析器测试完成")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    test_analyzer()