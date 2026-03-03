#!/usr/bin/env python3
"""
代码优化策略 - 定义具体的代码优化规则和实现
"""
import ast
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.config.manager import get_config


class OptimizationStrategy:
    """优化策略基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.applied_changes = []
    
    def analyze(self, file_path: str, content: str) -> Dict[str, Any]:
        """分析文件，返回优化建议"""
        raise NotImplementedError
    
    def apply(self, file_path: str, content: str) -> Tuple[str, Dict[str, Any]]:
        """应用优化，返回修改后的内容和变更信息"""
        raise NotImplementedError


class LineLengthOptimizer(OptimizationStrategy):
    """行长度优化策略"""
    
    def __init__(self):
        super().__init__(
            name="line_length_optimizer",
            description="优化过长的代码行，自动拆分"
        )
        # 从配置加载
        self.max_line_length = get_config().get('optimization.max_line_length', 100)
    
    def analyze(self, file_path: str, content: str) -> Dict[str, Any]:
        """查找过长的行"""
        lines = content.splitlines()
        long_lines = []
        
        for i, line in enumerate(lines):
            if len(line) > self.max_line_length:
                long_lines.append({
                    "line_number": i + 1,
                    "length": len(line),
                    "content": line
                })
        
        return {
            "file_path": file_path,
            "issues_found": len(long_lines),
            "long_lines": long_lines,
            "can_optimize": len(long_lines) > 0
        }
    
    def apply(self, file_path: str, content: str) -> Tuple[str, Dict[str, Any]]:
        """应用行长度优化"""
        lines = content.splitlines()
        optimized_lines = []
        changes = []
        
        for i, line in enumerate(lines):
            if len(line) <= self.max_line_length:
                optimized_lines.append(line)
                continue
            
            # 简单的拆分策略
            original_line = line.strip()
            
            # 查找可能的拆分点
            split_points = [
                (' ,', f',\n{" " * (len(line) - len(original_line) + 4)}'),  # 逗号后
                (' =', f'=\n{" " * (len(line) - len(original_line) + 4)}'),  # 等号后
                ('(', f'(\n{" " * (len(line) - len(original_line) + 8)}'),   # 括号后
            ]
            
            optimized_line = line
            modified = False
            
            for pattern, replacement in split_points:
                if pattern in optimized_line and not modified:
                    optimized_line = optimized_line.replace(pattern, replacement, 1)
                    modified = True
                    changes.append({
                        "type": "line_split",
                        "line": i + 1,
                        "original": original_line,
                        "optimized": optimized_line
                    })
                    break
            
            if modified:
                for opt_line in optimized_line.split('\n'):
                    optimized_lines.append(opt_line)
            else:
                optimized_lines.append(line)
        
        optimized_content = '\n'.join(optimized_lines)
        
        return optimized_content, {
            "strategy": self.name,
            "changes_count": len(changes),
            "changes": changes
        }


class ImportOptimizer(OptimizationStrategy):
    """导入优化策略"""
    
    def __init__(self):
        super().__init__(
            name="import_optimizer",
            description="优化和整理import语句"
        )
    
    def analyze(self, file_path: str, content: str) -> Dict[str, Any]:
        """分析导入语句"""
        lines = content.splitlines()
        import_issues = []
        imports = []
        
        in_import_section = True
        current_imports = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 检查是否仍在导入部分
            if stripped and not stripped.startswith(('import ', 'from ', '#')) and in_import_section:
                in_import_section = False
            elif stripped.startswith(('import ', 'from ')) and in_import_section:
                current_imports.append((i + 1, stripped))
                
                # 检查问题
                if '*' in stripped:
                    import_issues.append({
                        "type": "wildcard_import",
                        "line": i + 1,
                        "description": "通配符导入 (*)"
                    })
                elif stripped.count('import') > 1:
                    import_issues.append({
                        "type": "multiple_imports",
                        "line": i + 1,
                        "description": "一行多个导入"
                    })
        
        return {
            "file_path": file_path,
            "imports_found": len(current_imports),
            "issues_found": len(import_issues),
            "import_issues": import_issues,
            "can_optimize": len(import_issues) > 0
        }
    
    def apply(self, file_path: str, content: str) -> Tuple[str, Dict[str, Any]]:
        """应用导入优化"""
        lines = content.splitlines()
        optimized_lines = []
        changes = []
        
        # 简单的优化：按字母排序导入
        imports_section = []
        other_lines = []
        in_imports = True
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                imports_section.append(line)
            elif stripped == '' and in_imports:
                imports_section.append(line)  # 保留空行
            else:
                in_imports = False
                other_lines.append(line)
        
        # 重构导入部分
        if len(imports_section) > 2:  # 如果有多个导入
            # 提取导入语句
            import_lines = [line for line in imports_section if line.strip().startswith(('import ', 'from '))]
            non_import_lines = [line for line in imports_section if not line.strip().startswith(('import ', 'from '))]
            
            # 简单排序
            import_lines.sort(key=lambda x: x.lower())
            
            # 重建文件内容
            optimized_imports = non_import_lines + import_lines + ['']  # 添加空行分隔
            
            if optimized_imports != imports_section:
                changes.append({
                    "type": "import_organize",
                    "description": "重新组织import语句",
                    "original_count": len(import_lines),
                    "optimized_count": len(import_lines)
                })
            
            optimized_content = '\n'.join(optimized_imports + other_lines)
        else:
            optimized_content = content  # 没有足够 imports 需要优化
        
        return optimized_content, {
            "strategy": self.name,
            "changes_count": len(changes),
            "changes": changes
        }


class CommentOptimizer(OptimizationStrategy):
    """注释优化策略"""
    
    def __init__(self):
        super().__init__(
            name="comment_optimizer", 
            description="优化注释质量和格式"
        )
    
    def analyze(self, file_path: str, content: str) -> Dict[str, Any]:
        """分析注释"""
        lines = content.splitlines()
        comment_issues = []
        
        for i, line in enumerate(lines):
            # 检查TODO/FIXME注释
            lower_line = line.lower()
            if 'todo' in lower_line or 'fixme' in lower_line:
                comment_issues.append({
                    "type": "todo_comment",
                    "line": i + 1,
                    "description": "包含TODO/FIXME注释"
                })
            
            # 检查空的注释块
            stripped = line.strip()
            if stripped.startswith('#') and len(stripped) <= 1:
                comment_issues.append({
                    "type": "empty_comment",
                    "line": i + 1,
                    "description": "空的注释行"
                })
        
        return {
            "file_path": file_path,
            "issues_found": len(comment_issues),
            "comment_issues": comment_issues,
            "can_optimize": len(comment_issues) > 0
        }
    
    def apply(self, file_path: str, content: str) -> Tuple[str, Dict[str, Any]]:
        """应用注释优化"""
        lines = content.splitlines()
        optimized_lines = []
        changes = []
        
        for line in lines:
            stripped = line.strip()
            
            # 优化空注释
            if stripped.startswith('#') and len(stripped) <= 1:
                changes.append({
                    "type": "remove_empty_comment",
                    "line": len(optimized_lines) + 1,
                    "original": line
                })
                continue  # 跳过空注释
            
            # 优化TODO注释（添加格式）
            if 'todo' in stripped.lower():
                optimized_line = self._format_comment_with_todo(line)
                if optimized_line != line:
                    changes.append({
                        "type": "format_todo_comment",
                        "line": len(optimized_lines) + 1,
                        "original": line,
                        "optimized": optimized_line
                    })
                    line = optimized_line
            
            optimized_lines.append(line)
        
        optimized_content = '\n'.join(optimized_lines)
        
        return optimized_content, {
            "strategy": self.name,
            "changes_count": len(changes),
            "changes": changes
        }
    
    def _format_comment_with_todo(self, line: str) -> str:
        """格式化TODO注释"""
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        
        if 'todo' in stripped.lower():
            # 简单格式化：确保TODO后面有描述
            if stripped.lower() == '#todo':
                return f"{indent}# TODO: "
            elif stripped.lower().startswith('#todo') and ':' not in stripped:
                return stripped.replace('TODO', 'TODO: ')
        
        return line


class FunctionLengthOptimizer(OptimizationStrategy):
    """函数长度优化策略"""
    
    def __init__(self):
        super().__init__(
            name="function_length_optimizer",
            description="检测过长的函数并建议拆分"
        )
        # 从配置加载
        self.max_function_length = get_config().get('optimization.max_function_length', 50)
    
    def analyze(self, file_path: str, content: str) -> Dict[str, Any]:
        """分析函数长度"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"file_path": file_path, "error": "语法错误，无法解析"}
        
        long_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 计算函数行数
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                function_length = end_line - start_line + 1
                
                if function_length > self.max_function_length:
                    long_functions.append({
                        "name": node.name,
                        "line_number": start_line,
                        "length": function_length,
                        "type": node.__class__.__name__
                    })
        
        return {
            "file_path": file_path,
            "issues_found": len(long_functions),
            "long_functions": long_functions,
            "can_optimize": len(long_functions) > 0
        }
    
    def apply(self, file_path: str, content: str) -> Tuple[str, Dict[str, Any]]:
        """应用函数长度优化（目前只是添加注释标记）"""
        # 函数拆分比较复杂，先添加TODO注释
        lines = content.splitlines()
        optimized_lines = []
        changes = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content, {"strategy": self.name, "changes_count": 0, "changes": []}
        
        # 找到长函数所在行
        long_function_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                function_length = end_line - start_line + 1
                
                if function_length > self.max_function_length:
                    long_function_lines.add(start_line)
        
        # 在长函数前添加注释
        for i, line in enumerate(lines):
            optimized_lines.append(line)
            if i + 1 in long_function_lines and "# TODO" not in line:
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ""
                
                comment_line = f"{indent}# TODO: 函数过长 ({self.max_function_length}+ 行)，建议拆分"
                optimized_lines.insert(-1, comment_line)
                
                changes.append({
                    "type": "add_todo_comment",
                    "line": i + 1,
                    "description": f"标记过长函数: {line.strip()}"
                })
        
        optimized_content = '\n'.join(optimized_lines)
        
        return optimized_content, {
            "strategy": self.name,
            "changes_count": len(changes),
            "changes": changes
        }


class VariableNamingOptimizer(OptimizationStrategy):
    """变量命名优化策略"""
    
    def __init__(self):
        super().__init__(
            name="variable_naming_optimizer",
            description="改善变量命名质量"
        )
        # 检查的不良命名模式
        self.bad_patterns = [
            (r'\b[a-z]\b', '单字母变量名'),
            (r'\b[a-z]{2}\b', '过短变量名'),
            (r'\b[a-z]+_[a-z]+\b', '推荐使用snake_case'),
            (r'\b[A-Z][a-z]+[A-Z][a-z]+\b', '推荐使用snake_case而非camelCase')
        ]
    
    def analyze(self, file_path: str, content: str) -> Dict[str, Any]:
        """分析变量命名"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"file_path": file_path, "error": "语法错误，无法解析"}
        
        naming_issues = []
        lines = content.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                var_name = node.id
                
                # 检查命名模式
                for pattern, description in self.bad_patterns:
                    if re.match(pattern, var_name) and var_name not in ['i', 'j', 'k', 'x', 'y', 'z']:
                        line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        
                        naming_issues.append({
                            "variable": var_name,
                            "line_number": node.lineno,
                            "issue": description,
                            "context": line_text.strip()[:50]
                        })
        
        return {
            "file_path": file_path,
            "issues_found": len(naming_issues),
            "naming_issues": naming_issues,
            "can_optimize": len(naming_issues) > 0
        }
    
    def apply(self, file_path: str, content: str) -> Tuple[str, Dict[str, Any]]:
        """应用变量命名优化（添加建议注释）"""
        lines = content.splitlines()
        optimized_lines = []
        changes = []
        
        for i, line in enumerate(lines):
            optimized_lines.append(line)
            
            # 简单的命名问题检测和注释添加
            if ('i = ' in line or 'x = ' in line) and 'for' not in line:
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ""
                
                if "# " not in line:  # 避免重复注释
                    comment_line = f"{indent}# TODO: 考虑使用更具描述性的变量名"
                    optimized_lines.append(comment_line)
                    
                    changes.append({
                        "type": "suggest_naming",
                        "line": i + 1,
                        "original": line.strip(),
                        "suggestion": "使用更具描述性的变量名"
                    })
        
        optimized_content = '\n'.join(optimized_lines)
        
        return optimized_content, {
            "strategy": self.name,
            "changes_count": len(changes),
            "changes": changes
        }


class EmptyLineOptimizer(OptimizationStrategy):
    """空行规范化优化策略"""
    
    def __init__(self):
        super().__init__(
            name="empty_line_optimizer",
            description="规范代码中的空行使用"
        )
    
    def analyze(self, file_path: str, content: str) -> Dict[str, Any]:
        """分析空行使用"""
        lines = content.splitlines()
        empty_line_issues = []
        
        # 检测连续空行
        consecutive_empty = 0
        for i, line in enumerate(lines):
            if not line.strip():
                consecutive_empty += 1
                if consecutive_empty > 2:  # 超过2个连续空行
                    empty_line_issues.append({
                        "line_number": i + 1,
                        "type": "excessive_empty_lines",
                        "consecutive": consecutive_empty
                    })
            else:
                consecutive_empty = 0
        
        # 检测函数/类定义前缺少空行
        try:
            tree = ast.parse(content)
            definition_lines = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definition_lines.append(node.lineno)
            
            for line_num in definition_lines:
                if line_num > 1:
                    prev_line = lines[line_num - 2] if line_num > 1 else ""
                    if prev_line.strip():  # 前一行非空
                        empty_line_issues.append({
                            "line_number": line_num,
                            "type": "missing_empty_line_before_definition",
                            "definition": lines[line_num - 1].strip()
                        })
        except SyntaxError:
            pass
        
        return {
            "file_path": file_path,
            "issues_found": len(empty_line_issues),
            "empty_line_issues": empty_line_issues,
            "can_optimize": len(empty_line_issues) > 0
        }
    
    def apply(self, file_path: str, content: str) -> Tuple[str, Dict[str, Any]]:
        """应用空行规范化"""
        lines = content.splitlines()
        optimized_lines = []
        changes = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # 压缩连续空行到最多2个
            if not line_stripped:
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                
                empty_count = j - i
                if empty_count > 2:
                    # 只保留2个空行
                    optimized_lines.extend(["", ""])
                    changes.append({
                        "type": "compress_empty_lines",
                        "line": i + 1,
                        "removed": empty_count - 2
                    })
                    i = j
                    continue
                else:
                    optimized_lines.append(line)
            else:
                # 检查是否需要添加空行（在定义之前）
                if (line_stripped.startswith(('def ', 'class ', 'async def ')) and 
                    i > 0 and lines[i-1].strip()):
                    
                    # 添加空行
                    indent_match = re.match(r'^(\s*)', line)
                    indent = indent_match.group(1) if indent_match else ""
                    
                    optimized_lines.append("")
                    optimized_lines.append(line)
                    
                    changes.append({
                        "type": "add_empty_line_before_definition",
                        "line": i + 1,
                        "definition": line.strip()
                    })
                else:
                    optimized_lines.append(line)
            
            i += 1
        
        # 移除文件末尾多余空行
        while optimized_lines and not optimized_lines[-1].strip():
            optimized_lines.pop()
            if changes and changes[-1]["type"] == "compress_empty_lines":
                changes[-1]["removed"] += 1
        
        optimized_content = '\n'.join(optimized_lines)
        
        return optimized_content, {
            "strategy": self.name,
            "changes_count": len(changes),
            "changes": changes
        }


class DuplicateCodeOptimizer(OptimizationStrategy):
    """代码重复检测优化策略"""
    
    def __init__(self):
        super().__init__(
            name="duplicate_code_optimizer",
            description="检测重复代码块并建议提取"
        )
        self.min_duplicate_lines = 3
    
    def analyze(self, file_path: str, content: str) -> Dict[str, Any]:
        """分析代码重复"""
        lines = content.splitlines()
        duplicate_groups = []
        
        # 简单的重复检测：查找相似的代码块
        code_blocks = []
        
        # 提取代码块（忽略空行和注释）
        current_block = []
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                current_block.append(line.strip())
            else:
                if len(current_block) >= self.min_duplicate_lines:
                    block_text = '\n'.join(current_block)
                    code_blocks.append((block_text, len(current_block), 
                                      len(lines) - len(current_block)))  # 开始行号
                current_block = []
        
        # 处理最后一个块
        if len(current_block) >= self.min_duplicate_lines:
            block_text = '\n'.join(current_block)
            code_blocks.append((block_text, len(current_block), len(lines) - len(current_block)))
        
        # 查找重复
        seen_blocks = {}
        for block_text, block_length, start_line in code_blocks:
            if block_text in seen_blocks:
                first_occurrence = seen_blocks[block_text]
                duplicate_groups.append({
                    "block_hash": hash(block_text) % 10000,
                    "first_occurrence": first_occurrence,
                    "duplicate_line": start_line,
                    "length": block_length,
                    "similarity": "100%"  # 完全相同
                })
            else:
                seen_blocks[block_text] = start_line
        
        return {
            "file_path": file_path,
            "duplicates_found": len(duplicate_groups),
            "duplicate_groups": duplicate_groups,
            "can_optimize": len(duplicate_groups) > 0
        }
    
    def apply(self, file_path: str, content: str) -> Tuple[str, Dict[str, Any]]:
        """应用重复代码优化（添加标记注释）"""
        # 简化实现：在重复代码处添加注释
        result = self.analyze(file_path, content)
        lines = content.splitlines()
        optimized_lines = []
        changes = []
        
        for group in result.get("duplicate_groups", []):
            duplicate_line = group["duplicate_line"]
            if duplicate_line > 0 and duplicate_line <= len(lines):
                # 在重复代码处添加注释
                line_at_duplicate = lines[duplicate_line - 1]
                indent_match = re.match(r'^(\s*)', line_at_duplicate)
                indent = indent_match.group(1) if indent_match else ""
                
                comment = f"{indent}# TODO: 发现重复代码，建议提取为函数"
                lines.insert(duplicate_line - 1, comment)
                
                changes.append({
                    "type": "mark_duplicate_code",
                    "line": duplicate_line,
                    "similarity": group["similarity"],
                    "length": group["length"]
                })
        
        optimized_content = '\n'.join(lines)
        
        return optimized_content, {
            "strategy": self.name,
            "changes_count": len(changes),
            "changes": changes
        }


class CodeOptimizer:
    """代码优化器主类"""
    
    def __init__(self):
        self.strategies: List[OptimizationStrategy] = [
            LineLengthOptimizer(),
            ImportOptimizer(),
            CommentOptimizer(),
            FunctionLengthOptimizer(),
            VariableNamingOptimizer(),
            EmptyLineOptimizer(),
            DuplicateCodeOptimizer(),
        ]
        self.optimization_log = []
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析文件，获取所有优化建议"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                "file_path": file_path,
                "error": f"无法读取文件: {e}"
            }
        
        file_analysis = {
            "file_path": file_path,
            "content_stats": {
                "lines": len(content.splitlines()),
                "characters": len(content),
            },
            "strategy_results": {}
        }
        
        total_issues = 0
        optimizable_strategies = []
        
        # 运行每个策略
        for strategy in self.strategies:
            try:
                result = strategy.analyze(file_path, content)
                file_analysis["strategy_results"][strategy.name] = result
                
                if result.get("can_optimize", False):
                    total_issues += result.get("issues_found", 0)
                    optimizable_strategies.append({
                        "name": strategy.name,
                        "description": strategy.description,
                        "issues_count": result.get("issues_found", 0)
                    })
                    
            except Exception as e:
                file_analysis["strategy_results"][strategy.name] = {
                    "error": f"策略分析失败: {e}"
                }
        
        file_analysis.update({
            "total_issues": total_issues,
            "optimizable_strategies": optimizable_strategies,
            "needs_optimization": total_issues > 0
        })
        
        return file_analysis
    
    def optimize_file(self, file_path: str, selected_strategies: Optional[List[str]] = None, apply_changes: bool = False) -> Dict[str, Any]:
        """
        优化文件
        
        Args:
            file_path: 目标文件路径
            selected_strategies: 要应用的策略列表
            apply_changes: 是否直接保存修改（默认为False，返回优化后的内容）
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                "file_path": file_path,
                "error": f"无法读取文件: {e}"
            }
        
        # 确定要应用的策略
        if selected_strategies:
            strategies_to_apply = [s for s in self.strategies if s.name in selected_strategies]
        else:
            strategies_to_apply = self.strategies
        
        optimization_results = []
        modified_content = content
        applied_changes = []
        
        # 依次应用策略
        for strategy in strategies_to_apply:
            try:
                optimized_content, change_info = strategy.apply(file_path, modified_content)
                
                if optimized_content != modified_content:  # 有变更
                    modified_content = optimized_content
                    optimization_results.append(change_info)
                    applied_changes.extend(change_info.get('changes', []))
                    
            except Exception as e:
                optimization_results.append({
                    "strategy": strategy.name,
                    "error": f"应用策略失败: {e}"
                })
        
        # 构建结果
        result = {
            "file_path": file_path,
            "optimization_applied": modified_content != content,
            "strategies_applied": [s.name for s in strategies_to_apply],
            "changes_count": len(applied_changes),
            "specific_changes": applied_changes,
            "optimization_results": optimization_results,
            "optimized_content": modified_content if modified_content != content else None,
            "original_content": content
        }
        
        # 如果要求直接保存
        if apply_changes and modified_content != content:
            try:
                # 创建备份
                backup_path = file_path + '.backup'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # 保存优化后的文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                result["backup_created"] = True
                result["backup_path"] = backup_path
                
            except Exception as e:
                result["error"] = f"保存文件失败: {e}"
        
        return result


# 便捷函数
def analyze_code_for_optimization(file_path: str) -> Dict[str, Any]:
    """分析代码的优化潜力"""
    optimizer = CodeOptimizer()
    return optimizer.analyze_file(file_path)


def optimize_code_file(file_path: str, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
    """优化单个代码文件"""
    optimizer = CodeOptimizer()
    return optimizer.optimize_file(file_path, strategies)


if __name__ == "__main__":
    # 测试优化策略
    print("🔧 代码优化策略测试")
    print("=" * 60)
    
    # 测试当前文件
    test_file = __file__
    
    print("1. 分析优化潜力...")
    analysis = analyze_code_for_optimization(test_file)
    
    if 'error' in analysis:
        print(f"❌ 分析失败: {analysis['error']}")
    else:
        print(f"✅ 分析完成:")
        print(f"   文件: {os.path.basename(test_file)}")
        print(f"   总行数: {analysis['content_stats']['lines']}")
        print(f"   发现问题: {analysis['total_issues']} 个")
        
        if analysis.get('optimizable_strategies'):
            print(f"\n📋 可优化策略:")
            for strategy in analysis['optimizable_strategies']:
                print(f"   - {strategy['name']}: {strategy['issues_count']} 个问题")
                print(f"     {strategy['description']}")
        
        print("\n2. 尝试应用优化...")
        # 测试多个新策略
        test_strategies = ['comment_optimizer', 'empty_line_optimizer', 'variable_naming_optimizer']
        result = optimize_code_file(test_file, test_strategies)
        
        if result.get('optimization_applied'):
            print(f"✅ 优化完成:")
            print(f"   变更数: {result['changes_count']}")
            print(f"   备份文件: {test_file}.backup")
            print(f"   应用的策略: {', '.join(result['strategies_applied'])}")
        else:
            print(f"ℹ️ 未应用优化: {result.get('error', '无需要优化内容')}")
    
    print("\n✅ 优化策略测试完成")