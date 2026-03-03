#!/usr/bin/env python3
"""
文件扫描器 - 扫描项目中的代码文件
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional
import fnmatch

from src.config.manager import get_config


class FileScanner:
    """项目文件扫描器"""
    
    def __init__(self, project_path: str):
        """初始化扫描器"""
        self.project_path = Path(project_path).resolve()
        if not self.project_path.exists():
            raise ValueError(f"项目路径不存在: {project_path}")
        if not self.project_path.is_dir():
            raise ValueError(f"项目路径不是目录: {project_path}")
        
        # 加载配置
        config = get_config()
        scanner_config = config.get_file_scanner_config()
        
        # 代码文件扩展名
        self.code_extensions = scanner_config.get('code_extensions', {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.cc', '.cxx', '.h', '.hpp'],
            'go': ['.go'],
            'rust': ['.rs'],
            'ruby': ['.rb'],
            'php': ['.php'],
            'csharp': ['.cs'],
        })
        
        # 排除目录
        self.default_exclude_dirs = set(scanner_config.get('exclude_dirs', [
            '__pycache__', '.git', '.svn', '.hg', '.vscode', '.idea',
            'node_modules', 'venv', 'env', '.venv', '.env',
            'dist', 'build', 'target', '.pytest_cache', '.mypy_cache'
        ]))
    
    def scan_project(self, 
                     extensions: Optional[List[str]] = None,
                     exclude_dirs: Optional[Set[str]] = None,
                     exclude_files: Optional[Set[str]] = None) -> Dict[str, List[str]]:
        """
        扫描项目中的文件
        
        Args:
            extensions: 要扫描的文件扩展名，如 ['.py', '.js']
            exclude_dirs: 要排除的目录模式
            exclude_files: 要排除的文件模式
            
        Returns:
            字典 {扩展名: [文件路径列表]}
        """
        if exclude_dirs is None:
            exclude_dirs = self.default_exclude_dirs
        if exclude_files is None:
            exclude_files = set()
        
        result = {}
        extensions_set = set(extensions) if extensions else set()
        
        # 如果没有指定扩展名，扫描所有代码扩展名
        if not extensions:
            for ext_list in self.code_extensions.values():
                extensions_set.update(ext_list)
        
        # 初始化结果字典
        for ext in extensions_set:
            result[ext] = []
        
        for root, dirs, files in os.walk(self.project_path):
            # 排除不需要的目录
            dirs[:] = [
                d for d in dirs 
                if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_dirs)
            ]
            
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.project_path)
                
                # 检查是否应该排除该文件
                should_exclude = False
                for pattern in exclude_files:
                    if fnmatch.fnmatch(str(relative_path), pattern):
                        should_exclude = True
                        break
                       
                if should_exclude:
                    continue
                
                # 按扩展名分类
                for ext in extensions_set:
                    if file.endswith(ext):
                        result[ext].append(str(relative_path))
                        break
        
        return result
    
    def scan_python_files(self, exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """专门扫描Python文件"""
        exclude = set(exclude_patterns) if exclude_patterns else set()
        
        # 从配置获取排除模式
        config = get_config()
        skip_files = config.get('analysis.skip_files', [])
        exclude.update(skip_files)
        
        all_files = self.scan_project(
            extensions=self.code_extensions.get('python', ['.py']),
            exclude_files=exclude
        )
        
        return all_files.get('.py', [])
    
    def get_project_stats(self) -> Dict[str, any]:
        """获取项目统计信息"""
        stats = {
            'project_path': str(self.project_path),
            'total_files_by_type': {},
            'lines_by_type': {},
            'total_lines': 0,
        }
        
        # 扫描主要代码文件类型
        python_files = self.scan_python_files()
        stats['total_files_by_type']['python'] = len(python_files)
        
        # 计算Python文件行数
        python_lines = 0
        for file in python_files:
            try:
                full_path = self.project_path / file
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    python_lines += len([l for l in lines if l.strip()])
            except Exception as e:
                print(f"无法读取文件 {file}: {e}")
        
        stats['lines_by_type']['python'] = python_lines
        stats['total_lines'] = python_lines
        
        return stats
    
    def find_large_files(self, max_size_kb: int = None) -> List[Dict[str, str]]:
        """查找大的代码文件"""
        if max_size_kb is None:
            max_size_kb = get_config().get('file_scanner.max_file_size_kb', 100)
            
        large_files = []
        python_files = self.scan_python_files()
        
        for file_path in python_files:
            full_path = self.project_path / file_path
            try:
                size_kb = os.path.getsize(full_path) / 1024
                if size_kb > max_size_kb:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                    
                    large_files.append({
                        'path': file_path,
                        'size_kb': round(size_kb, 1),
                        'lines': lines,
                        'avg_line_length': round(os.path.getsize(full_path) / max(lines, 1), 1)
                    })
            except Exception as e:
                pass
        
        # 按大小排序
        large_files.sort(key=lambda x: x['size_kb'], reverse=True)
        return large_files
    
    def find_duplicate_filenames(self) -> Dict[str, List[str]]:
        """查找可能重复的文件名"""
        python_files = self.scan_python_files()
        filename_map = {}
        
        for file_path in python_files:
            filename = os.path.basename(file_path)
            if filename not in filename_map:
                filename_map[filename] = []
            filename_map[filename].append(file_path)
        
        # 只返回重复的文件名
        duplicates = {
            name: paths 
            for name, paths in filename_map.items() 
            if len(paths) > 1
        }
        
        return duplicates


def main():
    """测试文件扫描器"""
    test_path = os.getcwd()
    scanner = FileScanner(test_path)
    
    print(f"🔍 扫描项目: {test_path}")
    print("=" * 60)
    
    # 扫描Python文件
    python_files = scanner.scan_python_files()
    print(f"📁 找到 {len(python_files)} 个Python文件")
    
    if python_files:
        print("示例文件:")
        for file in python_files[:10]:
            print(f"  - {file}")
        if len(python_files) > 10:
            print(f"  ... 还有 {len(python_files) - 10} 个文件")
    
    # 获取统计信息
    stats = scanner.get_project_stats()
    print(f"\n📊 项目统计:")
    print(f"  Python文件数: {stats['total_files_by_type'].get('python', 0)}")
    print(f"  总代码行数: {stats['total_lines']}")
    
    # 查找大文件
    large_files = scanner.find_large_files(max_size_kb=10)
    if large_files:
        print(f"\n⚠️  发现 {len(large_files)} 个大文件 (>10KB):")
        for file in large_files:
            print(f"  - {file['path']} ({file['size_kb']} KB, {file['lines']} 行)")
    
    # 查找重复文件名
    duplicates = scanner.find_duplicate_filenames()
    if duplicates:
        print(f"\n🔴 发现 {len(duplicates)} 个重复文件名:")
        for name, paths in list(duplicates.items())[:5]:
            print(f"  - {name} 出现在:")
            for path in paths:
                print(f"      {path}")


if __name__ == "__main__":
    main()