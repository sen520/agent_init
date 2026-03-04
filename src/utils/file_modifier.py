#!/usr/bin/env python3
"""
安全的文件修改工具 - 支持备份和回滚
"""
import os
import shutil
import ast
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from src.config.manager import get_config


class FileModifier:
    """安全的文件修改器"""
    
    def __init__(self, backup_dir: str = None):
        # 从配置加载备份目录
        if backup_dir is None:
            backup_dir = get_config().get('file_modifier.backup_dir', '.optimization_backups')
        
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.modified_files = []
        self.backups = {}  # file_path -> backup_path
    
    def backup_file(self, file_path: str) -> str:
        """
        创建文件备份
        
        Returns:
            备份文件路径
        """
        original = Path(file_path)
        if not original.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 生成备份文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{original.stem}_{timestamp}{original.suffix}.bak"
        backup_path = self.backup_dir / backup_name
        
        # 创建备份
        shutil.copy2(file_path, backup_path)
        self.backups[file_path] = str(backup_path)
        
        return str(backup_path)
    
    def write_file(self, file_path: str, content: str, create_backup: bool = True) -> Tuple[bool, str]:
        """
        安全地写入文件（带备份和语法验证）
        
        Args:
            file_path: 目标文件路径
            content: 新内容
            create_backup: 是否创建备份
            
        Returns:
            (成功, 消息)
        """
        try:
            # 1. 如果是 Python 文件，验证语法
            if file_path.endswith('.py'):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    return False, f"语法错误: {e}"
            
            # 2. 创建备份
            if create_backup:
                backup_path = self.backup_file(file_path)
            
            # 3. 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.modified_files.append(file_path)
            
            if create_backup:
                return True, f"已修改 (备份: {Path(backup_path).name})"
            return True, "已修改"
            
        except Exception as e:
            return False, f"写入失败: {e}"
    
    def rollback_file(self, file_path: str) -> Tuple[bool, str]:
        """
        回滚文件到备份版本
        
        Returns:
            (成功, 消息)
        """
        if file_path not in self.backups:
            return False, "没有找到备份"
        
        backup_path = self.backups[file_path]
        
        try:
            shutil.copy2(backup_path, file_path)
            return True, f"已回滚到备份: {Path(backup_path).name}"
        except Exception as e:
            return False, f"回滚失败: {e}"
    
    def rollback_all(self) -> Tuple[int, int]:
        """
        回滚所有修改过的文件
        
        Returns:
            (成功数, 失败数)
        """
        success = 0
        failed = 0
        
        for file_path in self.modified_files:
            ok, msg = self.rollback_file(file_path)
            if ok:
                success += 1
            else:
                failed += 1
        
        return success, failed
    
    def cleanup_old_backups(self, days: int = None) -> int:
        """
        清理旧的备份文件
        
        Args:
            days: 保留天数（从配置读取默认值）
            
        Returns:
            删除的文件数
        """
        from datetime import timedelta
        
        if days is None:
            days = get_config().get('file_modifier.backup_retention_days', 7)
        
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        
        for backup_file in self.backup_dir.glob("*.bak"):
            try:
                mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if mtime < cutoff:
                    backup_file.unlink()
                    deleted += 1
            except Exception as e:
                logger.debug(f"删除旧备份失败: {e}")
                pass
        
        return deleted


def apply_optimization_safely(
    file_path: str,
    optimized_content: str,
    modifier: FileModifier
) -> Dict[str, Any]:
    """
    安全地应用优化
    
    Args:
        file_path: 目标文件
        optimized_content: 优化后的内容
        modifier: 文件修改器实例
        
    Returns:
        结果字典
    """
    result = {
        "success": False,
        "file_path": file_path,
        "changes_applied": False,
        "backup_path": None,
        "message": "",
        "error": None
    }
    
    try:
        # 读取原始内容
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 如果没有变化，跳过
        if original_content == optimized_content:
            result["message"] = "文件无需修改"
            result["success"] = True
            return result
        
        # 应用修改
        success, message = modifier.write_file(file_path, optimized_content)
        
        if success:
            result["success"] = True
            result["changes_applied"] = True
            result["backup_path"] = modifier.backups.get(file_path)
            result["message"] = message
        else:
            result["error"] = message
            
    except Exception as e:
        result["error"] = str(e)
    
    return result
