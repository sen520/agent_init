#!/usr/bin/env python3
"""
测试 sqlite_db - src/utils/sqlite_db.py
"""
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.sqlite_db import SQLiteDB


class TestSQLiteDB:
    """SQLiteDB 测试类"""
    
    def test_init_with_memory_db(self):
        """测试内存数据库初始化"""
        db = SQLiteDB(':memory:')
        assert db is not None
        assert db.conn is not None
    
    def test_init_with_file_db(self, tmp_path):
        """测试文件数据库初始化"""
        db_path = tmp_path / "test.db"
        db = SQLiteDB(str(db_path))
        
        assert db is not None
        assert db.conn is not None
        assert db_path.exists()
    
    def test_execute_select(self):
        """测试执行查询"""
        db = SQLiteDB(':memory:')
        
        # 创建表
        db.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        db.execute("INSERT INTO test VALUES (1, 'test')")
        
        # 查询
        result = db.execute("SELECT * FROM test")
        
        assert result is not None
    
    def test_execute_fetchall(self):
        """测试获取所有结果"""
        db = SQLiteDB(':memory:')
        
        db.execute("CREATE TABLE test (id INTEGER)")
        db.execute("INSERT INTO test VALUES (1)")
        db.execute("INSERT INTO test VALUES (2)")
        
        result = db.execute("SELECT * FROM test", fetch=True)
        
        assert len(result) == 2
    
    def test_close(self):
        """测试关闭数据库"""
        db = SQLiteDB(':memory:')
        db.close()
        
        # 应该成功关闭
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
