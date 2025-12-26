import sqlite3
from typing import List, Any, Dict


class SQLiteDB:
    def __init__(self, db_path: str = "./database.db"):
        """初始化SQLite数据库"""
        self.db_path = db_path

    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 启用行工厂，支持列名访问
        return conn

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行SQL查询并返回结果"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # 获取列名
            columns = [description[0] for description in cursor.description] if cursor.description else []
            # 将结果转换为字典列表
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            conn.close()
            return results

        except sqlite3.Error as e:
            return [{"error": f"SQL错误: {str(e)}"}]

    def execute_update(self, query: str, params: tuple = None) -> str:
        """执行更新操作（INSERT/UPDATE/DELETE）"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            conn.commit()
            affected_rows = cursor.rowcount
            conn.close()

            return f"操作成功，影响行数: {affected_rows}"

        except sqlite3.Error as e:
            raise e
