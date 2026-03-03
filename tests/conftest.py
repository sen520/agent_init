"""
pytest配置文件
"""
import pytest
import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_python_file():
    """示例Python文件内容"""
    return '''
def example_function():
    # 这是一个示例函数
    x = 1
    y = 2
    return x + y

class ExampleClass:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value
'''


@pytest.fixture
def sample_analysis_result():
    """示例分析结果"""
    return {
        "file_path": "example.py",
        "issues": [
            {
                "type": "complexity",
                "message": "函数过长",
                "line": 1,
                "severity": "medium"
            }
        ],
        "metrics": {
            "lines_of_code": 15,
            "complexity": 3
        }
    }


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        "OPENAI_API_KEY": "test-key",
        "max_iterations": 3,
        "log_level": "INFO"
    }