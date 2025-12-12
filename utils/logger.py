import os
import sys
from loguru import logger


# 2. 自定义控制台 sink（实现截断）
def truncated_console_sink(message):
    """控制台输出截断日志（最大200字符）"""
    max_length = int(os.environ.get('LOG_MAX_LENGTH', '200'))  # 单行最大长度
    formatted = str(message)

    # 按行处理多行日志
    lines = []
    for line in formatted.splitlines():
        if len(line) > max_length:

            lines.append(f"{line[:max_length]}... [TRUNCATED {len(line) - max_length} chars]")
        else:
            lines.append(line)

    sys.stderr.write("\n".join(lines) + "\n")


def create_logger():
    # 1. 移除默认 sink 避免重复输出
    logger.remove()

    # 3. 添加控制台 sink（截断输出）
    logger.add(
        sink=truncated_console_sink,
        level="DEBUG",
        colorize=True,  # 保持颜色输出
        backtrace=True,  # 启用异常回溯
        diagnose=True  # 显示诊断信息
    )

    # 4. 添加文件 sink（完整日志）
    logger.add(
        sink=os.environ.get('LOG_FILE', 'test.log'),
        rotation=os.environ.get('LOG_SIZE', '20 MB'),  # 按大小轮转
        retention=os.environ.get('LOG_RETENTION', '30 days'),  # 保留30天
        compression=os.environ.get('LOG_COMPRESSION', 'zip'),  # 压缩存档
        level=os.environ.get('LOG_LEVEL', 'DEBUG'),  # 记录所有级别
        enqueue=True,  # 多进程安全
        serialize=False,  # 禁用序列化（可读格式）
        backtrace=True,  # 完整异常回溯
        diagnose=True,  # 完整诊断信息
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}"  # 详细格式
    )
    return logger


# 测试日志
def generate_long_log():
    """生成超长日志的测试函数"""
    logger.info("Start transaction: T-001")
    logger.debug("Data payload: " + "x" * 1000)  # 1000字符的超长日志
    # try:
    #     1 / 0
    # except Exception:
    #     logger.exception("Calculation failed")


if __name__ == "__main__":
    logger = create_logger()
    generate_long_log()
    # logger.success("Test completed")
