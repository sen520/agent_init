# 🤖 LangGraph 自我优化代码助手

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](./)
[![Coverage](https://img.shields.io/badge/coverage-63%25-brightgreen.svg)](./)

**基于 LangGraph 工作流的智能代码优化系统**

🤖 自动分析 · 🎯 智能优化 · 📊 质量提升 · ✅ 测试覆盖

</div>

---

## 📖 项目简介

这是一个基于 **LangGraph 工作流框架** 构建的智能代码自我优化系统，能够自动扫描代码项目，识别质量问题，并提供基于最佳实践的优化建议和自动修复方案。

### ✨ 核心特性

- 🔍 **智能代码扫描** - 递归扫描项目，支持多种编程语言
- 🧠 **AST深度分析** - 基于抽象语法树的代码结构解析
- 🎨 **7大优化策略** - 全覆盖代码质量检查维度
- 🔄 **LangGraph 工作流** - 状态驱动的优化流程
- 📊 **详细报告** - HTML格式的可视化分析报告
- 🛡️ **安全可靠** - 自动备份和验证机制
- ✅ **高测试覆盖** - 63% 代码覆盖率（核心模块 80%+）

---

## 🚀 快速开始

### 📋 环境要求
```
✅ Python 3.9+
💾 内存: 512MB+
📦 磁盘: 100MB+
```

### 🛠️ 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd agent_init

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate        # Linux/Mac
# 或 venv\Scripts\activate      # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -m pytest tests/ -v       # 运行测试
python main.py help             # 查看帮助
```

### 🎯 立即体验

```bash
# 显示帮助信息
python main.py help

# 运行测试套件
python -m pytest tests/ -v

# 完整优化分析
python main.py full
```

---

## 📋 使用指南

### 🎮 基本命令

```bash
# 📖 查看帮助
python main.py help

# 🧪 快速测试模式
python main.py test
# 快速扫描当前项目，发现主要问题

# 🚀 完整优化模式
python main.py full
# 深度分析 + 多轮优化 + HTML报告
```

### 🧪 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行测试并生成覆盖率报告
python -m pytest tests/ --cov=src --cov-report=html

# 只运行单元测试
python -m pytest tests/ -m unit -v
```

---

## 🎨 功能详解

### 🔍 **代码分析能力**

| 分析类型 | 功能描述 | 支持语言 | 输出结果 |
|----------|----------|----------|----------|
| **语法分析** | AST解析和结构检查 | Python | 语法错误、结构问题 |
| **复杂度分析** | 圈复杂度和认知复杂度 | Python | 复杂度评分 |
| **安全扫描** | 安全漏洞检测 | Python | 安全问题列表 |
| **风格检查** | PEP8和编码规范 | Python | 风格违规点 |
| **导入检查** | import语句规范化 | Python | 导入优化建议 |

### 🎯 **7大优化策略**

1. **🔧 行长度检查** - 识别超长代码行 (< 100字符)
2. **📦 导入规范** - 优化import顺序和分组 (PEP8标准)
3. **📝 注释规范** - 补充缺失的文档注释和TODO
4. **🔧 函数长度** - 识别过长函数 (< 50行)
5. **🏷️ 变量命名** - 检查命名规范和可读性
6. **📏 空行规范** - 统一空行使用标准
7. **♻️ 重复代码** - 发现并建议合并重复片段

---

## 📁 项目结构

```
📁 agent_init/
├── 📁 src/                          # 核心源码目录
│   ├── 📁 analyzers/                # 代码分析器模块
│   ├── 📁 config/                   # 配置管理
│   │   ├── manager.py               # 配置管理器（带验证）
│   │   └── settings.py              # 配置定义
│   ├── 📁 graph/                    # LangGraph工作流
│   │   └── base.py                  # 图构建基础
│   ├── 📁 llm/                      # LLM 集成
│   │   └── enhancer.py              # 代码增强器
│   ├── 📁 nodes/                    # 工作流节点
│   │   ├── optimization.py          # 优化节点
│   │   ├── phase2.py                # Phase 2 节点
│   │   └── real.py                  # 真实节点实现
│   ├── 📁 state/                    # 状态管理
│   │   └── base.py                  # Pydantic 状态模型
│   ├── 📁 strategies/               # 优化策略
│   │   ├── optimization_strategies.py  # 7大策略实现
│   │   └── strategy_definitions.py     # 策略定义
│   ├── 📁 testing/                  # 测试验证
│   │   └── validator.py             # 代码验证器
│   ├── 📁 tools/                    # 工具集
│   │   ├── ast_parser.py            # AST解析器
│   │   ├── code_analyzer.py         # 代码分析工具
│   │   ├── file_scanner.py          # 文件扫描器
│   │   └── json_generator.py        # JSON策略生成
│   └── 📁 utils/                    # 工具函数
│       ├── file_modifier.py         # 文件修改器（带备份）
│       ├── logging_config.py        # 日志配置
│       ├── report_generator.py      # 报告生成器
│       └── utils.py                 # 通用工具
├── 📁 tests/                         # 测试目录（224个测试）
├── 📁 reports/                      # 分析报告输出
├── 📄 config.json                   # 配置文件
├── 📄 main.py                       # 主程序入口
├── 📄 requirements.txt              # 核心依赖
├── 📄 .coveragerc                   # 覆盖率配置
└── 📄 README.md                     # 项目文档

📊 代码统计:
   • 总Python文件: 48个
   • 测试文件: 22个
   • 测试用例: 224个
   • 代码覆盖率: 63%
```

---

## ⚙️ 配置说明

### 📄 **config.json 配置示例**

```json
{
  "version": "1.0.0",
  "workflow": {
    "max_iterations": 5
  },
  "analysis": {
    "max_files_to_analyze": 20,
    "max_files_to_optimize": 15
  },
  "optimization": {
    "max_line_length": 100,
    "max_function_length": 50,
    "enable_auto_fix": true,
    "strategies": {
      "enabled": [
        "line_length_optimizer",
        "import_optimizer",
        "comment_optimizer"
      ]
    }
  },
  "file_modifier": {
    "backup_dir": ".optimization_backups",
    "backup_retention_days": 7
  },
  "testing": {
    "timeout": 300
  }
}
```

配置项说明：
- `workflow.max_iterations`: 最大优化迭代次数 (1-100)
- `analysis.max_files_to_analyze`: 最大分析文件数 (1-1000)
- `optimization.max_line_length`: 最大行长度 (50-500)
- `file_modifier.backup_retention_days`: 备份保留天数 (1-365)

---

## 🏗️ 技术架构

### 🔄 **工作流程**

```
🎯 用户命令
     ↓
📁 文件扫描器 (FileScanner)
     ↓  
🧠 代码分析器 (CodeAnalyzer)
     ↓
🤖 LangGraph 工作流引擎
     ↓
🎨 优化策略引擎 (7大策略)
     ↓
✅ 测试验证 (CodeValidator)
     ↓
📊 报告生成器 (HTML报告)
```

### 🧱 **核心技术栈**

| 组件 | 版本 | 用途 |
|------|------|------|
| **LangGraph** | 0.2.55+ | 工作流引擎 |
| **LangChain** | 0.3.11+ | LLM集成 |
| **Pydantic** | 2.10.3+ | 数据验证 |
| **Pytest** | 8.3.4+ | 测试框架 |
| **Coverage** | 7.0.0+ | 覆盖率统计 |

---

## 🧪 测试指南

### ✅ **运行测试**

```bash
# 运行所有测试
python -m pytest tests/ -v

# 快速测试（无覆盖率）
python -m pytest tests/ -q

# 生成 HTML 覆盖率报告
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # 查看报告

# 只运行特定测试
python -m pytest tests/test_config_manager.py -v
```

### 📊 **覆盖率报告**

当前测试覆盖率：**63%**（实际使用模块）

```
核心模块覆盖率:
✅ src/nodes/real.py              95%
✅ src/utils/file_modifier.py     85%
✅ src/utils/report_generator.py  92%
✅ src/tools/ast_parser.py        80%
✅ src/config/manager.py          90%
✅ src/testing/validator.py       80%
```

---

## 💡 使用示例

### 🧪 **示例1: 快速测试**

```bash
$ python main.py test

🤖 LangGraph 自我优化代码助手 v0.2
============================================================
🔍 开始快速测试...
📁 当前项目: /home/user/my-project
📄 发现文件: 12个Python文件
📊 总代码行数: 2,456行

🎯 分析结果:
   ⚠️  发现5个潜在问题
   ✅  自动修复了2个问题
   📝  需要手动处理3个问题

📊 详细报告: reports/quick_analysis_20260303_112045.html
💡 建议查看报告后处理剩余问题
```

### 🚀 **示例2: 完整优化**

```bash
$ python main.py full

🤖 LangGraph 自我优化代码助手 v0.2
============================================================
🚀 启动完整自我优化流程...

📁 扫描路径: /home/user/my-project
📄 文件统计: 23个Python文件 | 6,789行代码
⏱️ 扫描耗时: 1.8秒

🔧 优化执行:
   ✅ 导入优化: 8个文件重新排序
   ✅ 格式化: 156处风格修复
   ✅ 注释补充: 12个函数添加docstring
   ⚠️  重复代码: 发现3处推荐合并
   ⚠️  超长函数: 2个函数建议拆分

📈 质量提升:
   - 代码风格: B → A- (+18%)
   - 可维护性: B+ → A (+15%)
   - 总体优化: 16.7%

🎉 优化完成! 报告: reports/full_optimization_20260303_112245.html
```

### 🧪 **示例3: 运行测试**

```bash
$ python -m pytest tests/ -v

============================= test session starts ==============================
platform linux -- Python 3.12.3
...
test/test_config_manager.py::TestConfigManager::test_singleton_pattern PASSED
test/test_file_modifier.py::TestFileModifier::test_backup_file PASSED
test/test_validator.py::TestCodeValidator::test_run_tests PASSED
...
======================== 183 passed, 1 warning in 3.0s ========================
```

---

## 🛠️ 开发指南

### 🔧 **开发环境**

```bash
# 1. 安装开发依赖
pip install -r requirements-dev.txt

# 2. 代码格式化
black src/ test/
isort src/ test/

# 3. 静态分析
ruff check src/ --fix
mypy src/

# 4. 运行测试
pytest test/ --cov=src

# 5. 检查覆盖率
pytest test/ --cov=src --cov-report=term-missing
```

### 📝 **添加新测试**

```python
# test/test_new_feature.py
import pytest
from src.new_module import NewFeature

class TestNewFeature:
    def test_basic_functionality(self):
        feature = NewFeature()
        result = feature.do_something()
        assert result is not None
```

---

## 🔍 故障排除

### ❓ **常见问题**

**Q: ❌ 模块导入错误？**
```bash
# 确保虚拟环境激活
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/  # 验证安装
```

**Q: 🧪 测试失败？**
```bash
# 检查是否在项目根目录
pwd  # 应该是 /path/to/agent_init

# 重新安装依赖
pip install -r requirements.txt
```

**Q: 🚫 内存占用过高？**
```bash
# 使用test模式
python main.py test

# 或减少分析文件数（修改 config.json）
```

**Q: 📊 覆盖率低于预期？**
```bash
# 检查 .coveragerc 配置
# 未使用模块已排除，这是正常的
```

---

## 📈 更新日志

### v0.3.0 (2026-03-04)
- ✅ 添加 224 个单元测试，覆盖率 63%
- 🔧 修复 6 处裸 except 语句
- 📝 替换 359 处 print 为 logger 统一日志
- 🗑️ 清理 14 个不属于项目的文件
- 🧹 删除 8,928 个 Python 缓存文件
- 📦 整理 requirements 文件（删除 2 个冗余）
- 🗂️ 移动项目报告到 reports/archived/

### v0.2.0 (2026-03-03)
- 🔧 重构配置管理，添加配置验证
- 🛠️ 修复代码中的问题（类名冲突等）
- 📝 添加 .coveragerc 配置

### v0.1.0 (2026-03-02)
- 🎉 初始版本发布
- 🤖 LangGraph 工作流集成
- 🎨 7大优化策略实现
- 📊 HTML 报告生成

---

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢以下优秀开源项目：

- 🚀 [LangChain](https://github.com/langchain-ai/langchain) - LLM应用框架
- 🔄 [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流引擎
- 🛠️ [Ruff](https://github.com/astral-sh/ruff) - 代码检查器
- 📦 [Pydantic](https://github.com/pydantic/pydantic) - 数据验证

---

<div align="center">

**⭐ 如果这个工具对你有帮助，请给我们一个Star！**

**🚀 一起打造更智能的代码质量工具！**

---

*最后更新: 2026-03-04 | 版本: v0.3.0 | 测试: 224 passed | 覆盖率: 63%*

</div>