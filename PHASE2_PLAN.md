# Phase 2 计划

## 目标
集成 LLM 和测试验证，实现智能代码优化

## 任务清单

### 1. LLM 集成 ✅
- [x] 配置 LLM 客户端（Kimi API）
- [x] 创建 `src/llm/enhancer.py`
- [x] 实现代码分析和建议功能
- [x] 集成到工作流

### 2. 测试验证 ✅
- [x] 创建 `src/testing/validator.py`
- [x] 集成 pytest 运行测试
- [x] 验证优化前后的代码
- [x] 失败时自动回滚

### 3. 智能重构 ⏳
- [ ] 复杂重构建议生成
- [ ] 函数拆分建议
- [ ] 代码结构优化

## 新增文件
- `src/llm/enhancer.py` - LLM 增强器
- `src/llm/__init__.py`
- `src/testing/validator.py` - 测试验证器
- `src/testing/__init__.py`
- `src/nodes/phase2.py` - Phase 2 节点

## 使用方法

### 配置 Kimi API
设置环境变量：
```bash
export KIMI_API_KEY="your_api_key"
```

### 运行 Phase 2
```bash
python main.py phase2
```

## 功能特性

### LLM 功能
- 智能代码问题分析
- 重构建议生成
- 问题解释
- 文档字符串生成

### 测试验证
- 语法验证
- pytest 集成
- 自动回滚机制

## 依赖
- openai (兼容 Kimi API)
- pytest

## 进度
- 开始时间: 2026-03-04
- 完成时间: 2026-03-04
