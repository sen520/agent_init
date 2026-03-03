# 测试文件说明

## 测试结构

```
tests/
├── __init__.py                    # 测试包初始化
├── conftest.py                    # pytest配置和共享fixtures
├── README.md                      # 测试说明文档
├── test_*.py                      # 单元测试文件
├── test_*_integration.py          # 集成测试文件
└── test_*_simulation.py           # 模拟测试文件
```

## 测试文件分类

### 单元测试 (Unit Tests)
- `test_json_generator.py` - JSON策略生成器测试
- `test_strategy_definitions.py` - 策略定义测试

### 集成测试 (Integration Tests)
- `test_json_strategy_integration.py` - JSON策略集成测试
- `test_yaml_parser_integration.py` - YAML解析器集成测试

### 模拟测试 (Simulation Tests)
- `test_bare_simulation.py` - 基础功能模拟测试

## 运行测试

### 运行所有测试
```bash
pytest tests/ -v
```

### 运行特定测试文件
```bash
pytest tests/test_strategy_definitions.py -v
```

### 运行特定测试类
```bash
pytest tests/test_strategy_definitions.py::TestStrategyDefinitions -v
```

### 运行特定测试方法
```bash
pytest tests/test_strategy_definitions.py::TestStrategyDefinitions::test_security_strategy_default_config -v
```

### 查看测试覆盖率
```bash
pytest tests/ --cov=src --cov-report=html
```

## 测试规范

1. **测试命名**: 测试类以`Test`开头，测试方法以`test_`开头
2. **Fixtures**: 使用`@pytest.fixture`创建测试数据
3. **断言**: 使用标准的assert语句
4. **文档**: 每个测试都要有清晰的docstring说明
5. **独立**: 测试之间应该相互独立，不依赖执行顺序

## 贡献代码

添加新的测试时，请：

1. 确定测试类型（单元/集成/模拟）
2. 按命名规范创建测试文件
3. 添加必要的fixtures到conftest.py
4. 编写清晰的测试文档
5. 确保测试通过当代码正确时，失败当代码错误时