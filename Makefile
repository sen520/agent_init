.PHONY: help install test format lint type-check security clean build upload docs install-dev

# 默认目标
help:
	@echo "可用命令:"
	@echo "  install      安装项目依赖"
	@echo "  install-dev  安装开发依赖"
	@echo "  test         运行测试套件"
	@echo "  test-cov     运行测试并生成覆盖率报告"
	@echo "  format       格式化代码 (black + isort)"
	@echo "  lint         代码质量检查 (ruff + flake8)"
	@echo "  type-check   类型检查 (mypy)"
	@echo "  security     安全检查 (bandit + safety)"
	@echo "  check        运行所有检查 (格式、代码质量、类型、安全)"
	@echo "  clean        清理临时文件和缓存"
	@echo "  build        构建包"
	@echo "  upload       上传到 PyPI"
	@echo "  docs         生成文档"
	@echo "  run-help     运行项目帮助命令"

# 安装依赖
install:
	./venv/bin/pip install -r requirements.txt

install-dev:
	./venv/bin/pip install -r requirements-dev.txt
	./venv/bin/pre-commit install

# 测试相关
test:
	./venv/bin/python -m pytest tests/ -v

test-cov:
	./venv/bin/python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	./venv/bin/python -m pytest tests/ -v -m "not slow"

# 代码格式化
format:
	./venv/bin/python -m black src/ tests/
	./venv/bin/python -m isort src/ tests/

format-check:
	./venv/bin/python -m black --check src/ tests/
	./venv/bin/python -m isort --check-only src/ tests/

# 代码质量检查
lint:
	./venv/bin/python -m ruff check src/ tests/
	./venv/bin/python -m flake8 src/ tests/

lint-fix:
	./venv/bin/python -m ruff check src/ tests/ --fix

# 类型检查
type-check:
	./venv/bin/python -m mypy src/

# 安全检查
security:
	./venv/bin/python -m bandit -r src/
	./venv/bin/python -m safety check

# 完整的代码检查
check: format-check lint type-check security
	@echo "所有检查完成！"

check-and-fix: format lint-fix type-check
	@echo "代码检查和修复完成！"

# 项目运行
run-help:
	./venv/bin/python main.py help

run-test:
	./venv/bin/python main.py test

# 清理
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf src/__pycache__ src/**/__pycache__
	rm -rf tests/__pycache__ tests/**/__pycache__

# 构建和发布
build: clean
	./venv/bin/python -m build

upload: build
	./venv/bin/python -m twine upload dist/*

upload-test: build
	./venv/bin/python -m twine upload --repository testpypi dist/*

# 文档
docs:
	./venv/bin/python -m mkdocs build
	./venv/bin/python -m mkdocs serve

# 性能分析
profile:
	./venv/bin/python -m memory_profiler main.py test
	./venv/bin/python -m line_profiler -k main.py

# 依赖分析
deps-analysis:
	./venv/bin/pipdeptree
	./venv/bin/pip-audit

# 开发环境设置
setup-dev: install-dev
	./venv/bin/pre-commit install
	@echo "开发环境设置完成！"

# 快速检查 (适用于 CI/CD)
ci-check: format-check lint type-check security test
	@echo "CI/CD 检查完成！"