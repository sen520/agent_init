#!/usr/bin/env python3
"""
LLM 增强器 - 使用 Kimi API 进行智能代码分析和建议
"""
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI
from src.config.manager import get_config


class LLMEnhancer:
    """LLM 代码增强器"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: API 密钥（可选，从环境变量读取）
            base_url: API 基础 URL（可选）
        """
        config = get_config()
        llm_config = config.get_llm_config()
        
        # 使用传入的参数或配置
        self.api_key = api_key or os.getenv('KIMI_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('KIMI_BASE_URL', 'https://api.moonshot.cn/v1')
        self.model = llm_config.get('model', 'kimi-k2-5')
        self.max_tokens = llm_config.get('max_tokens', 1000)
        self.temperature = llm_config.get('temperature', 0.3)
        self.enabled = llm_config.get('enabled', False)
        
        # 初始化客户端
        self.client = None
        if self.api_key and self.enabled:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except Exception as e:
                print(f"⚠️  LLM 客户端初始化失败: {e}")
    
    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        return self.client is not None and self.enabled
    
    def analyze_code_issues(self, code: str, issues: List[Dict]) -> str:
        """
        分析代码问题并提供改进建议
        
        Args:
            code: 代码内容
            issues: 发现的问题列表
            
        Returns:
            LLM 的分析和建议
        """
        if not self.is_available():
            return "LLM 不可用，无法提供智能建议"
        
        # 构建提示
        issues_text = "\n".join([
            f"- [{i.get('severity', 'medium')}] {i.get('type', 'unknown')}: {i.get('description', '')}"
            for i in issues[:10]  # 最多10个问题
        ])
        
        prompt = f"""请分析以下 Python 代码，针对发现的问题提供具体的改进建议：

## 代码
```python
{code[:2000]}  # 限制代码长度
```

## 发现的问题
{issues_text}

## 任务
1. 评估这些问题的严重程度
2. 提供具体的修复建议
3. 如果有复杂问题，提供重构思路

请用中文回答，简洁明了。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 Python 代码审查专家，擅长代码优化和重构。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM 分析失败: {e}"
    
    def suggest_refactoring(self, code: str, target: str = "readability") -> str:
        """
        建议代码重构方案
        
        Args:
            code: 代码内容
            target: 重构目标（readability/performance/maintainability）
            
        Returns:
            重构建议
        """
        if not self.is_available():
            return "LLM 不可用"
        
        targets = {
            "readability": "可读性",
            "performance": "性能",
            "maintainability": "可维护性"
        }
        target_cn = targets.get(target, "可读性")
        
        prompt = f"""请针对以下 Python 代码，提供提升{target_cn}的重构建议：

```python
{code[:1500]}
```

请提供：
1. 具体的重构方案
2. 重构后的代码示例
3. 预期效果

用中文回答。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个资深的 Python 架构师，擅长代码重构和设计模式。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"重构建议获取失败: {e}"
    
    def explain_issue(self, issue_type: str, description: str) -> str:
        """
        解释代码问题
        
        Args:
            issue_type: 问题类型
            description: 问题描述
            
        Returns:
            问题解释和修复方法
        """
        if not self.is_available():
            return "LLM 不可用"
        
        prompt = f"""请解释以下 Python 代码问题，并提供修复方法：

问题类型: {issue_type}
问题描述: {description}

请提供：
1. 问题原因解释
2. 为什么这是个问题
3. 如何修复
4. 最佳实践示例

用中文回答，简洁明了。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个 Python 代码质量专家。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"解释获取失败: {e}"
    
    def generate_docstring(self, code: str) -> str:
        """
        为函数生成文档字符串
        
        Args:
            code: 函数代码
            
        Returns:
            生成的文档字符串
        """
        if not self.is_available():
            return "LLM 不可用"
        
        prompt = f"""请为以下 Python 函数生成符合 Google Style 的文档字符串：

```python
{code}
```

要求：
1. 简要描述函数功能
2. 说明参数（如果有）
3. 说明返回值（如果有）
4. 简要说明可能抛出的异常（如果有）

只返回文档字符串，不要其他解释。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个 Python 文档专家。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"文档生成失败: {e}"


# 便捷函数
def get_llm_enhancer() -> LLMEnhancer:
    """获取 LLM 增强器实例"""
    return LLMEnhancer()


def analyze_with_llm(code: str, issues: List[Dict]) -> str:
    """使用 LLM 分析代码"""
    enhancer = get_llm_enhancer()
    return enhancer.analyze_code_issues(code, issues)


if __name__ == "__main__":
    # 测试 LLM 功能
    print("🧠 LLM 增强器测试")
    print("=" * 50)
    
    enhancer = LLMEnhancer()
    
    if enhancer.is_available():
        print("✅ LLM 客户端已初始化")
        print(f"   模型: {enhancer.model}")
        print(f"   API URL: {enhancer.base_url}")
        
        # 测试解释功能
        result = enhancer.explain_issue(
            "long_line",
            "代码行长度超过 100 个字符"
        )
        print("\n测试输出:")
        print(result[:500])
    else:
        print("⚠️  LLM 未启用或配置缺失")
        print("   请设置 KIMI_API_KEY 环境变量")
        print("   并在 config.json 中设置 llm.enabled: true")
