#!/usr/bin/env python3
"""
LLM 增强器 - 使用 Kimi API 进行智能代码分析和建议
"""
import os
from typing import Dict, List, Any, Optional
import logging
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from src.config.manager import get_config

# 获取 logger
logger = logging.getLogger(__name__)


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
                logger.info(f"LLM 客户端初始化成功: {self.model}")
            except ValueError as e:
                logger.error(f"LLM 客户端参数错误: {e}")
            except Exception as e:
                logger.error(f"LLM 客户端初始化失败: {type(e).__name__}: {e}")
    
    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        return self.client is not None and self.enabled
    
    def _handle_api_error(self, e: Exception, operation: str) -> str:
        """
        处理 API 错误，返回用户友好的错误信息
        
        Args:
            e: 异常对象
            operation: 操作名称
            
        Returns:
            错误信息字符串
        """
        if isinstance(e, AuthenticationError):
            logger.error(f"{operation} 失败: API 认证失败，请检查 KIMI_API_KEY")
            return f"❌ {operation} 失败: API Key 无效或已过期"
        elif isinstance(e, RateLimitError):
            logger.warning(f"{operation} 失败: 请求频率超限")
            return f"⚠️  {operation} 失败: 请求太频繁，请稍后再试"
        elif isinstance(e, APIError):
            logger.error(f"{operation} 失败: API 错误 - {e}")
            status_code = getattr(e, 'status_code', 'unknown')
            return f"❌ {operation} 失败: API 服务错误 ({status_code})"
        elif isinstance(e, TimeoutError):
            logger.warning(f"{operation} 失败: 请求超时")
            return f"⚠️  {operation} 失败: 请求超时，请重试"
        elif isinstance(e, ConnectionError):
            logger.error(f"{operation} 失败: 网络连接错误")
            return f"❌ {operation} 失败: 网络连接失败，请检查网络"
        else:
            logger.error(f"{operation} 失败: {type(e).__name__}: {e}")
            return f"❌ {operation} 失败: {str(e)[:100]}"
    
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
            logger.warning("LLM 不可用，跳过智能分析")
            return "⚠️  LLM 不可用，无法提供智能建议。请配置 KIMI_API_KEY 环境变量。"
        
        if not code or not code.strip():
            logger.warning("代码内容为空，跳过分析")
            return "⚠️  代码内容为空，无法分析"
        
        if not issues:
            logger.info("没有问题需要分析")
            return "✅ 代码没有发现明显问题"
        
        # 构建提示
        prompt = self._build_analysis_prompt(code, issues)
        
        return self._call_llm_for_analysis(prompt, len(issues))
    
    def _build_analysis_prompt(self, code: str, issues: List[Dict]) -> str:
        """构建代码分析提示"""
        issues_text = "\n".join([
            f"- [{i.get('severity', 'medium')}] {i.get('type', 'unknown')}: {i.get('description', '')}"
            for i in issues[:10]  # 最多10个问题
        ])
        
        return f"""请分析以下 Python 代码，针对发现的问题提供具体的改进建议：

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
    
    def _call_llm_for_analysis(self, prompt: str, issue_count: int) -> str:
        """调用 LLM 进行分析"""
        try:
            logger.info(f"发送 LLM 请求分析 {issue_count} 个问题")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 Python 代码审查专家，擅长代码优化和重构。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30  # 30秒超时
            )
            logger.info("LLM 分析完成")
            return response.choices[0].message.content
        except (APIError, AuthenticationError, RateLimitError, TimeoutError, ConnectionError) as e:
            return self._handle_api_error(e, "代码分析")
        except Exception as e:
            logger.exception(f"代码分析时发生未知错误")
            return f"❌ 代码分析失败: {str(e)[:100]}"
    
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
            logger.warning("LLM 不可用，无法提供重构建议")
            return "⚠️  LLM 不可用"
        
        if not code or not code.strip():
            logger.warning("代码内容为空，跳过重构建议")
            return "⚠️  代码内容为空"
        
        valid_targets = ["readability", "performance", "maintainability"]
        if target not in valid_targets:
            logger.warning(f"无效的重构目标: {target}，使用默认值 readability")
            target = "readability"
        
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
            logger.info(f"请求重构建议，目标: {target}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个资深的 Python 架构师，擅长代码重构和设计模式。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30
            )
            logger.info("重构建议获取完成")
            return response.choices[0].message.content
        except (APIError, AuthenticationError, RateLimitError, TimeoutError, ConnectionError) as e:
            return self._handle_api_error(e, "重构建议")
        except Exception as e:
            logger.exception(f"获取重构建议时发生未知错误")
            return f"❌ 重构建议获取失败: {str(e)[:100]}"
    
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
            logger.warning("LLM 不可用，无法解释问题")
            return "⚠️  LLM 不可用"
        
        if not issue_type or not description:
            logger.warning(f"问题信息不完整: type={issue_type}, desc={description}")
            return "⚠️  问题信息不完整"
        
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
            logger.info(f"请求解释问题: {issue_type}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个 Python 代码质量专家。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3,
                timeout=30
            )
            logger.info("问题解释获取完成")
            return response.choices[0].message.content
        except (APIError, AuthenticationError, RateLimitError, TimeoutError, ConnectionError) as e:
            return self._handle_api_error(e, "问题解释")
        except Exception as e:
            logger.exception(f"获取问题解释时发生未知错误")
            return f"❌ 解释获取失败: {str(e)[:100]}"
    
    def generate_docstring(self, code: str) -> str:
        """
        为函数生成文档字符串
        
        Args:
            code: 函数代码
            
        Returns:
            生成的文档字符串
        """
        if not self.is_available():
            logger.warning("LLM 不可用，无法生成文档")
            return "⚠️  LLM 不可用"
        
        if not code or not code.strip():
            logger.warning("代码内容为空，跳过文档生成")
            return "⚠️  代码内容为空"
        
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
            logger.info("请求生成文档字符串")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个 Python 文档专家。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3,
                timeout=30
            )
            logger.info("文档字符串生成完成")
            return response.choices[0].message.content.strip()
        except (APIError, AuthenticationError, RateLimitError, TimeoutError, ConnectionError) as e:
            return self._handle_api_error(e, "文档生成")
        except Exception as e:
            logger.exception(f"生成文档时发生未知错误")
            return f"❌ 文档生成失败: {str(e)[:100]}"


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
    logging.basicConfig(level=logging.INFO)
    logger.info("🧠 LLM 增强器测试")
    logger.info("=" * 50)
    
    enhancer = LLMEnhancer()
    
    if enhancer.is_available():
        logger.info("✅ LLM 客户端已初始化")
        logger.info(f"   模型: {enhancer.model}")
        logger.info(f"   API URL: {enhancer.base_url}")
        
        # 测试解释功能
        result = enhancer.explain_issue(
            "long_line",
            "代码行长度超过 100 个字符"
        )
        logger.info("测试输出:")
        logger.info(result[:500])
    else:
        logger.warning("⚠️  LLM 未启用或配置缺失")
        logger.warning("   请设置 KIMI_API_KEY 环境变量")
        logger.warning("   并在 config.json 中设置 llm.enabled: true")
