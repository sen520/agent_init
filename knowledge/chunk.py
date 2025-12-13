import re
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


## 1. 基于语义分割
def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def _split_sentences(text: str) -> list[str]:
    """中文分句（优先用HanLP，备用正则）"""
    try:
        from pyhanlp import HanLP
        return HanLP.splitSentence(text)
    except ImportError:
        sentences = re.split(r'(。|！|？|；|\n)', text)
        return [s + sep for s, sep in zip(sentences[::2], sentences[1::2])] if len(sentences) > 1 else sentences


def split_by_semantic(
        text: str,
        model,
        tokenizer,
        chunk_size: int = 512,
        similarity_threshold: float = 0.7
) -> list[str]:
    """  
    基于语义相似度分割（bge-small-zh-v1.5生成嵌入，余弦相似度判断分割点）  

    Args:        text: 待分割文本  
        model: SentenceTransformer实例（bge-small-zh-v1.5）  
        tokenizer: Tokenizer实例  
        chunk_size: 每个chunk的最大Token数  
        similarity_threshold: 语义相似度阈值（低于则分割）  

    Returns:        分割后的chunk列表  
    """  # 分句
    sentences = _split_sentences(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

        # 生成句子嵌入
    embeddings = model.encode(sentences)

    # 寻找语义分割点（相邻句子相似度低于阈值）
    split_points = []
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < similarity_threshold:
            split_points.append(i)

            # 合并句子为Chunk（确保Token数不超限）
    chunks = []
    start_idx = 0
    for point in split_points:
        chunk_candidate = ''.join(sentences[start_idx:point])
        # 若Chunk过大，进一步拆分
        while len(tokenizer.encode(chunk_candidate)) > chunk_size:
            mid = (start_idx + point) // 2
            split_points.insert(0, mid)
            split_points.sort()
            point = mid
        chunks.append(chunk_candidate)
        start_idx = point

        # 处理最后一个Chunk
    final_chunk = ''.join(sentences[start_idx:])
    if final_chunk:
        chunks.append(final_chunk)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


## 2. 基于token_size分割
def split_by_tokens(
        text: str,
        tokenizer,
        chunk_size: int = 512,
        chunk_overlap: int = 50
) -> list[str]:
    """  
    按Token数量分割文本，控制每个chunk的Token数不超过阈值  

    Args:        text: 待分割文本  
        tokenizer: Tokenizer实例（如bge-small-zh-v1.5的tokenizer）  
        chunk_size: 每个chunk的最大Token数  
        chunk_overlap: 相邻chunk的重叠Token数  

    Returns:        分割后的chunk列表  
    """  # 编码文本为Token IDs
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        # 计算当前chunk的结束位置
        end_idx = start_idx + chunk_size
        chunk_tokens = tokens[start_idx:end_idx]
        # 解码Token为文本
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        # 移动起始位置（保留重叠）
        start_idx += chunk_size - chunk_overlap

    return [chunk.strip() for chunk in chunks if chunk.strip()]


## 3.基于markdown格式分割
def _split_markdown_element(
        element: str,
        tokenizer,
        chunk_size: int = 512
) -> list[str]:
    """递归拆分单个Markdown元素（段落/标题/代码块等）"""
    token_count = len(tokenizer.encode(element))
    if token_count <= chunk_size:
        return [element]

        # 按中文句子拆分（备用方案）
    sentences = re.split(r'(。|！|？|；|\n)', element)
    sentences = [s + sep for s, sep in zip(sentences[::2], sentences[1::2])] if len(sentences) > 1 else sentences

    chunks = []
    current_chunk = ""
    for sent in sentences:
        current_token_count = len(tokenizer.encode(current_chunk + sent))
        if current_token_count <= chunk_size:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_by_markdown_recursive(
        text: str,
        tokenizer,
        chunk_size: int = 512
) -> list[str]:
    """
    按Markdown结构递归分割（标题/代码块/列表/段落）

    Args:        text: 待分割的Markdown文本
        tokenizer: Tokenizer实例
        chunk_size: 每个chunk的最大Token数

    Returns:        分割后的chunk列表
    """
    chunks = []

    # 第一步：分割代码块（```...```）
    code_blocks = re.split(r'(```[\s\S]*?```)', text)
    for block in code_blocks:
        if block.startswith('```'):
            # 处理代码块
            chunks.extend(_split_markdown_element(block, tokenizer, chunk_size))
        else:
            # 第二步：分割标题（#/##/###...）
            headings = re.split(r'(#{1,6}\s.*?\n)', block)
            for heading in headings:
                if heading.startswith('#'):
                    # 处理标题
                    chunks.extend(_split_markdown_element(heading, tokenizer, chunk_size))
                else:
                    # 第三步：分割列表项（* / - / 数字.）
                    list_items = re.split(r'(\n\* |\n- |\n\d+\. )', heading)
                    for item in list_items:
                        _cosine_similarity
                        if item.strip() and (
                                item.startswith('* ') or item.startswith('- ') or re.match(r'\d+\. ', item)):
                            # 处理列表项
                            chunks.extend(_split_markdown_element(item, tokenizer, chunk_size))
                        else:
                            # 第四步：处理普通段落
                            chunks.extend(_split_markdown_element(item, tokenizer, chunk_size))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


if __name__ == '__main__':
    model_name = "../models/bge-small-zh-v1.5"
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 测试文本（Markdown格式）
    test_text = r"""  
# 知识库分割方法介绍  

知识库分割是将长文本拆分成小块的过程，便于后续的检索和处理。主要有以下几种方法：  

## 1. 基于Token Size的分割  
这种方法根据文本的Token数量来分割，比如每个Chunk不超过512个Token，保留一定的重叠。例如，使用Transformers的Tokenizer计算Token数，然后拆分。  

## 2. 基于Markdown递归分割  
这种方法解析Markdown的结构，按标题、列表、代码块等元素递归拆分。比如：  
```python  
def split_markdown(text):  
    # 分割代码块  
    code_blocks = re.split(r'(```[\s\S]*?```)', text)    return code_blocks    3. 基于语义的分割  
    这种方法利用语义模型（如 bge-small-zh-v1.5）生成句子嵌入，计算相邻句子的余弦相似度，在相似度低的地方分割。例如，句子 A 和句子 B 的相似度低于 0.7，就分割。  
    语义分割的优势是保持语义的连贯性，缺点是计算成本较高。Token 分割速度快，适合对语义连贯性要求不高的场景。Markdown 分割适合结构化的文档，能保留文档的结构信息。"""

    # 测试 token分割
    token_chunks = split_by_tokens(test_text, tokenizer, chunk_size=128, chunk_overlap=10)
    print("=== 1. 基于 Token Size 的分割结果 ===")
    for i, chunk in enumerate(token_chunks, 1): print(f"{i}. {chunk}")
    print("-" * 60)

    # 测试markdown分割
    md_chunks = split_by_markdown_recursive(test_text, tokenizer, chunk_size=128)
    print("\n=== 2. 基于 Markdown 递归的分割结果 ===")
    for i, chunk in enumerate(md_chunks, 1): print(f"{i}. {chunk}")
    print("-" * 60)

    # 测试基于语义分割
    semantic_chunks = split_by_semantic(test_text, model, tokenizer, chunk_size=128, similarity_threshold=0.7)
    print("\n=== 3. 基于语义的分割结果 ===")
    for i, chunk in enumerate(semantic_chunks, 1): print(f"{i}. {chunk}")
    print("-" * 60)
