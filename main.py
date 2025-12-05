import os
import re
import json
import traceback

from loguru import logger
from datetime import datetime

from dotenv import load_dotenv
from typing import Optional

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from utils import CustomModel, load_config, SQLitePriceDB, CustomEmbedding

# 加载环境变量
load_dotenv()
load_config()

llm = CustomModel(model_name=os.environ.get("model_name"), api_url=os.environ.get("model_url"))
embeddings = CustomEmbedding(model_name=os.environ.get('emb'), api_url=os.environ.get('emb_url'))
# 初始化数据库连接器
db = SQLitePriceDB()


@tool("vector_db_search", return_direct=True)
def vector_db_search(query: str, k: int = 3) -> str:
    """
    在向量数据库中搜索相关的价格信息

    Args:
        query: 搜索查询
        k: 返回结果数量

    Returns:
        str: 搜索结果
    """
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # 加载向量数据库

    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # 搜索
    results = vectorstore.similarity_search(query, k=k)

    # 格式化结果
    formatted_results = "\n".join([f"- {doc.page_content}" for doc in results])
    return f"找到以下相关信息：\n{formatted_results}"


# 定义SQL工具
@tool("sql_query_tool", return_direct=False)
def sql_query_tool(natural_language_query: str) -> str:
    """
    将自然语言查询转换为SQL并执行查询价格表

    Args:
        natural_language_query: 自然语言描述的查询需求

    Returns:
        str: 格式化的查询结果
    """
    logger.debug('sql_query_tool...')
    # 使用LLM将自然语言转换为SQL
    prompt = ChatPromptTemplate.from_template("""
    你是一个SQL专家，需要将自然语言查询转换为SQLite兼容的SQL语句。

    数据库表结构：
    - 表名：price_table
    - 字段：
      * product_name (TEXT): 产品名称
      * specification (TEXT): 规格型号
      * price (REAL): 产品价格
      * description (TEXT): 产品描述

    自然语言查询：{query}

    要求：
    1. 只返回SQL语句，不要返回其他内容
    2. 使用SQLite语法
    3. 结果返回全部字段
    4. 不要使用DROP/ALTER等修改表结构的语句
    """)

    # 生成SQL
    sql_chain = prompt | llm | StrOutputParser()
    sql_query = sql_chain.invoke({"query": natural_language_query}).strip()

    # 安全检查：只允许SELECT查询
    if not sql_query.strip().upper().startswith("SELECT"):
        return "安全限制：只允许执行SELECT查询"

    # 执行SQL查询
    results = db.execute_query(sql_query)

    # 格式化结果
    if results and "error" not in results[0]:
        if len(results) == 0:
            return "未找到匹配的数据"

        formatted_results = []
        for i, row in enumerate(results, 1):
            row_str = f"{i}. 产品：{row.get('product_name', '未知')}"
            if 'specification' in row:
                row_str += f" | 规格：{row.get('specification', '标准')}"
            if 'price' in row:
                row_str += f" | 价格：{row.get('price', 0):.2f}元"
            if 'description' in row:
                row_str += f" | 描述：{row.get('description', '无')}"
            formatted_results.append(row_str)

        return "\n".join(formatted_results)
    else:
        error_msg = results[0].get("error", "未知错误") if results else "查询无结果"
        return f"查询错误：{error_msg}"


@tool("price_update_tool", return_direct=False)
def price_update_tool(product_name: str, specification: str, new_price: float) -> str:
    """
    更新产品价格

    Args:
        product_name: 产品名称
        specification: 产品规格
        new_price: 新价格

    Returns:
        str: 更新结果
    """
    logger.debug('price_update_tool...')
    query = '''
        UPDATE price_table 
        SET price = ?, updated_at = ? 
        WHERE product_name = ? AND specification = ?
    '''
    params = (new_price, datetime.now().isoformat(), product_name, specification)

    result = db.execute_update(query, params)
    return result


@tool("product_info_tool", return_direct=False)
def product_info_tool(product_name: Optional[str] = None, min_price: Optional[float] = None,
                      max_price: Optional[float] = None) -> str:
    """
    查询产品信息，支持多条件筛选

    Args:
        product_name: 产品名称（可选）
        min_price: 最低价格（可选）
        max_price: 最高价格（可选）

    Returns:
        str: 筛选结果
    """
    logger.debug('product_info_tool...')
    query_parts = ["SELECT * FROM price_table WHERE 1=1"]
    params = []

    if product_name:
        query_parts.append("AND product_name LIKE ?")
        params.append(f"%{product_name}%")

    if min_price is not None:
        query_parts.append("AND price >= ?")
        params.append(min_price)

    if max_price is not None:
        query_parts.append("AND price <= ?")
        params.append(max_price)

    query = " ".join(query_parts)
    results = db.execute_query(query, tuple(params))

    if results and "error" not in results[0]:
        if len(results) == 0:
            return "未找到符合条件的产品"

        formatted = [f"共找到{len(results)}个产品："]
        for row in results:
            formatted.append(f"- {row['product_name']} ({row['specification']}): {row['price']:.2f}元")

        return "\n".join(formatted)
    else:
        return results[0].get("error", "查询失败")


# 定义LangGraph状态
class PriceAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str
    tool_result: str


# 定义节点：决定下一步行动
def decide_action(state: PriceAgentState) -> PriceAgentState:
    logger.debug('decide_action...')
    messages = state["messages"]
    last_message = messages[-1].content

    prompt = ChatPromptTemplate.from_template("""
    根据用户的问题，决定需要执行的操作：

    如果是查询产品价格或信息，返回"query"
    如果是更新产品价格，返回"update"
    如果是复杂条件筛选，返回"filter"
    如果只是简单问答，返回"answer"

    用户问题：{input}

    只返回操作类型，不要返回其他内容。
    """)

    action_chain = prompt | llm | StrOutputParser()

    data = action_chain.invoke({"input": last_message}).strip().lower()
    state['next_action'] = data
    return state


# 定义节点：执行SQL查询
def execute_query(state: PriceAgentState) -> PriceAgentState:
    logger.debug('execute_query...')
    last_message = state["messages"][-1].content
    result = sql_query_tool.invoke({"natural_language_query": last_message})
    state["tool_result"] = result
    state["messages"].append(AIMessage(content=f"查询结果：\n{result}"))
    return state


# 定义节点：执行价格更新
def execute_update(state: PriceAgentState) -> PriceAgentState:
    logger.debug('execute_update...')
    last_message = state["messages"][-1].content

    # 解析更新参数
    prompt = ChatPromptTemplate.from_template("""
    从用户输入中提取产品名称、规格和新价格：

    用户输入：{input}

    返回JSON格式：{{"product_name": "...", "specification": "...", "new_price": ...}}
    不要返回markdown字符```json```
    """)

    parser_chain = prompt | llm | StrOutputParser()

    try:
        params = json.loads(parser_chain.invoke({"input": last_message}))
        result = price_update_tool.invoke(params)
    except:
        traceback.print_exc()
        result = "参数解析失败，请提供明确的产品名称、规格和价格"

    state["tool_result"] = result
    state["messages"].append(AIMessage(content=f"更新结果：\n{result}"))
    return state


# 定义节点：执行筛选查询
def execute_filter(state: PriceAgentState) -> PriceAgentState:
    logger.debug('execute_filter...')
    last_message = state["messages"][-1].content

    # 解析筛选条件
    prompt = ChatPromptTemplate.from_template("""
    从用户输入中提取筛选条件：

    用户输入：{input}

    返回JSON格式：{{"product_name": "...", "min_price": ..., "max_price": ...}}
    没有的条件填null
    不要返回markdown字符```json```
    """)

    parser_chain = prompt | llm | StrOutputParser()

    try:
        params = json.loads(parser_chain.invoke({"input": last_message}))
        # 清理None值
        params = {k: v for k, v in params.items() if v is not None}
        result = product_info_tool.invoke(params)
    except:
        traceback.print_exc()
        result = "参数解析失败，请提供明确的筛选条件"
    state["tool_result"] = result
    state["messages"].append(AIMessage(content=f"筛选结果：\n{result}"))
    return state


# 定义节点：生成最终回答
def generate_answer(state: PriceAgentState) -> PriceAgentState:
    logger.debug('generate_answer...')
    messages = state["messages"]
    tool_result = state.get("tool_result", "")
    user_query = messages[0].content
    last_query = messages[-1].content
    prompt = ChatPromptTemplate.from_template("""
    根据工具执行结果，用友好的自然语言回答用户的问题。

    工具结果：
    {tool_result}
    
    工具结果的来源吧：
    {last_query}

    用户问题：{user_query}

    回答要清晰、准确、有用。
    """)

    answer_chain = prompt | llm | StrOutputParser()

    answer = answer_chain.invoke({
        "tool_result": tool_result,
        "user_query": user_query,
        "last_query": last_query
    })

    state["messages"].append(AIMessage(content=answer))
    return state


# 定义路由函数
def route_action(state: PriceAgentState) -> str:
    next_action = state["next_action"]
    action_map = {
        "query": "execute_query",
        "update": "execute_update",
        "filter": "execute_filter",
        "answer": "generate_answer"
    }
    return action_map.get(next_action, "generate_answer")


# 创建LangGraph工作流
workflow = StateGraph(PriceAgentState)

# 添加节点
workflow.add_node("decide_action", decide_action)
workflow.add_node("execute_query", execute_query)
workflow.add_node("execute_update", execute_update)
workflow.add_node("execute_filter", execute_filter)
workflow.add_node("generate_answer", generate_answer)

# 设置起始节点
workflow.set_entry_point("decide_action")

# 添加条件边
workflow.add_conditional_edges(
    "decide_action",
    route_action,
    {
        "execute_query": "execute_query",
        "execute_update": "execute_update",
        "execute_filter": "execute_filter",
        "generate_answer": "generate_answer"
    }
)

# 添加普通边
workflow.add_edge("execute_query", "generate_answer")
workflow.add_edge("execute_update", "generate_answer")
workflow.add_edge("execute_filter", "generate_answer")
workflow.add_edge("generate_answer", END)

# 编译图
app = workflow.compile()


# 交互式对话函数
def chat():
    print("=== SQLite价格查询助手 ===")
    print("输入'quit'退出，输入'help'查看帮助\n")

    while True:
        user_input = input("你: ")

        if user_input.lower() == 'quit':
            print("助手: 再见！")
            break

        if user_input.lower() == 'help':
            print("""
支持的查询类型：
- 产品A的价格是多少？
- 所有价格低于200元的产品
- 更新产品A标准版的价格为129元
- 显示所有产品信息
- 产品B有哪些规格？
            """)
            continue

        # 执行查询
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)],
            "next_action": "",
            "tool_result": ""
        })

        # 获取最终回答
        final_answer = result["messages"][-1].content
        print(f"助手: {final_answer}\n")


# 批量测试函数
def batch_test():
    test_cases = [
        "产品A的价格是多少？",
        "所有价格低于200元的产品有哪些？",
        "产品名是产品B有哪些规格版本？",
        "最贵的产品是什么？",
        "更新产品A标准版的价格为109元",
        "查询价格在200到500元之间的产品"
    ]

    print("=== 批量测试 ===\n")
    for query in test_cases:
        print(f"查询: {query}")

        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "next_action": "",
            "tool_result": ""
        })

        answer = result["messages"][-1].content
        print(f"回答: {answer}\n{'-' * 50}\n")


# 主函数
if __name__ == "__main__":
    # 初始化数据库（自动创建并插入示例数据）
    print("初始化SQLite数据库...")

    # 运行交互式对话
    # chat()

    # 运行批量测试
    batch_test()
