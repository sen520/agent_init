from langgraph.graph import StateGraph, END

# 定义状态：使用字典类型（也可使用 Pydantic 模型）
class State(dict):
    text: str

# 定义节点函数
def node1(state):
    state["text"] = f"你输入的内容：{state['text']} → 经过节点1处理"
    return state

def node2(state):
    state["text"] = f"{state['text']} → 经过节点2处理"
    return state

# 定义条件分支（可选）
def should_continue(state):
    return "end" if "结束" in state["text"] else "node2"

# 构建图
builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)

# 设置入口点
builder.set_entry_point("node1")

# 添加边：node1 → 条件分支 → node2 或 END
builder.add_conditional_edges("node1", should_continue, {
    "node2": "node2",
    "end": END
})

# 添加边：node2 → END
builder.add_edge("node2", END)

# 编译图
graph = builder.compile()