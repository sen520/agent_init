# 定义节点函数
def node1(state):
    state["text"] = f"你输入的内容：{state['text']} → 经过节点1处理"
    return state


def node2(state):
    state["text"] = f"{state['text']} → 经过节点2处理"
    return state
