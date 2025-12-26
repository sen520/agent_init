from langgraph.graph import StateGraph, END
from src.state.base import State
from src.nodes.base import node1, node2


# 定义条件分支（可选）
def should_continue(state):
    return "end" if "结束" in state["text"] else "node2"


def build_graph():
    # 构建图
    workflow = StateGraph(State)
    workflow.add_node("node1", node1)
    workflow.add_node("node2", node2)
    # 添加节点
    workflow.add_node("init_conversation", node1)
    workflow.add_node("scan_codebase", node1)
    workflow.add_node("summarize_analysis", node1)
    workflow.add_node("reduce_context", node1)  # 上下文缩减节点
    workflow.add_node("handle_user_input", node1)
    # 设置入口点
    workflow.set_entry_point("node1")

    # 添加边：node1 → 条件分支 → node2 或 END
    # 添加边
    workflow.set_entry_point("init_conversation")
    workflow.add_edge("init_conversation", "scan_codebase")
    workflow.add_edge("scan_codebase", "summarize_analysis")

    # 添加边：node2 → END
    workflow.add_edge("node2", END)
    # 条件边：总结后判断下一步
    workflow.add_conditional_edges(
        "summarize_analysis",
        should_continue,
        {
            "reduce_context": "reduce_context",
            "scan_codebase": "scan_codebase"
        }
    )

    # 上下文缩减后处理用户输入
    workflow.add_edge("reduce_context", "handle_user_input")

    # 用户输入后循环或结束
    workflow.add_conditional_edges(
        "handle_user_input",
        lambda s: "reduce_context" if s.get("user_input") else END,
        {
            "reduce_context": "reduce_context",
            END: END
        }
    )
    # 编译图
    return workflow.compile()
