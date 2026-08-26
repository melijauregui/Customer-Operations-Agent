"""Focused tests for LangGraph state and workflow behavior."""

from langgraph.graph import END, START, StateGraph

from app.graph import AgentState


def test_agent_state_appends_messages_instead_of_replacing_them():
    """The messages reducer preserves input history when a node adds a message."""

    def add_assistant_message(state: AgentState) -> dict:
        return {
            "messages": [{"role": "assistant", "content": "Hello."}],
            "tool_iterations": state["tool_iterations"] + 1,
        }

    builder = StateGraph(AgentState)
    builder.add_node("add_assistant_message", add_assistant_message)
    builder.add_edge(START, "add_assistant_message")
    builder.add_edge("add_assistant_message", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "Hi"}],
            "customer_id": 1,
            "tool_iterations": 0,
            "verification_error": None,
        }
    )

    assert result["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ]
    assert result["customer_id"] == 1
    assert result["tool_iterations"] == 1
    assert result["verification_error"] is None
