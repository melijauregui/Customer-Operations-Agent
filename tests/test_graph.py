"""Focused tests for LangGraph state and workflow behavior."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langgraph.graph import END, START, StateGraph

from app.config import OPENAI_MODEL
from app.graph import AgentState, call_model
from app.llm import SYSTEM_PROMPT, TOOLS


def _fake_client(message):
    """Return an OpenAI client double that yields one predefined message."""
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = SimpleNamespace()
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(return_value=response))
    )
    return client


def _initial_state() -> AgentState:
    return {
        "messages": [{"role": "user", "content": "Where is order 123?"}],
        "customer_id": 1,
        "tool_iterations": 0,
        "verification_error": None,
    }


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


async def test_call_model_returns_only_the_assistant_state_update():
    assistant_message = SimpleNamespace(content="Order 123 is processing.", tool_calls=None)
    client = _fake_client(assistant_message)
    state = _initial_state()

    update = await call_model(state, client=client)

    assert update == {
        "messages": [
            {
                "role": "assistant",
                "content": "Order 123 is processing.",
            }
        ],
    }
    # A node returns an update; it does not mutate the state it received.
    assert state["tool_iterations"] == 0
    assert len(state["messages"]) == 1

    client.chat.completions.create.assert_awaited_once_with(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Where is order 123?"},
        ],
        tools=TOOLS,
    )


async def test_call_model_preserves_structured_tool_calls():
    tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="get_order",
            arguments=json.dumps({"order_id": 123}),
        ),
    )
    assistant_message = SimpleNamespace(content=None, tool_calls=[tool_call])
    client = _fake_client(assistant_message)

    update = await call_model(_initial_state(), client=client)

    assert update["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": '{"order_id": 123}',
                    },
                }
            ],
        }
    ]
