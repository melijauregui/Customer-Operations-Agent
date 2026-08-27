"""Focused tests for LangGraph state and workflow behavior."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langgraph.graph import END, START, StateGraph

from app.config import OPENAI_MODEL
from app.graph import (
    SAFE_VERIFICATION_FAILURE,
    AgentState,
    after_verification,
    build_graph,
    call_model,
    conversation_config,
    execute_tools,
    run_conversation_turn,
    should_continue,
    verify_results,
)
from app.llm import SYSTEM_PROMPT, TOOLS
from app.tools.orders import orders


def _fake_client(*messages):
    """Return an OpenAI client double that yields predefined messages in order."""
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=message)])
        for message in messages
    ]
    client = SimpleNamespace()
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(side_effect=responses))
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


def test_should_continue_routes_tool_calls_to_tool_execution():
    state = _initial_state()
    state["messages"].append(
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
    )

    assert should_continue(state) == "execute_tools"


def test_should_continue_routes_final_text_to_end():
    state = _initial_state()
    state["messages"].append(
        {
            "role": "assistant",
            "content": "Order 123 is processing.",
        }
    )

    assert should_continue(state) == END


def test_should_continue_rejects_an_unexpected_last_message_role():
    state = _initial_state()

    try:
        should_continue(state)
    except ValueError as exc:
        assert str(exc) == "should_continue requires an assistant message"
    else:
        raise AssertionError("Routing should fail when call_model did not produce the last message")


def test_execute_tools_returns_the_real_tool_result_as_a_state_update():
    state = _initial_state()
    state["messages"].append(
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
    )

    update = execute_tools(state)

    assert update["tool_iterations"] == 1
    assert len(update["messages"]) == 1
    assert update["messages"][0]["role"] == "tool"
    assert update["messages"][0]["tool_call_id"] == "call_123"
    assert json.loads(update["messages"][0]["content"]) == {
        "success": True,
        "id": 123,
        "customer_id": 1,
        "status": "processing",
        "delivery_date": "2026-08-18",
    }
    # LangGraph applies this update later; the node does not append it itself.
    assert len(state["messages"]) == 2
    assert state["tool_iterations"] == 0


def test_execute_tools_handles_several_calls_from_one_assistant_message():
    state = _initial_state()
    state["messages"].append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "cancel_123",
                    "type": "function",
                    "function": {
                        "name": "cancel_order",
                        "arguments": '{"order_id": 123}',
                    },
                },
                {
                    "id": "get_456",
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": '{"order_id": 456}',
                    },
                },
            ],
        }
    )

    update = execute_tools(state)
    results = [json.loads(message["content"]) for message in update["messages"]]

    assert [message["tool_call_id"] for message in update["messages"]] == [
        "cancel_123",
        "get_456",
    ]
    assert results[0] == {"success": True, "order_id": 123, "status": "cancelled"}
    assert results[1]["id"] == 456
    assert orders[123]["status"] == "cancelled"
    # One model response is one tool round, regardless of the number of calls.
    assert update["tool_iterations"] == 1


def test_execute_tools_injects_customer_id_from_graph_state():
    state = _initial_state()
    state["messages"].append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "foreign_order",
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": '{"order_id": 789}',
                    },
                }
            ],
        }
    )

    update = execute_tools(state)
    result = json.loads(update["messages"][0]["content"])

    # Order 789 exists but belongs to customer 2. The state belongs to customer
    # 1, so the business tool must behave exactly as if that order did not exist.
    assert result["success"] is False
    assert result["error"] == "order_not_found"


def test_execute_tools_returns_invalid_arguments_as_a_structured_result():
    state = _initial_state()
    state["messages"].append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "invalid_order_id",
                    "type": "function",
                    "function": {
                        "name": "cancel_order",
                        "arguments": '{"order_id": "not-an-integer"}',
                    },
                }
            ],
        }
    )

    update = execute_tools(state)
    result = json.loads(update["messages"][0]["content"])

    assert result["success"] is False
    assert result["error"] == "invalid_arguments"
    assert orders[123]["status"] == "processing"


def test_verify_results_accepts_a_structured_business_failure():
    state = _initial_state()
    state["messages"].extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "cancel_456",
                        "type": "function",
                        "function": {
                            "name": "cancel_order",
                            "arguments": '{"order_id": 456}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "cancel_456",
                "content": json.dumps(
                    {
                        "success": False,
                        "error": "order_not_cancellable",
                    }
                ),
            },
        ]
    )

    update = verify_results(state)

    assert update == {"verification_error": None}
    state.update(update)
    assert after_verification(state) == "call_model"


def test_verify_results_stops_when_a_tool_result_is_missing():
    state = _initial_state()
    state["messages"].append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "get_123",
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": '{"order_id": 123}',
                    },
                }
            ],
        }
    )

    update = verify_results(state)

    assert "tool result ids do not match" in update["verification_error"]
    assert update["messages"] == [
        {
            "role": "assistant",
            "content": SAFE_VERIFICATION_FAILURE,
        }
    ]
    state["verification_error"] = update["verification_error"]
    assert after_verification(state) == END


def test_verify_results_stops_when_tool_result_json_is_invalid():
    state = _initial_state()
    state["messages"].extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "get_123",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id": 123}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "get_123",
                "content": "not-json",
            },
        ]
    )

    update = verify_results(state)

    assert "invalid JSON result" in update["verification_error"]
    assert update["messages"][0]["content"] == SAFE_VERIFICATION_FAILURE


def test_verify_results_stops_when_success_is_not_boolean():
    state = _initial_state()
    state["messages"].extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "get_123",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id": 123}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "get_123",
                "content": json.dumps({"success": "yes", "id": 123}),
            },
        ]
    )

    update = verify_results(state)

    assert "invalid result contract" in update["verification_error"]
    assert update["messages"][0]["content"] == SAFE_VERIFICATION_FAILURE


async def test_compiled_graph_ends_when_model_returns_final_text():
    client = _fake_client(
        SimpleNamespace(content="Hello!", tool_calls=None),
    )
    graph = build_graph(client=client)

    result = await graph.ainvoke(
        _initial_state(),
        config=conversation_config("direct-response"),
    )

    assert result["messages"] == [
        {"role": "user", "content": "Where is order 123?"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert result["tool_iterations"] == 0
    assert client.chat.completions.create.await_count == 1


async def test_compiled_graph_executes_a_tool_and_returns_to_the_model():
    get_order_call = SimpleNamespace(
        id="get_123",
        function=SimpleNamespace(
            name="get_order",
            arguments='{"order_id": 123}',
        ),
    )
    client = _fake_client(
        SimpleNamespace(content=None, tool_calls=[get_order_call]),
        SimpleNamespace(content="Order 123 is processing.", tool_calls=None),
    )
    graph = build_graph(client=client)

    result = await graph.ainvoke(
        _initial_state(),
        config=conversation_config("one-tool-round"),
    )

    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_result = json.loads(result["messages"][2]["content"])
    assert tool_result["success"] is True
    assert tool_result["id"] == 123
    assert result["messages"][-1]["content"] == "Order 123 is processing."
    assert result["tool_iterations"] == 1
    assert client.chat.completions.create.await_count == 2


async def test_compiled_graph_can_chain_dependent_tool_rounds():
    list_orders_call = SimpleNamespace(
        id="list_orders",
        function=SimpleNamespace(name="get_customer_orders", arguments="{}"),
    )
    cancel_order_call = SimpleNamespace(
        id="cancel_123",
        function=SimpleNamespace(
            name="cancel_order",
            arguments='{"order_id": 123}',
        ),
    )
    client = _fake_client(
        SimpleNamespace(content=None, tool_calls=[list_orders_call]),
        SimpleNamespace(content=None, tool_calls=[cancel_order_call]),
        SimpleNamespace(content="Order 123 was cancelled.", tool_calls=None),
    )
    graph = build_graph(client=client)

    result = await graph.ainvoke(
        _initial_state(),
        config=conversation_config("dependent-tool-rounds"),
    )

    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result["tool_iterations"] == 2
    assert orders[123]["status"] == "cancelled"
    assert result["messages"][-1]["content"] == "Order 123 was cancelled."
    assert client.chat.completions.create.await_count == 3


async def test_same_conversation_id_restores_checkpointed_history():
    client = _fake_client(
        SimpleNamespace(content="Which order?", tool_calls=None),
        SimpleNamespace(content="Order 123 is processing.", tool_calls=None),
    )
    graph = build_graph(client=client)

    first_result = await run_conversation_turn(
        graph,
        message="Where is my order?",
        customer_id=1,
        conversation_id="conversation-a",
        create_if_missing=True,
    )
    second_result = await run_conversation_turn(
        graph,
        message="The 123 one",
        customer_id=1,
        conversation_id="conversation-a",
    )

    assert first_result["messages"] == [
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "Which order?"},
    ]
    assert second_result["messages"] == [
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "Which order?"},
        {"role": "user", "content": "The 123 one"},
        {"role": "assistant", "content": "Order 123 is processing."},
    ]

    second_llm_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
    assert {"role": "user", "content": "Where is my order?"} in second_llm_messages
    assert {"role": "assistant", "content": "Which order?"} in second_llm_messages
    assert {"role": "user", "content": "The 123 one"} in second_llm_messages

    snapshot = await graph.aget_state(conversation_config("conversation-a"))
    assert snapshot.values["messages"] == second_result["messages"]
    assert snapshot.values["customer_id"] == 1


async def test_different_conversation_id_starts_with_empty_history():
    client = _fake_client(
        SimpleNamespace(content="First response", tool_calls=None),
        SimpleNamespace(content="Second response", tool_calls=None),
    )
    graph = build_graph(client=client)

    await run_conversation_turn(
        graph,
        message="Message for conversation A",
        customer_id=1,
        conversation_id="conversation-a",
        create_if_missing=True,
    )
    result_b = await run_conversation_turn(
        graph,
        message="Message for conversation B",
        customer_id=1,
        conversation_id="conversation-b",
        create_if_missing=True,
    )

    assert result_b["messages"] == [
        {"role": "user", "content": "Message for conversation B"},
        {"role": "assistant", "content": "Second response"},
    ]


async def test_customer_cannot_reuse_another_customers_checkpoint():
    client = _fake_client(
        SimpleNamespace(content="Customer one's response", tool_calls=None),
    )
    graph = build_graph(client=client)

    await run_conversation_turn(
        graph,
        message="Customer one's message",
        customer_id=1,
        conversation_id="private-conversation",
        create_if_missing=True,
    )

    try:
        await run_conversation_turn(
            graph,
            message="Customer two trying to continue",
            customer_id=2,
            conversation_id="private-conversation",
        )
    except ValueError as exc:
        assert str(exc) == "conversation_not_found"
    else:
        raise AssertionError("A customer must not access another customer's checkpoint")

    # Ownership is rejected before another model call can spend tokens or act.
    assert client.chat.completions.create.await_count == 1


async def test_unknown_conversation_id_is_not_created_implicitly():
    client = _fake_client()
    graph = build_graph(client=client)

    try:
        await run_conversation_turn(
            graph,
            message="Continue an unknown conversation",
            customer_id=1,
            conversation_id="unknown-conversation",
        )
    except ValueError as exc:
        assert str(exc) == "conversation_not_found"
    else:
        raise AssertionError("An unknown supplied conversation id must not create a thread")

    assert client.chat.completions.create.await_count == 0
