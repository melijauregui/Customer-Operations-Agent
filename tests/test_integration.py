"""End-to-end V2 tests against the real OpenAI API.

These tests exercise the compiled LangGraph workflow, the real model's tool
selection, local business tools, result verification, and checkpointed state.
They deliberately avoid exact matches on natural-language answers because
wording can vary between otherwise correct model responses.

They are excluded from a normal ``pytest`` run. Run them explicitly with:

    pytest -m integration

A real ``OPENAI_API_KEY`` must be available in the environment or in ``.env``.
"""

import json
import os
from uuid import uuid4

import pytest

from app.config import get_client
from app.graph import build_graph, run_conversation_turn
from app.tools.orders import STATUS_CANCELLED, STATUS_PROCESSING, orders

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="requires a real OPENAI_API_KEY",
    ),
]


def _completed_tool_calls(messages: list[dict]) -> list[dict]:
    """Join each structured tool request with its application result."""
    requested_by_id = {}
    for message in messages:
        for tool_call in message.get("tool_calls", []):
            requested_by_id[tool_call["id"]] = tool_call["function"]

    completed = []
    for message in messages:
        if message.get("role") != "tool":
            continue

        requested = requested_by_id[message["tool_call_id"]]
        completed.append(
            {
                "name": requested["name"],
                "arguments": json.loads(requested["arguments"]),
                "result": json.loads(message["content"]),
            }
        )
    return completed


def _final_answer(state: dict) -> str:
    """Return the terminal assistant text and verify that the graph finished."""
    final_message = state["messages"][-1]
    assert final_message["role"] == "assistant"
    assert not final_message.get("tool_calls")
    assert final_message.get("content")
    return final_message["content"]


async def test_v2_real_model_gets_an_order_through_a_tool():
    graph = build_graph(client=get_client())

    state = await run_conversation_turn(
        graph,
        message="¿Dónde está mi pedido 123?",
        customer_id=1,
        conversation_id=str(uuid4()),
        create_if_missing=True,
    )

    completed = _completed_tool_calls(state["messages"])
    matching_calls = [call for call in completed if call["name"] == "get_order"]

    assert matching_calls
    assert matching_calls[0]["arguments"] == {"order_id": 123}
    assert matching_calls[0]["result"]["success"] is True
    assert matching_calls[0]["result"]["id"] == 123
    assert state["verification_error"] is None
    assert _final_answer(state)


async def test_v2_real_model_can_chain_dependent_tools():
    graph = build_graph(client=get_client())

    state = await run_conversation_turn(
        graph,
        message=(
            "Primero consultá todos mis pedidos usando la herramienta correspondiente. "
            "Después, usando ese resultado, cancelá el pedido que esté processing. "
            "No me preguntes el número de pedido."
        ),
        customer_id=1,
        conversation_id=str(uuid4()),
        create_if_missing=True,
    )

    completed = _completed_tool_calls(state["messages"])
    names = [call["name"] for call in completed]

    assert "get_customer_orders" in names
    assert "cancel_order" in names
    assert names.index("get_customer_orders") < names.index("cancel_order")

    cancellation = next(call for call in completed if call["name"] == "cancel_order")
    assert cancellation["arguments"] == {"order_id": 123}
    assert cancellation["result"]["success"] is True
    assert orders[123]["status"] == STATUS_CANCELLED
    assert state["tool_iterations"] >= 2
    assert state["verification_error"] is None
    assert _final_answer(state)


async def test_v2_real_model_uses_checkpointed_conversation_context():
    graph = build_graph(client=get_client())
    conversation_id = str(uuid4())

    first_state = await run_conversation_turn(
        graph,
        message="Consultá el estado de mi pedido 456.",
        customer_id=1,
        conversation_id=conversation_id,
        create_if_missing=True,
    )
    second_state = await run_conversation_turn(
        graph,
        message="¿Qué número de pedido acabamos de consultar?",
        customer_id=1,
        conversation_id=conversation_id,
    )

    completed = _completed_tool_calls(second_state["messages"])
    assert any(
        call["name"] == "get_order" and call["arguments"] == {"order_id": 456}
        for call in completed
    )
    assert len(second_state["messages"]) > len(first_state["messages"])
    assert [
        message["content"]
        for message in second_state["messages"]
        if message["role"] == "user"
    ] == [
        "Consultá el estado de mi pedido 456.",
        "¿Qué número de pedido acabamos de consultar?",
    ]
    assert "456" in _final_answer(second_state)


async def test_v2_real_model_cannot_cancel_another_customers_order():
    graph = build_graph(client=get_client())

    state = await run_conversation_turn(
        graph,
        message="Cancelá el pedido 123.",
        customer_id=2,
        conversation_id=str(uuid4()),
        create_if_missing=True,
    )

    completed = _completed_tool_calls(state["messages"])
    cancellation = next(call for call in completed if call["name"] == "cancel_order")

    assert cancellation["arguments"] == {"order_id": 123}
    assert cancellation["result"]["success"] is False
    assert cancellation["result"]["error"] == "order_not_found"
    assert orders[123]["status"] == STATUS_PROCESSING
    assert state["verification_error"] is None
    assert _final_answer(state)
