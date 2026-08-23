"""Tests for the tool-calling loop in app/agent.py, with the OpenAI client mocked.

They don't hit the network or spend tokens: we simulate the LLM's response (which
tool it "decided" to call) and verify that the application actually executed the
real tool, that the real (not invented) result made it back to the LLM, and that
`orders` state only changed when it was supposed to.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agent import MAX_AGENT_ITERATIONS, conversations, handle_message
from app.tools.orders import orders


def _tool_call_response(name: str, arguments: dict, call_id: str = "call_1"):
    """Builds a fake response mimicking the real shape of a ChatCompletion when
    the model decides to call a tool (instead of answering with text)."""
    # `arguments` travels as a JSON string, just like in the real API response
    # (agent.py does json.loads on this, so it has to be a string).
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    # content=None because when the model requests a tool it doesn't return text
    # yet; tool_calls is a list because the API supports requesting more than one
    # tool at once.
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    # agent.py only reads response.choices[0].message, so simulating that is enough.
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _multiple_tool_calls_response(*calls):
    """Build one model response containing several tool calls in the same round."""
    tool_calls = [
        SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )
        for call_id, name, arguments in calls
    ]
    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _text_response(text: str):
    """Builds the response for when the model does NOT request any tool and
    answers directly (e.g. the loop's 2nd call, once the tool result is in
    the history)."""
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_client(*responses):
    """Stand-in for AsyncOpenAI: each call to create() returns, in order, one of
    the given `responses` (side_effect). This lets us control exactly what the
    LLM "decides" on the loop's 1st and 2nd calls, with no network or real tokens."""
    client = SimpleNamespace()
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(side_effect=list(responses)))
    )
    return client


def _tool_result_sent_to_llm(client) -> dict:
    """The content of the role="tool" message that was sent to the LLM on the 2nd call."""
    # await_args_list[1] = the kwargs of the 2nd invocation of create() (the 1st
    # is the one that "picks" the tool). We look there for the message agent.py
    # builds with the real tool result, to confirm nothing was lost or invented.
    second_call_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_messages, "the real tool result should be passed back to the LLM"
    return json.loads(tool_messages[0]["content"])


async def test_where_is_my_order_uses_get_order():
    # Case 1 from the spec: "Where is my order 123?" -> get_order(order_id=123).
    client = _fake_client(
        _tool_call_response("get_order", {"order_id": 123}),
        _text_response("Tu pedido 123 está en preparación, llega el 18 de agosto."),
    )

    await handle_message("¿Dónde está mi pedido 123?", customer_id=1, client=client)

    assert _tool_result_sent_to_llm(client) == {
        "success": True,
        "id": 123,
        "customer_id": 1,
        "status": "processing",
        "delivery_date": "2026-08-18",
    }


async def test_cancel_order_uses_cancel_order_and_mutates_state():
    # Case 2: successful cancellation. We check not just the result the LLM sees,
    # but that the order actually changed state in the system.
    client = _fake_client(
        _tool_call_response("cancel_order", {"order_id": 123}),
        _text_response("Listo, cancelé tu pedido 123."),
    )

    await handle_message("Cancelame el pedido 123", customer_id=1, client=client)

    assert _tool_result_sent_to_llm(client) == {
        "success": True,
        "order_id": 123,
        "status": "cancelled",
    }
    assert orders[123]["status"] == "cancelled"


async def test_change_delivery_date_uses_change_delivery_date_tool():
    # Case 3: changing the delivery date with natural language ("August 21, 2026"),
    # which the (here simulated) LLM must translate into new_date="2026-08-21" in its tool call.
    client = _fake_client(
        _tool_call_response("change_delivery_date", {"order_id": 123, "new_date": "2026-08-21"}),
        _text_response("Listo, tu pedido 123 llega el 21 de agosto."),
    )

    await handle_message(
        "Quiero cambiar la entrega del pedido 123 al 21 de agosto de 2026",
        customer_id=1,
        client=client,
    )

    assert _tool_result_sent_to_llm(client) == {
        "success": True,
        "order_id": 123,
        "delivery_date": "2026-08-21",
    }
    assert orders[123]["delivery_date"] == "2026-08-21"


async def test_cancel_nonexistent_order_never_reports_success():
    # Case 4 (negative): order 999 doesn't exist. The LLM still calls cancel_order
    # (it shouldn't "know" beforehand that it doesn't exist) and the tool is what reports the error.
    client = _fake_client(
        _tool_call_response("cancel_order", {"order_id": 999}),
        _text_response("No encontré ningún pedido con ese número."),
    )

    await handle_message("Cancelame el pedido 999", customer_id=1, client=client)

    tool_result = _tool_result_sent_to_llm(client)
    assert tool_result["success"] is False
    assert tool_result["error"] == "order_not_found"
    assert 999 not in orders


async def test_cancel_shipped_order_does_not_mutate_state():
    # Case 5 (negative): order 456 was already shipped. The business rule in
    # tools/orders.py must prevent cancellation, no matter what the LLM "wants".
    client = _fake_client(
        _tool_call_response("cancel_order", {"order_id": 456}),
        _text_response("No pude cancelar tu pedido 456 porque ya fue enviado."),
    )

    await handle_message("Cancelame el pedido 456", customer_id=1, client=client)

    tool_result = _tool_result_sent_to_llm(client)
    assert tool_result["success"] is False
    assert tool_result["error"] == "order_not_cancellable"
    # no matter what the final text says, a shipped order never actually gets cancelled
    assert orders[456]["status"] == "shipped"


async def test_agent_can_chain_dependent_tool_calls():
    """The second tool is selected only after the model sees the first result."""
    client = _fake_client(
        _tool_call_response("get_customer_orders", {}, call_id="list_orders"),
        _tool_call_response("cancel_order", {"order_id": 123}, call_id="cancel_order"),
        _text_response("Listo, cancelé tu pedido 123."),
    )

    response, conversation_id = await handle_message(
        "Cancelá mi pedido en procesamiento",
        customer_id=1,
        client=client,
    )

    assert response == "Listo, cancelé tu pedido 123."
    assert conversation_id in conversations
    assert client.chat.completions.create.await_count == 3
    assert orders[123]["status"] == "cancelled"


async def test_agent_executes_multiple_tool_calls_from_one_model_response():
    client = _fake_client(
        _multiple_tool_calls_response(
            ("get_123", "get_order", {"order_id": 123}),
            ("get_456", "get_order", {"order_id": 456}),
        ),
        _text_response("El pedido 123 está en preparación y el 456 fue enviado."),
    )

    await handle_message("Compará mis pedidos 123 y 456", customer_id=1, client=client)

    second_call_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [message for message in second_call_messages if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["get_123", "get_456"]
    assert [json.loads(message["content"])["id"] for message in tool_messages] == [123, 456]


async def test_conversation_history_is_reused_across_turns():
    client = _fake_client(
        _tool_call_response("get_customer_orders", {}),
        _text_response("Tenés los pedidos 123 y 456. ¿Cuál querés cancelar?"),
        _tool_call_response("cancel_order", {"order_id": 123}, call_id="call_2"),
        _text_response("Listo, cancelé tu pedido 123."),
    )

    _, conversation_id = await handle_message(
        "Quiero cancelar mi pedido",
        customer_id=1,
        client=client,
    )
    response, returned_id = await handle_message(
        "El 123",
        customer_id=1,
        conversation_id=conversation_id,
        client=client,
    )

    assert response == "Listo, cancelé tu pedido 123."
    assert returned_id == conversation_id
    fourth_call_messages = client.chat.completions.create.await_args_list[2].kwargs["messages"]
    assert {"role": "user", "content": "Quiero cancelar mi pedido"} in fourth_call_messages
    assert {"role": "user", "content": "El 123"} in fourth_call_messages


async def test_customer_cannot_reuse_another_customers_conversation():
    client = _fake_client(_text_response("Hola."))
    _, conversation_id = await handle_message("Hola", customer_id=1, client=client)

    try:
        await handle_message(
            "Continuemos",
            customer_id=2,
            conversation_id=conversation_id,
            client=client,
        )
    except ValueError as exc:
        assert str(exc) == "conversation_not_found"
    else:
        raise AssertionError("A customer must not access another customer's conversation")


async def test_agent_stops_after_iteration_limit():
    repeated_calls = [
        _tool_call_response("get_customer_orders", {}, call_id=f"call_{index}")
        for index in range(MAX_AGENT_ITERATIONS)
    ]
    client = _fake_client(*repeated_calls)

    response, _ = await handle_message("Seguí buscando", customer_id=1, client=client)

    assert "No pude completar" in response
    assert client.chat.completions.create.await_count == MAX_AGENT_ITERATIONS
