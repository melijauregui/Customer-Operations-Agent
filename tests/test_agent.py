"""Tests del loop de tool calling en app/agent.py, con el cliente de OpenAI mockeado.

No pegan a la red ni gastan tokens: simulamos la respuesta del LLM (qué tool
"decidió" llamar) y verificamos que la aplicación haya ejecutado la tool real,
que el resultado real (no inventado) haya vuelto al LLM, y que el estado de
`orders` haya cambiado solo cuando correspondía.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agent import handle_message
from app.tools.orders import orders


def _tool_call_response(name: str, arguments: dict, call_id: str = "call_1"):
    """Fabrica una respuesta falsa que imita la forma real de un ChatCompletion
    cuando el modelo decide llamar a una tool (en vez de responder texto)."""
    # `arguments` viaja como string JSON, igual que en la respuesta real de la API
    # (agent.py hace json.loads sobre esto, así que tiene que ser un string).
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    # content=None porque cuando el modelo pide una tool no devuelve texto todavía;
    # tool_calls es una lista porque la API soporta pedir más de una tool a la vez.
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    # agent.py solo lee response.choices[0].message, así que alcanza con simular eso.
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _text_response(text: str):
    """Fabrica la respuesta cuando el modelo NO pide ninguna tool y contesta directo
    (ej: la 2da llamada del loop, ya con el resultado de la tool en el historial)."""
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_client(*responses):
    """Doble de AsyncOpenAI: cada llamada a create() devuelve, en orden, uno de los
    `responses` pasados (side_effect). Así controlamos exactamente qué "decide" el
    LLM en la 1ra y la 2da llamada del loop, sin red ni tokens reales."""
    client = SimpleNamespace()
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(side_effect=list(responses)))
    )
    return client


def _tool_result_sent_to_llm(client) -> dict:
    """El contenido del mensaje role="tool" que se le mandó al LLM en la 2da llamada."""
    # await_args_list[1] = los kwargs de la 2da invocación a create() (la 1ra es la
    # que "elige" la tool). Ahí buscamos el mensaje que agent.py arma con el
    # resultado real de la tool para confirmar que no se perdió ni se inventó nada.
    second_call_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_messages, "el resultado real de la tool debería pasarse de vuelta al LLM"
    return json.loads(tool_messages[0]["content"])


async def test_where_is_my_order_uses_get_order():
    # Caso 1 del enunciado: "¿Dónde está mi pedido 123?" -> get_order(order_id=123).
    client = _fake_client(
        _tool_call_response("get_order", {"order_id": 123}),
        _text_response("Tu pedido 123 está en preparación, llega el 18 de agosto."),
    )

    await handle_message("¿Dónde está mi pedido 123?", client=client)

    assert _tool_result_sent_to_llm(client) == {
        "success": True,
        "id": 123,
        "status": "processing",
        "delivery_date": "2026-08-18",
    }


async def test_cancel_order_uses_cancel_order_and_mutates_state():
    # Caso 2: cancelación exitosa. Verificamos no solo el resultado que ve el LLM,
    # sino que el pedido haya cambiado de estado de verdad en el sistema.
    client = _fake_client(
        _tool_call_response("cancel_order", {"order_id": 123}),
        _text_response("Listo, cancelé tu pedido 123."),
    )

    await handle_message("Cancelame el pedido 123", client=client)

    assert _tool_result_sent_to_llm(client) == {
        "success": True,
        "order_id": 123,
        "status": "cancelled",
    }
    assert orders[123]["status"] == "cancelled"


async def test_change_delivery_date_uses_change_delivery_date_tool():
    # Caso 3: cambio de fecha de entrega con lenguaje natural ("21 de agosto de 2026"),
    # que el LLM (acá simulado) debe traducir a new_date="2026-08-21" en su tool call.
    client = _fake_client(
        _tool_call_response("change_delivery_date", {"order_id": 123, "new_date": "2026-08-21"}),
        _text_response("Listo, tu pedido 123 llega el 21 de agosto."),
    )

    await handle_message(
        "Quiero cambiar la entrega del pedido 123 al 21 de agosto de 2026", client=client
    )

    assert _tool_result_sent_to_llm(client) == {
        "success": True,
        "order_id": 123,
        "delivery_date": "2026-08-21",
    }
    assert orders[123]["delivery_date"] == "2026-08-21"


async def test_cancel_nonexistent_order_never_reports_success():
    # Caso 4 (negativo): pedido 999 no existe. El LLM igual llama a cancel_order
    # (no debe "saber" de antemano que no existe) y la tool es la que informa el error.
    client = _fake_client(
        _tool_call_response("cancel_order", {"order_id": 999}),
        _text_response("No encontré ningún pedido con ese número."),
    )

    await handle_message("Cancelame el pedido 999", client=client)

    tool_result = _tool_result_sent_to_llm(client)
    assert tool_result["success"] is False
    assert tool_result["error"] == "order_not_found"
    assert 999 not in orders


async def test_cancel_shipped_order_does_not_mutate_state():
    # Caso 5 (negativo): pedido 456 ya fue enviado. La regla de negocio en
    # tools/orders.py debe impedir la cancelación, sin importar qué "quiera" el LLM.
    client = _fake_client(
        _tool_call_response("cancel_order", {"order_id": 456}),
        _text_response("No pude cancelar tu pedido 456 porque ya fue enviado."),
    )

    await handle_message("Cancelame el pedido 456", client=client)

    tool_result = _tool_result_sent_to_llm(client)
    assert tool_result["success"] is False
    assert tool_result["error"] == "order_not_cancellable"
    # pase lo que pase en el texto final, el pedido shipped nunca se cancela de verdad
    assert orders[456]["status"] == "shipped"
