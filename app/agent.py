"""Integración con el LLM y loop de tool calling.

El modelo decide QUÉ tool llamar y con qué argumentos (tool/function calling nativo
de OpenAI) — este módulo nunca decide la acción por reglas propias (nada de
`if "cancelar" in message`). Toda la lógica de negocio vive en app/tools/orders.py;
acá solo se orquesta: LLM -> tool -> LLM.
"""

import json
import logging

from pydantic import ValidationError

from app.config import OPENAI_MODEL, get_client
from app.models import CancelOrderArgs, ChangeDeliveryDateArgs, GetOrderArgs
from app.tools import orders as orders_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sos un agente de atención al cliente de un e-commerce. Para consultar, "
    "modificar o cancelar pedidos, usá siempre las tools disponibles; nunca inventes el "
    "resultado de una acción ni afirmes que algo se hizo si no llamaste a la tool "
    "correspondiente. Si una tool devuelve success=false, explicale el problema "
    "al usuario sin decir que la acción tuvo éxito."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get the status and delivery date of an existing order by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "The order id."},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an existing order, if it is still in a cancellable state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "The order id."},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_delivery_date",
            "description": "Change the delivery date of an existing order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "The order id."},
                    "new_date": {
                        "type": "string",
                        "description": "New delivery date, in YYYY-MM-DD format.",
                    },
                },
                "required": ["order_id", "new_date"],
            },
        },
    },
]

_ARG_MODELS = {
    "get_order": GetOrderArgs,
    "cancel_order": CancelOrderArgs,
    "change_delivery_date": ChangeDeliveryDateArgs,
}

_TOOL_DISPATCH = {
    "get_order": lambda args: orders_tools.get_order(args.order_id),
    "cancel_order": lambda args: orders_tools.cancel_order(args.order_id),
    "change_delivery_date": lambda args: orders_tools.change_delivery_date(
        args.order_id, args.new_date.isoformat()
    ),
}


def _execute_tool_call(tool_call) -> dict:
    name = tool_call.function.name
    raw_args = json.loads(tool_call.function.arguments)

    arg_model = _ARG_MODELS.get(name)
    if arg_model is None:
        return {"success": False, "error": "unknown_tool", "message": f"Tool '{name}' no existe."}

    try:
        args = arg_model(**raw_args)
    except ValidationError as exc:
        logger.warning("tool_arguments_invalid tool=%s raw_args=%s error=%s", name, raw_args, exc)
        return {
            "success": False,
            "error": "invalid_arguments",
            "message": f"Argumentos inválidos para '{name}': {exc}",
        }

    logger.info("tool_selected=%s tool_arguments=%s", name, args.model_dump(mode="json"))
    result = _TOOL_DISPATCH[name](args)
    logger.info("tool_result=%s", result)
    return result


async def handle_message(message: str, client=None) -> str:
    """Corre el loop completo: LLM elige tool -> se ejecuta -> LLM redacta respuesta final."""
    # client=None en producción -> se crea el real; en tests se pasa un mock acá.
    client = client or get_client()
    logger.info("user_message=%s", message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    # 1ra llamada al LLM: le pasamos las tools disponibles y el modelo decide
    # si necesita alguna (tool_choice="auto" es el default de la API).
    first_response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=TOOLS,
    )
    assistant_message = first_response.choices[0].message

    # El modelo respondió directo, sin pedir ninguna tool (ej: saludo, pregunta
    # que no requiere datos reales). No hay nada que ejecutar.
    if not assistant_message.tool_calls:
        final_text = assistant_message.content or ""
        logger.info("final_response=%s", final_text)
        return final_text

    # El modelo pidió una o más tools: las ejecutamos nosotros (nunca el LLM
    # directamente) y devolvemos cada resultado como mensaje role="tool",
    # enlazado por tool_call_id a la llamada que lo originó.
    messages.append(assistant_message)
    for tool_call in assistant_message.tool_calls:
        result = _execute_tool_call(tool_call)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

    # 2da llamada al LLM: ahora tiene el resultado real de la(s) tool(s) en el
    # historial y solo redacta la respuesta en base a eso (no vuelve a decidir
    # tools acá, así que no le pasamos `tools=`).
    second_response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )
    final_text = second_response.choices[0].message.content or ""
    logger.info("final_response=%s", final_text)
    return final_text
