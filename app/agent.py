"""LLM integration and tool-calling loop.

The model decides WHICH tool to call and with what arguments (native OpenAI
tool/function calling) — this module never decides the action via its own rules
(no `if "cancel" in message`). All business logic lives in app/tools/orders.py;
this only orchestrates: LLM -> tool -> LLM.
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
    """Runs the full loop: LLM picks a tool -> it gets executed -> LLM writes the final answer."""
    # client=None in production -> a real one gets created; tests pass a mock here.
    client = client or get_client()
    logger.info("user_message=%s", message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    # 1st call to the LLM: we pass the available tools and the model decides
    # whether it needs one (tool_choice="auto" is the API's default).
    first_response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=TOOLS,
    )
    assistant_message = first_response.choices[0].message

    # The model answered directly, without requesting any tool (e.g. a greeting,
    # a question that doesn't need real data). Nothing to execute.
    if not assistant_message.tool_calls:
        final_text = assistant_message.content or ""
        logger.info("final_response=%s", final_text)
        return final_text

    # The model requested one or more tools: we execute them ourselves (never
    # the LLM directly) and return each result as a role="tool" message, linked
    # by tool_call_id to the call that triggered it.
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

    # 2nd call to the LLM: it now has the real result of the tool(s) in the
    # history and just writes the response based on that (it doesn't decide on
    # tools again here, so we don't pass `tools=`).
    second_response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )
    final_text = second_response.choices[0].message.content or ""
    logger.info("final_response=%s", final_text)
    return final_text
