"""LLM integration, conversation state, and the tool-calling loop.

The model decides which tool to call and with which business arguments. The
application injects trusted session context, validates arguments, executes the
real business functions, and sends their results back to the model.
"""

import json
import logging
from uuid import uuid4

from pydantic import ValidationError

from app.config import OPENAI_MODEL, get_client
from app.models import (
    CancelOrderArgs,
    ChangeDeliveryDateArgs,
    GetCustomerOrdersArgs,
    GetOrderArgs,
)
from app.tools import orders as orders_tools

logger = logging.getLogger(__name__)

# This guard prevents a confused model from creating an infinite tool loop.
MAX_AGENT_ITERATIONS = 5

# V1 keeps state in process memory. Each conversation is bound to one customer
# so a conversation id cannot be reused to operate as another customer.
conversations: dict[str, dict] = {}

SYSTEM_PROMPT = (
    "Sos un agente de atención al cliente de un e-commerce. Para consultar, "
    "modificar o cancelar pedidos, usá siempre las tools disponibles; nunca inventes el "
    "resultado de una acción ni afirmes que algo se hizo si no llamaste a la tool "
    "correspondiente. Si una tool devuelve success=false, explicale el problema "
    "al usuario sin decir que la acción tuvo éxito. El customer_id es contexto seguro "
    "de la aplicación: nunca se lo pidas al usuario ni intentes elegirlo."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get the status and delivery date of one of the current customer's orders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "The order id."},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_orders",
            "description": "List all orders belonging to the current authenticated customer.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel one of the current customer's orders, if it is cancellable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "The order id."},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_delivery_date",
            "description": "Change the delivery date of one of the current customer's orders.",
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
                "additionalProperties": False,
            },
        },
    },
]

_ARG_MODELS = {
    "get_order": GetOrderArgs,
    "get_customer_orders": GetCustomerOrdersArgs,
    "cancel_order": CancelOrderArgs,
    "change_delivery_date": ChangeDeliveryDateArgs,
}

# Every dispatch function receives customer_id from the trusted application
# context. It is never read from model-generated tool arguments.
_TOOL_DISPATCH = {
    "get_order": lambda args, customer_id: orders_tools.get_order(args.order_id, customer_id),
    "get_customer_orders": lambda args, customer_id: orders_tools.get_customer_orders(customer_id),
    "cancel_order": lambda args, customer_id: orders_tools.cancel_order(args.order_id, customer_id),
    "change_delivery_date": lambda args, customer_id: orders_tools.change_delivery_date(
        args.order_id, args.new_date.isoformat(), customer_id
    ),
}


def _assistant_message_to_dict(assistant_message) -> dict:
    """Convert an SDK message into plain data suitable for API history and storage."""
    message = {"role": "assistant", "content": assistant_message.content}
    if assistant_message.tool_calls:
        message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in assistant_message.tool_calls
        ]
    return message


def _execute_tool_call(tool_call, customer_id: int) -> dict:
    """Validate one model request and execute the matching business function."""
    name = tool_call.function.name
    arg_model = _ARG_MODELS.get(name)
    if arg_model is None:
        return {"success": False, "error": "unknown_tool", "message": f"Tool '{name}' no existe."}

    try:
        raw_args = json.loads(tool_call.function.arguments)
        args = arg_model(**raw_args)
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        logger.warning(
            "tool_arguments_invalid tool=%s raw_args=%s error=%s",
            name,
            tool_call.function.arguments,
            exc,
        )
        return {
            "success": False,
            "error": "invalid_arguments",
            "message": f"Argumentos inválidos para '{name}': {exc}",
        }

    logger.info("tool_selected=%s tool_arguments=%s", name, args.model_dump(mode="json"))
    result = _TOOL_DISPATCH[name](args, customer_id)
    logger.info("tool_result=%s", result)
    return result


def _load_conversation(conversation_id: str | None, customer_id: int) -> tuple[str, list]:
    """Create a conversation or return a copy of an existing customer's history."""
    if conversation_id is None:
        return str(uuid4()), []

    conversation = conversations.get(conversation_id)
    if conversation is None or conversation["customer_id"] != customer_id:
        # Use the same error for unknown and foreign ids to avoid leaking state.
        raise ValueError("conversation_not_found")
    return conversation_id, list(conversation["messages"])


async def handle_message(
    message: str,
    customer_id: int,
    conversation_id: str | None = None,
    client=None,
) -> tuple[str, str]:
    """Run tool rounds until the model produces final text or reaches the safety limit."""
    client = client or get_client()
    conversation_id, history = _load_conversation(conversation_id, customer_id)
    logger.info(
        "user_message=%s customer_id=%s conversation_id=%s",
        message,
        customer_id,
        conversation_id,
    )

    # Work on a local history copy. It is committed to in-memory state only
    # after this turn reaches a final answer.
    history.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    iteration = 0
    while iteration < MAX_AGENT_ITERATIONS:
        iteration += 1
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        assistant_message = response.choices[0].message
        assistant_history_message = _assistant_message_to_dict(assistant_message)
        messages.append(assistant_history_message)
        history.append(assistant_history_message)

        # No tool calls means the model completed the turn with user-facing text.
        if not assistant_message.tool_calls:
            final_text = assistant_message.content or ""
            conversations[conversation_id] = {
                "customer_id": customer_id,
                "messages": history,
            }
            logger.info("final_response=%s", final_text)
            return final_text, conversation_id

        # A single model response may contain several independent tool calls.
        # After executing all of them, the loop calls the model again so it can
        # either answer or request another tool based on these real results.
        for tool_call in assistant_message.tool_calls:
            result = _execute_tool_call(tool_call, customer_id)
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
            messages.append(tool_message)
            history.append(tool_message)

        logger.info("agent_iteration=%s requested_more_processing=true", iteration)

    # This deterministic fallback avoids asking the model for an unbounded
    # number of tool calls and never claims that an operation succeeded.
    final_text = "No pude completar la solicitud de forma segura. Por favor, intentá nuevamente."
    history.append({"role": "assistant", "content": final_text})
    conversations[conversation_id] = {
        "customer_id": customer_id,
        "messages": history,
    }
    logger.warning("agent_iteration_limit_reached conversation_id=%s", conversation_id)
    logger.info("final_response=%s", final_text)
    return final_text, conversation_id
