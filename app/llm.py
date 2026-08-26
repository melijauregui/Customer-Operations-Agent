"""Prompt, tool schemas, and message conversion shared by agent orchestrators."""

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


def assistant_message_to_dict(assistant_message) -> dict:
    """Convert an OpenAI SDK message into plain serializable graph state."""
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
