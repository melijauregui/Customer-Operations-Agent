"""Validated dispatch from model-generated tool requests to business functions."""

import json
import logging

from pydantic import ValidationError

from app.models import (
    CancelOrderArgs,
    ChangeDeliveryDateArgs,
    GetCustomerOrdersArgs,
    GetOrderArgs,
)
from app.tools import orders as orders_tools

logger = logging.getLogger(__name__)

_ARG_MODELS = {
    "get_order": GetOrderArgs,
    "get_customer_orders": GetCustomerOrdersArgs,
    "cancel_order": CancelOrderArgs,
    "change_delivery_date": ChangeDeliveryDateArgs,
}

# Every dispatch function receives customer_id from trusted application state.
# It is never read from model-generated arguments.
_TOOL_DISPATCH = {
    "get_order": lambda args, customer_id: orders_tools.get_order(args.order_id, customer_id),
    "get_customer_orders": lambda args, customer_id: orders_tools.get_customer_orders(customer_id),
    "cancel_order": lambda args, customer_id: orders_tools.cancel_order(args.order_id, customer_id),
    "change_delivery_date": lambda args, customer_id: orders_tools.change_delivery_date(
        args.order_id, args.new_date.isoformat(), customer_id
    ),
}


def execute_tool_call(name: str, arguments_json: str, customer_id: int) -> dict:
    """Validate model arguments and execute one matching business function."""
    arg_model = _ARG_MODELS.get(name)
    if arg_model is None:
        return {"success": False, "error": "unknown_tool", "message": f"Tool '{name}' no existe."}

    try:
        raw_args = json.loads(arguments_json)
        args = arg_model(**raw_args)
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        logger.warning(
            "tool_arguments_invalid tool=%s raw_args=%s error=%s",
            name,
            arguments_json,
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
