"""Lógica de negocio de pedidos. El LLM nunca ejecuta este código directamente:
solo puede pedir que se llame a una de estas funciones, con argumentos ya validados."""

from datetime import date

STATUS_PROCESSING = "processing"
STATUS_SHIPPED = "shipped"
STATUS_DELIVERED = "delivered"
STATUS_CANCELLED = "cancelled"

_CANCELLABLE_STATUSES = {STATUS_PROCESSING}

orders: dict[int, dict] = {
    123: {"id": 123, "status": STATUS_PROCESSING, "delivery_date": "2026-08-18"},
    456: {"id": 456, "status": STATUS_SHIPPED, "delivery_date": "2026-08-20"},
}


def get_order(order_id: int) -> dict:
    order = orders.get(order_id)
    if order is None:
        return {
            "success": False,
            "order_id": order_id,
            "error": "order_not_found",
            "message": f"No existe ningún pedido con id {order_id}.",
        }
    return {"success": True, **order}


def cancel_order(order_id: int) -> dict:
    order = orders.get(order_id)
    if order is None:
        return {
            "success": False,
            "order_id": order_id,
            "error": "order_not_found",
            "message": f"No existe ningún pedido con id {order_id}.",
        }

    if order["status"] == STATUS_CANCELLED:
        return {
            "success": False,
            "order_id": order_id,
            "error": "already_cancelled",
            "message": f"El pedido {order_id} ya estaba cancelado.",
        }

    if order["status"] not in _CANCELLABLE_STATUSES:
        return {
            "success": False,
            "order_id": order_id,
            "error": "order_not_cancellable",
            "message": f"El pedido {order_id} ya está en estado '{order['status']}' y no puede cancelarse.",
        }

    order["status"] = STATUS_CANCELLED
    return {"success": True, "order_id": order_id, "status": STATUS_CANCELLED}


def change_delivery_date(order_id: int, new_date: str) -> dict:
    order = orders.get(order_id)
    if order is None:
        return {
            "success": False,
            "order_id": order_id,
            "error": "order_not_found",
            "message": f"No existe ningún pedido con id {order_id}.",
        }

    try:
        parsed_date = date.fromisoformat(new_date)
    except ValueError:
        return {
            "success": False,
            "order_id": order_id,
            "error": "invalid_date",
            "message": f"'{new_date}' no es una fecha válida (formato esperado: YYYY-MM-DD).",
        }

    order["delivery_date"] = parsed_date.isoformat()
    return {"success": True, "order_id": order_id, "delivery_date": order["delivery_date"]}
