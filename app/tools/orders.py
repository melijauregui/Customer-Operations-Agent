"""Order business logic. The LLM never executes this code directly:
it can only request that one of these functions be called, with already-validated arguments."""

from datetime import date

STATUS_PROCESSING = "processing"
STATUS_SHIPPED = "shipped"
STATUS_DELIVERED = "delivered"
STATUS_CANCELLED = "cancelled"

_CANCELLABLE_STATUSES = {STATUS_PROCESSING}

orders: dict[int, dict] = {
    123: {"id": 123, "customer_id": 1, "status": STATUS_PROCESSING, "delivery_date": "2026-08-18"},
    456: {"id": 456, "customer_id": 1, "status": STATUS_SHIPPED, "delivery_date": "2026-08-20"},
    789: {"id": 789, "customer_id": 2, "status": STATUS_DELIVERED, "delivery_date": "2026-08-10"},
}


def _not_found_error(order_id: int) -> dict:
    return {
        "success": False,
        "order_id": order_id,
        "error": "order_not_found",
        "message": f"No existe ningún pedido con id {order_id}.",
    }


def _find_owned_order(order_id: int, customer_id: int) -> dict | None:
    """Looks up the order and validates it belongs to the customer requesting it.
    If it's not theirs, treated the same as "not found" — we never confirm the
    existence of someone else's order."""
    order = orders.get(order_id)
    if order is None or order["customer_id"] != customer_id:
        return None
    return order


def get_order(order_id: int, customer_id: int) -> dict:
    order = _find_owned_order(order_id, customer_id)
    if order is None:
        return _not_found_error(order_id)
    return {"success": True, **order}


def get_customer_orders(customer_id: int) -> dict:
    """Lists a customer's orders. An empty list is not an error: it means that
    customer has no orders, not that something went wrong."""
    matching_orders = [order for order in orders.values() if order["customer_id"] == customer_id]
    return {"success": True, "customer_id": customer_id, "orders": matching_orders}


def cancel_order(order_id: int, customer_id: int) -> dict:
    order = _find_owned_order(order_id, customer_id)
    if order is None:
        return _not_found_error(order_id)

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


def change_delivery_date(order_id: int, new_date: str, customer_id: int) -> dict:
    order = _find_owned_order(order_id, customer_id)
    if order is None:
        return _not_found_error(order_id)

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
