"""Unit tests for the business logic in app/tools/orders.py — no LLM involved."""

from app.tools.orders import cancel_order, change_delivery_date, get_order


def test_get_existing_order():
    assert get_order(123) == {
        "success": True,
        "id": 123,
        "status": "processing",
        "delivery_date": "2026-08-18",
    }


def test_get_nonexistent_order():
    result = get_order(999)
    assert result["success"] is False
    assert result["error"] == "order_not_found"


def test_cancel_cancellable_order():
    assert cancel_order(123) == {"success": True, "order_id": 123, "status": "cancelled"}


def test_cancel_nonexistent_order():
    result = cancel_order(999)
    assert result["success"] is False
    assert result["error"] == "order_not_found"


def test_cancel_already_cancelled_order():
    cancel_order(123)
    result = cancel_order(123)
    assert result["success"] is False
    assert result["error"] == "already_cancelled"


def test_cancel_shipped_order():
    result = cancel_order(456)
    assert result["success"] is False
    assert result["error"] == "order_not_cancellable"


def test_change_delivery_date_successfully():
    assert change_delivery_date(123, "2026-08-21") == {
        "success": True,
        "order_id": 123,
        "delivery_date": "2026-08-21",
    }


def test_change_delivery_date_nonexistent_order():
    result = change_delivery_date(999, "2026-08-21")
    assert result["success"] is False
    assert result["error"] == "order_not_found"


def test_change_delivery_date_invalid_date():
    result = change_delivery_date(123, "21 de agosto")
    assert result["success"] is False
    assert result["error"] == "invalid_date"
