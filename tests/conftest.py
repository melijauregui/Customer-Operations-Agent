"""Shared fixtures.

`orders` in app/tools/orders.py is a mutable global dict (our in-memory "DB").
Without resetting it, a test that cancels order 123 would leak that state into
tests that run afterward. reset_orders() restores it before every test.
"""

import copy

import pytest

from app.tools import orders as orders_module
from app.agent import conversations

_INITIAL_ORDERS = copy.deepcopy(orders_module.orders)


@pytest.fixture(autouse=True)
def reset_orders():
    orders_module.orders.clear()
    orders_module.orders.update(copy.deepcopy(_INITIAL_ORDERS))
    conversations.clear()
