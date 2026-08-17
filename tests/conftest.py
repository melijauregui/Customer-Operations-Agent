"""Fixtures compartidas.

`orders` en app/tools/orders.py es un dict global mutable (nuestra "DB" en memoria).
Sin resetearlo, un test que cancela el pedido 123 dejaría ese estado filtrado a
los tests que corren después. reset_orders() lo repone antes de cada test.
"""

import copy

import pytest

from app.tools import orders as orders_module

_INITIAL_ORDERS = copy.deepcopy(orders_module.orders)


@pytest.fixture(autouse=True)
def reset_orders():
    orders_module.orders.clear()
    orders_module.orders.update(copy.deepcopy(_INITIAL_ORDERS))
