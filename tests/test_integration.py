"""Test de integración real contra la API de OpenAI.

No corre en `pytest` normal (ver addopts en pytest.ini). Se ejecuta a propósito con:
    pytest -m integration
y requiere OPENAI_API_KEY real configurada en el entorno o en .env.
"""

import os

import pytest

from app.agent import handle_message

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requiere OPENAI_API_KEY real")
async def test_agent_selects_get_order_for_real_question():
    response = await handle_message("¿Dónde está mi pedido 123?")
    assert "123" in response
