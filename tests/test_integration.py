"""Real integration test against the OpenAI API.

Doesn't run in a normal `pytest` invocation (see addopts in pytest.ini). Run it
on purpose with:
    pytest -m integration
It requires a real OPENAI_API_KEY set in the environment or in .env.
"""

import os

import pytest

from app.agent import handle_message

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY")
async def test_agent_selects_get_order_for_real_question():
    response = await handle_message("¿Dónde está mi pedido 123?")
    assert "123" in response
