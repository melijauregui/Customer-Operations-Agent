"""Configuración y creación del cliente de OpenAI.

get_client() se llama en tiempo de ejecución (no a nivel de módulo) para que los
tests puedan reemplazar fácilmente el cliente sin necesitar el API key real.
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
