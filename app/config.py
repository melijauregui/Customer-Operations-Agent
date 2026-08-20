"""Configuration and creation of the OpenAI client.

get_client() is called at runtime (not at module level) so tests can easily
swap out the client without needing a real API key.
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
