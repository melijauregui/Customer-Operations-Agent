"""FastAPI app: exposes the agent via POST /chat.

This layer has no logic of its own — it just takes the user's message, passes
it to the agent (app/agent.py), and returns whatever it responds.
"""

import logging

from fastapi import FastAPI

from app.agent import handle_message
from app.models import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Customer Operations Agent - V0")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    response_text = await handle_message(request.message)
    return ChatResponse(response=response_text)
