"""FastAPI app: expone el agente vía POST /chat.

Esta capa no tiene lógica propia — solo recibe el mensaje del usuario, se lo
pasa al agente (app/agent.py) y devuelve lo que este responda.
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
