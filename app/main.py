"""FastAPI app: exposes the agent via POST /chat.

This layer has no logic of its own — it just takes the user's message, passes
it to the agent (app/agent.py), and returns whatever it responds.
"""

import logging

from fastapi import FastAPI, HTTPException

from app.agent import handle_message
from app.models import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Customer Operations Agent - V1")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        response_text, conversation_id = await handle_message(
            message=request.message,
            customer_id=request.customer_id,
            conversation_id=request.conversation_id,
        )
    except ValueError as exc:
        if str(exc) != "conversation_not_found":
            raise
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    return ChatResponse(response=response_text, conversation_id=conversation_id)
