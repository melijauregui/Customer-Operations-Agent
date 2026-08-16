"""Schemas Pydantic: contrato HTTP (Chat*) y argumentos estructurados de cada tool.

Los *Args se usan para validar/convertir lo que el LLM manda como argumentos de una
tool call, antes de que esos valores lleguen a la lógica de negocio en app/tools/orders.py.
"""

from datetime import date

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class GetOrderArgs(BaseModel):
    order_id: int


class CancelOrderArgs(BaseModel):
    order_id: int


class ChangeDeliveryDateArgs(BaseModel):
    order_id: int
    new_date: date
