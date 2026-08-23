"""Pydantic schemas: HTTP contract (Chat*) and structured arguments for each tool.

The *Args models validate/convert what the LLM sends as arguments for a tool call,
before those values reach the business logic in app/tools/orders.py.

Important: `customer_id` never appears in these *Args. It's session context (it
comes from ChatRequest, not from what the LLM decides) and gets injected in
agent.py — never a parameter the model fills in, so it can't "choose" to act as
another customer.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict


class ChatRequest(BaseModel):
    message: str
    customer_id: int
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


class ToolArgs(BaseModel):
    """Base validation shared by model-generated tool arguments."""

    model_config = ConfigDict(extra="forbid")


class GetOrderArgs(ToolArgs):
    order_id: int


class GetCustomerOrdersArgs(ToolArgs):
    """No fields: the LLM doesn't choose which customer's orders to list — the
    session's customer_id already determines that."""


class CancelOrderArgs(ToolArgs):
    order_id: int


class ChangeDeliveryDateArgs(ToolArgs):
    order_id: int
    new_date: date
