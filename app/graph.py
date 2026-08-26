"""LangGraph state and workflow construction for V2.

Step 1 defines only the shared state. The production agent still uses the V1
manual loop until the graph nodes and routing are implemented and tested.
"""

import json
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END

from app.config import OPENAI_MODEL
from app.llm import SYSTEM_PROMPT, TOOLS, assistant_message_to_dict
from app.tool_executor import execute_tool_call

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared state that will move through every node in the agent graph.

    This class defines the state schema; it does not create one persistent
    object per customer. LangGraph will keep a separate state for each
    conversation, identified by its `thread_id` (our API's `conversation_id`).
    Therefore, the same customer may own several conversations, each with its
    own message history but the same `customer_id`.

    `AgentState` is a TypedDict, so each concrete state is still a normal Python
    dictionary. The TypedDict provides type information; it does not perform
    runtime validation like Pydantic.

    `messages` uses a reducer so node updates append new messages instead of
    replacing the complete conversation history.
    """

    messages: Annotated[list[dict], operator.add]
    customer_id: int
    tool_iterations: int
    verification_error: str | None


async def call_model(state: AgentState, *, client) -> dict:
    """Ask the LLM to either answer the user or request one or more tools.

    The node receives the complete graph state but returns only its state
    update. The messages reducer will append the returned assistant message to
    the existing conversation history.

    `client` is injected by the graph builder (and directly by unit tests), so
    this node never creates a real OpenAI client or requires an API key itself.
    """
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *state["messages"],
        ],
        tools=TOOLS,
    )
    assistant_message = response.choices[0].message
    state_message = assistant_message_to_dict(assistant_message)

    logger.info("graph_node=call_model tool_calls=%s", bool(assistant_message.tool_calls))
    return {"messages": [state_message]}


def should_continue(state: AgentState) -> str:
    """Route from the model node to tool execution or graph completion.

    Routing depends only on the structured `tool_calls` field produced by the
    model. It never parses natural-language content or uses keyword rules such
    as checking whether the message contains "cancel".

    This function runs immediately after `call_model`, so the last state
    message must have the assistant role.
    """
    last_message = state["messages"][-1]

    if last_message.get("role") != "assistant":
        raise ValueError("should_continue requires an assistant message")

    if last_message.get("tool_calls"):
        return "execute_tools"

    return END


def execute_tools(state: AgentState) -> dict:
    """Execute all tool calls requested in the latest assistant message.

    The node validates model-generated business arguments through the shared
    tool executor and injects `customer_id` from trusted graph state. It returns
    one `role="tool"` message per request and never writes user-facing claims.
    """
    assistant_message = state["messages"][-1]
    tool_calls = assistant_message.get("tool_calls")

    if assistant_message.get("role") != "assistant" or not tool_calls:
        raise ValueError("execute_tools requires an assistant message with tool calls")

    tool_messages = []
    for tool_call in tool_calls:
        function = tool_call["function"]
        result = execute_tool_call(
            name=function["name"],
            arguments_json=function["arguments"],
            customer_id=state["customer_id"],
        )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            }
        )

    logger.info("graph_node=execute_tools tool_count=%s", len(tool_messages))
    return {
        "messages": tool_messages,
        "tool_iterations": state["tool_iterations"] + 1,
    }
