"""LangGraph state, nodes, routing, and workflow construction for V2.

The production agent still uses the V1 manual loop until the graph has result
verification, conversation checkpoints, and behavioral parity.
"""

import json
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import OPENAI_MODEL
from app.llm import SYSTEM_PROMPT, TOOLS, assistant_message_to_dict
from app.tool_executor import execute_tool_call

logger = logging.getLogger(__name__)

SAFE_VERIFICATION_FAILURE = (
    "No pude verificar de forma segura el resultado de la operación. "
    "Por favor, intentá nuevamente."
)


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


def _verification_failure(error: str) -> dict:
    """Build a safe terminal update without exposing internal details."""
    logger.error("graph_node=verify_results verification_error=%s", error)
    return {
        "verification_error": error,
        "messages": [
            {
                "role": "assistant",
                "content": SAFE_VERIFICATION_FAILURE,
            }
        ],
    }


def verify_results(state: AgentState) -> dict:
    """Verify the structural contract between requested tools and their results.

    `success=False` is a valid business result and must go back to the model.
    Verification fails only when the workflow itself cannot prove which result
    belongs to which tool call or cannot parse the expected result envelope.

    Example of a valid state received after `execute_tools`:
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Cancel order 123",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "cancel_order",
                                "arguments": '{"order_id": 123}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": '{"success": true, "order_id": 123, "status": "cancelled"}',
                },
            ],
            "customer_id": 1,
            "tool_iterations": 1,
            "verification_error": None,
        }
    """
    assistant_index = None
    for index in range(len(state["messages"]) - 1, -1, -1):
        message = state["messages"][index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            assistant_index = index
            break

    if assistant_index is None:
        return _verification_failure("missing assistant tool-call message")

    assistant_message = state["messages"][assistant_index]
    try:
        expected_ids = [tool_call["id"] for tool_call in assistant_message["tool_calls"]]
    except (KeyError, TypeError):
        return _verification_failure("malformed assistant tool-call message")

    messages_after_call = state["messages"][assistant_index + 1 :]
    tool_messages = [message for message in messages_after_call if message.get("role") == "tool"]
    actual_ids = [message.get("tool_call_id") for message in tool_messages]

    if actual_ids != expected_ids:
        return _verification_failure(
            f"tool result ids do not match: expected={expected_ids} actual={actual_ids}"
        )

    for tool_message in tool_messages:
        try:
            result = json.loads(tool_message["content"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return _verification_failure(
                f"invalid JSON result for tool_call_id={tool_message.get('tool_call_id')}"
            )

        if not isinstance(result, dict) or type(result.get("success")) is not bool:
            return _verification_failure(
                f"invalid result contract for tool_call_id={tool_message.get('tool_call_id')}"
            )

    logger.info("graph_node=verify_results verified_count=%s", len(tool_messages))
    return {"verification_error": None}


def after_verification(state: AgentState) -> str:
    """Continue model reasoning only when tool results passed verification."""
    if state["verification_error"]:
        return END
    return "call_model"


def build_graph(*, client):
    """Build the first executable V2 graph without persistence.

    The OpenAI client is captured by the model-node closure, keeping it out of
    checkpointable graph state and allowing tests to inject a deterministic
    client double.
    """

    async def model_node(state: AgentState) -> dict:
        return await call_model(state, client=client)

    builder = StateGraph(AgentState)
    builder.add_node("call_model", model_node)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("verify_results", verify_results)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "execute_tools": "execute_tools",
            END: END,
        },
    )
    builder.add_edge("execute_tools", "verify_results")
    builder.add_conditional_edges(
        "verify_results",
        after_verification,
        {
            "call_model": "call_model",
            END: END,
        },
    )

    return builder.compile()
