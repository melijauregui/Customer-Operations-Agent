"""LLM integration, conversation state, and the tool-calling loop.

The model decides which tool to call and with which business arguments. The
application injects trusted session context, validates arguments, executes the
real business functions, and sends their results back to the model.
"""

import json
import logging
from uuid import uuid4

from app.config import OPENAI_MODEL, get_client
from app.llm import SYSTEM_PROMPT, TOOLS, assistant_message_to_dict
from app.tool_executor import execute_tool_call

logger = logging.getLogger(__name__)

# This guard prevents a confused model from creating an infinite tool loop.
MAX_AGENT_ITERATIONS = 5

# V1 keeps state in process memory. Each conversation is bound to one customer
# so a conversation id cannot be reused to operate as another customer.
conversations: dict[str, dict] = {}

def _load_conversation(conversation_id: str | None, customer_id: int) -> tuple[str, list]:
    """Create a conversation or return a copy of an existing customer's history."""
    if conversation_id is None:
        return str(uuid4()), []

    conversation = conversations.get(conversation_id)
    if conversation is None or conversation["customer_id"] != customer_id:
        # Use the same error for unknown and foreign ids to avoid leaking state.
        raise ValueError("conversation_not_found")
    return conversation_id, list(conversation["messages"])


async def handle_message(
    message: str,
    customer_id: int,
    conversation_id: str | None = None,
    client=None,
) -> tuple[str, str]:
    """Run tool rounds until the model produces final text or reaches the safety limit."""
    client = client or get_client()
    conversation_id, history = _load_conversation(conversation_id, customer_id)
    logger.info(
        "user_message=%s customer_id=%s conversation_id=%s",
        message,
        customer_id,
        conversation_id,
    )

    # Work on a local history copy. It is committed to in-memory state only
    # after this turn reaches a final answer.
    history.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    iteration = 0
    while iteration < MAX_AGENT_ITERATIONS:
        iteration += 1
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        assistant_message = response.choices[0].message
        assistant_history_message = assistant_message_to_dict(assistant_message)
        messages.append(assistant_history_message)
        history.append(assistant_history_message)

        # No tool calls means the model completed the turn with user-facing text.
        if not assistant_message.tool_calls:
            final_text = assistant_message.content or ""
            conversations[conversation_id] = {
                "customer_id": customer_id,
                "messages": history,
            }
            logger.info("final_response=%s", final_text)
            return final_text, conversation_id

        # A single model response may contain several independent tool calls.
        # After executing all of them, the loop calls the model again so it can
        # either answer or request another tool based on these real results.
        for tool_call in assistant_message.tool_calls:
            result = execute_tool_call(
                name=tool_call.function.name,
                arguments_json=tool_call.function.arguments,
                customer_id=customer_id,
            )
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
            messages.append(tool_message)
            history.append(tool_message)

        logger.info("agent_iteration=%s requested_more_processing=true", iteration)

    # This deterministic fallback avoids asking the model for an unbounded
    # number of tool calls and never claims that an operation succeeded.
    final_text = "No pude completar la solicitud de forma segura. Por favor, intentá nuevamente."
    history.append({"role": "assistant", "content": final_text})
    conversations[conversation_id] = {
        "customer_id": customer_id,
        "messages": history,
    }
    logger.warning("agent_iteration_limit_reached conversation_id=%s", conversation_id)
    logger.info("final_response=%s", final_text)
    return final_text, conversation_id
