"""LangGraph state and workflow construction for V2.

Step 1 defines only the shared state. The production agent still uses the V1
manual loop until the graph nodes and routing are implemented and tested.
"""

import operator
from typing import Annotated, TypedDict


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
