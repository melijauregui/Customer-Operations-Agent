
## Versions

1. **V0 — Single agent + 3 tools (current). DONE**
   See [v0.md](v0.md) for the full spec. Focus: tool calling, structured outputs, and the separation between the LLM and business logic.

2. **V1 — Real conversations**
   Add cases where information is missing:

   ```text
   User: "I want to cancel my order."
   Agent: "Which one of your orders?"
   ```

   The agent looks up:

   ```python
   get_customer_orders(customer_id)
   ```

   and keeps conversation state.

3. **V2 — LangGraph**
   This is where LangGraph would actually come in.

   Turn the flow into an explicit state machine:

   ```text
   START
     ↓
   understand_request
     ↓
   retrieve_information
     ↓
   decide_action
     ↓
   execute_tool
     ↓
   verify_result
     ↓
   END
   ```

   This is where you'll really understand what LangGraph is for.

4. **V3 — Simulator + evals**
   Before scaling up, make the agent measurable.

   Build 50-100 test cases:

   ```json
   {
     "user": "Cancel my order 123",
     "expected_tool": "cancel_order",
     "expected_arguments": {
       "order_id": 123
     }
   }
   ```

   And compute:

   ```text
   Tool Selection Accuracy
   Tool Argument Accuracy
   Task Success Rate
   Hallucination Rate
   ```

   This stage is **fundamental**.

5. **V4 — More complex Customer Operations**
   Add:

   ```python
   get_payment()
   refund_payment()
   create_return()
   replace_product()
   track_delivery()
   reschedule_delivery()
   check_inventory()
   ```

   And situations with more than one problem at once:

   > "I got charged twice and my order still hasn't arrived."

6. **V5 — RAG / policies**
   The agent can no longer just decide on its own whether to issue a refund.

   It has to consult:

   ```text
   return policy
   refund policy
   cancellation policy
   shipping policy
   ```

   That's where embeddings/vector DB or document search come in.

7. **V6 — Human-in-the-loop**
   Some actions:

   ```text
   refund $5 → automatic

   refund $2000 → requires approval
   ```

   The workflow stays suspended and can resume later.

8. **V7 — Multi-agent**
   Only once there's enough complexity:

   ```text
                Customer Agent
                     │
               Orchestrator
              /      |       \
          Orders  Payments  Delivery
           Agent    Agent     Agent
   ```

   And compare:

   ```text
   single-agent vs multi-agent
   ```

   including evals to see if it actually improved.

9. **V8 — Distribution**
   Now for real:

   ```text
   FastAPI
      ↓
   RabbitMQ
      ↓
   Agent Workers
   W1 W2 W3 W4
      ↓
   Redis/Postgres
   ```

   Simulate hundreds/thousands of concurrent executions.

10. **V9 — Production**
    Docker + AWS + OpenTelemetry + dashboards + load testing + retries + idempotency + rate limits + cost control.
