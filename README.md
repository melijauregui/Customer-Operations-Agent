# Customer Operations Agent

AI agent platform for Customer Operations, built progressively across versions. **V2 is in progress**: the V1 agent loop is being migrated to a direct LangGraph workflow with explicit nodes, routing, result verification, and checkpointed conversation state.


## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Setting up the API key

```bash
cp .env.example .env
```

Edit `.env` and fill in your `OPENAI_API_KEY` (optionally `OPENAI_MODEL`, default `gpt-4o-mini`). `.env` is gitignored — it's never committed.

## Running FastAPI

```bash
uvicorn app.main:app --reload
```

Server at `http://127.0.0.1:8000`. Interactive docs (Swagger) at `http://127.0.0.1:8000/docs`.

## Running the tests

The unit tests don't spend tokens or depend on OpenAI. The LLM is mocked where needed, including the V2 graph tests. These are the tests that run by default:

```bash
pytest
```

The V2 integration tests in `test_integration.py` call the real OpenAI API and are excluded from the normal run (see `pytest.ini`). They exercise real tool selection, dependent multi-tool execution, LangGraph checkpoints, and customer isolation. Run them deliberately with a real `OPENAI_API_KEY`:

```bash
pytest -m integration
```

Run one integration test while iterating:

```bash
pytest -m integration tests/test_integration.py::test_v2_real_model_can_chain_dependent_tools -v
```

Because these tests depend on a remote probabilistic model, they are slower,
cost tokens, and may occasionally need a retry even though their assertions
focus on structured tool calls rather than exact response wording.

## Example requests

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order 123?", "customer_id": 1}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancel order 123", "customer_id": 1}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to change the delivery of order 123 to August 21, 2026", "customer_id": 1}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancel order 999", "customer_id": 1}'
# the order doesn't exist -> the agent reports it, never invents a cancellation
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancel order 456", "customer_id": 1}'
# order 456 was already shipped -> it can't be cancelled, the agent explains why
```

Start a new conversation by omitting `conversation_id`. Continue it by sending
the ID returned by the first response:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to cancel my order", "customer_id": 1}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "The 123 one", "customer_id": 1, "conversation_id": "<id-from-the-first-response>"}'
```


## Versions

- **V0 (done)** — single agent and three tools. See [versions/v0.md](versions/v0.md).
- **V1 (done)** — conversation state, customer-scoped orders, and multi-step tool calling. See [versions/v1.md](versions/v1.md).
- **V2 (in progress)** — direct LangGraph workflow, explicit verification, and checkpointed state. See [versions/v2.md](versions/v2.md).
- **V3 → V9 (planned)** — see [versions/next.md](versions/next.md) for the roadmap: evals, RAG, multi-agent, distribution, and production.
