# Customer Operations Agent

AI agent platform for Customer Operations, built progressively across versions. This is **V0**: a single agent, three tools, and a hard focus on understanding tool calling, structured outputs, and the separation between the LLM and business logic.


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

The unit tests (`test_tools.py` + `test_agent.py`) don't spend tokens or depend on OpenAI — the LLM is mocked in `test_agent.py`. These are the ones that run by default:

```bash
pytest
```

The integration test (`test_integration.py`) calls the real OpenAI API and is excluded from the normal run (see `pytest.ini`). Run it on purpose, and it requires a real `OPENAI_API_KEY`:

```bash
pytest -m integration
```

## Example requests

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order 123?"}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancel order 123"}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to change the delivery of order 123 to August 21, 2026"}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancel order 999"}'
# the order doesn't exist -> the agent reports it, never invents a cancellation
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancel order 456"}'
# order 456 was already shipped -> it can't be cancelled, the agent explains why
```


## Versions

- **V0 (current)** — single agent, three tools (`get_order`, `cancel_order`, `change_delivery_date`). See [versions/v0.md](versions/v0.md) for the full spec: goal, architecture, request flow, and current limitations.
- **V1 → V9 (planned)** — see [versions/next.md](versions/next.md) for the full roadmap: conversational state, LangGraph, evals, RAG, multi-agent, distribution, production.
