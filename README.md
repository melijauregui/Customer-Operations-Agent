# Customer Operations Agent

Plataforma de agentes de IA para Customer Operations, desarrollada progresivamente en versiones. Esta es la **V0**: un único agente, tres tools, y foco absoluto en entender bien tool calling, structured outputs y la separación entre el LLM y la lógica de negocio.

## Objetivo de la V0

Construir un agente capaz de recibir un mensaje de un cliente sobre un pedido y decidir, por sí mismo (vía tool/function calling), qué acción tomar — sin reglas tipo `if "cancelar" in message`. El ciclo que se busca entender:

```text
User
 ↓
LLM understands request
 ↓
LLM selects tool
 ↓
Application executes tool
 ↓
Tool returns result
 ↓
LLM interprets result
 ↓
Response to user
```

Principio central: **el LLM nunca es la fuente de verdad**. Si una tool devuelve `success: false`, el agente jamás debe afirmar que la acción tuvo éxito.

## Arquitectura actual

```text
customer-agent/
│
├── app/
│   ├── main.py           # FastAPI — POST /chat
│   ├── agent.py           # loop de tool calling (LLM ↔ tools)
│   ├── models.py           # schemas Pydantic (HTTP + argumentos de tools)
│   ├── config.py           # carga de OPENAI_API_KEY / cliente OpenAI
│   │
│   └── tools/
│       └── orders.py       # lógica de negocio + pedidos en memoria
│
├── tests/
│   ├── conftest.py          # fixture: resetea el estado en memoria entre tests
│   ├── test_tools.py         # unit tests de orders.py (sin LLM)
│   ├── test_agent.py          # tests del loop con el LLM mockeado
│   └── test_integration.py     # test real contra OpenAI (aparte, opcional)
│
├── pytest.ini
├── requirements.txt
└── .env.example
```

- `main.py` no tiene lógica propia: recibe el mensaje, se lo pasa al agente, devuelve la respuesta.
- `agent.py` orquesta el loop LLM → tool → LLM. Nunca decide reglas de negocio.
- `tools/orders.py` contiene toda la lógica de negocio (qué se puede cancelar, validación de fechas, etc.) y el "estado" (un dict en memoria — no hay base de datos todavía).
- `models.py` valida tanto el contrato HTTP (`ChatRequest`/`ChatResponse`) como los argumentos que el LLM manda para cada tool, antes de que lleguen a `orders.py`.

## Flujo de una request

```text
POST /chat {"message": "Cancelame el pedido 123"}
        ↓
main.py llama a agent.handle_message(message)
        ↓
1ª llamada al LLM, con las 3 tools disponibles
        ↓
LLM responde: "quiero llamar a cancel_order(order_id=123)"
        ↓
agent.py valida los argumentos con Pydantic
        ↓
se ejecuta tools/orders.cancel_order(123) (lógica de negocio real)
        ↓
la tool devuelve {"success": true/false, ...}
        ↓
2ª llamada al LLM, con ese resultado real en el historial
        ↓
LLM redacta la respuesta final en base al resultado (no lo inventa)
        ↓
{"response": "Listo, cancelé tu pedido 123."}
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configurar la API key

```bash
cp .env.example .env
```

Editá `.env` y completá tu `OPENAI_API_KEY` (y opcionalmente `OPENAI_MODEL`, default `gpt-4o-mini`). El archivo `.env` está ignorado por git — nunca se commitea.

## Cómo ejecutar FastAPI

```bash
uvicorn app.main:app --reload
```

Servidor en `http://127.0.0.1:8000`. Docs interactivas (Swagger) en `http://127.0.0.1:8000/docs`.

## Cómo ejecutar los tests

Los tests unitarios (`test_tools.py` + `test_agent.py`) no gastan tokens ni dependen de OpenAI — el LLM está mockeado en `test_agent.py`. Son los que corren por default:

```bash
pytest
```

El test de integración (`test_integration.py`) llama a la API real de OpenAI y está excluido del run normal (ver `pytest.ini`). Se corre a propósito, y requiere `OPENAI_API_KEY` real:

```bash
pytest -m integration
```

## Ejemplos de requests

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Dónde está mi pedido 123?"}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancelame el pedido 123"}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quiero cambiar la entrega del pedido 123 al 21 de agosto de 2026"}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancelame el pedido 999"}'
# el pedido no existe -> el agente lo informa, nunca inventa una cancelación
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancelame el pedido 456"}'
# el pedido 456 ya fue enviado -> no puede cancelarse, el agente explica el motivo
```

## Limitaciones actuales

- Pedidos en memoria (`dict` de Python): se pierden al reiniciar el proceso. No hay base de datos.
- Sin autenticación ni multi-tenant: cualquiera que le pegue al endpoint puede operar sobre cualquier pedido.
- Sin estado de conversación: cada request a `/chat` es independiente: el agente no recuerda mensajes previos.
- Sin manejo de "falta información" (ej. "quiero cancelar mi pedido" sin decir cuál) — eso es la V1.
- Sin frontend: se prueba vía `curl`/Swagger.
- Un solo agente, tres tools fijas: nada de multi-agente, RAG, ni orquestación más compleja.
- Sin métricas de calidad (tool selection accuracy, hallucination rate, etc.) — eso llega con los evals de la V3.

## Futuras versiones resumen

1. **V1 — Conversaciones reales.** Estado de conversación y manejo de información faltante (`get_customer_orders(customer_id)`, el agente pregunta cuando falta un dato).
2. **V2 — LangGraph.** El flujo se convierte en una máquina de estados explícita (`understand_request → retrieve_information → decide_action → execute_tool → verify_result`).
3. **V3 — Simulator + evals.** 50-100 casos de prueba con `expected_tool`/`expected_arguments`, midiendo Tool Selection Accuracy, Tool Argument Accuracy, Task Success Rate y Hallucination Rate.
4. **V4 — Customer Operations más complejo.** Nuevas tools (`get_payment`, `refund_payment`, `create_return`, `replace_product`, `track_delivery`, `reschedule_delivery`, `check_inventory`) y casos con más de un problema a la vez.
5. **V5 — RAG / políticas.** El agente consulta políticas reales (devolución, reembolso, cancelación, envío) en vez de decidir todo por su cuenta.
6. **V6 — Human-in-the-loop.** Acciones por debajo de un umbral se ejecutan solas; por encima, requieren aprobación humana y el workflow queda suspendido.
7. **V7 — Multi-agent.** Un orquestador coordinando agentes especializados (Orders, Payments, Delivery), comparado contra el enfoque single-agent con evals.
8. **V8 — Distribución.** FastAPI → cola de mensajes (RabbitMQ) → workers de agentes → Redis/Postgres, para simular carga concurrente real.
9. **V9 — Producción.** Docker, AWS, OpenTelemetry, dashboards, load testing, retries, idempotencia, rate limits y control de costos.

## Futuras versiones 

2. **V1 — Conversaciones reales**
   Agregás casos donde falta información:

   ```text
   User: "Quiero cancelar mi pedido."
   Agent: "¿Cuál de tus pedidos?"
   ```

   El agente consulta:

   ```python
   get_customer_orders(customer_id)
   ```

   y mantiene estado de conversación.

3. **V2 — LangGraph**
   Recién acá metería LangGraph.

   Convertís el flujo en una máquina de estados:

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

   Acá vas a entender realmente para qué sirve LangGraph.

4. **V3 — Simulator + evals**
   Antes de escalar, hacé que el agente sea medible.

   Creás 50-100 casos:

   ```json
   {
     "user": "Cancelá mi pedido 123",
     "expected_tool": "cancel_order",
     "expected_arguments": {
       "order_id": 123
     }
   }
   ```

   Y calculás:

   ```text
   Tool Selection Accuracy
   Tool Argument Accuracy
   Task Success Rate
   Hallucination Rate
   ```

   Para mí, **esta etapa es fundamental**.

5. **V4 — Customer Operations más complejo**
   Agregás:

   ```python
   get_payment()
   refund_payment()
   create_return()
   replace_product()
   track_delivery()
   reschedule_delivery()
   check_inventory()
   ```

   Y situaciones con más de un problema:

   > “Me cobraron dos veces y todavía no llegó mi pedido.”

6. **V5 — RAG / políticas**
   El agente ya no puede simplemente decidir si hacer un refund.

   Tiene que consultar:

   ```text
   return policy
   refund policy
   cancellation policy
   shipping policy
   ```

   Ahí metés embeddings/vector DB o búsqueda sobre documentos.

7. **V6 — Human-in-the-loop**
   Algunas acciones:

   ```text
   refund $5 → automático

   refund $2000 → requiere aprobación
   ```

   El workflow queda suspendido y puede continuar después.

8. **V7 — Multi-agent**
   Solo cuando ya tengas suficiente complejidad:

   ```text
                Customer Agent
                     │
               Orchestrator
              /      |       \
          Orders  Payments  Delivery
           Agent    Agent     Agent
   ```

   Y comparás:

   ```text
   single-agent vs multi-agent
   ```

   incluso con evals para saber si realmente mejoró.

9. **V8 — Distribución**
   Ahora sí:

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

   Simulás cientos/miles de ejecuciones concurrentes.

10. **V9 — Producción**
    Docker + AWS + OpenTelemetry + dashboards + load testing + retries + idempotency + rate limits + costos.