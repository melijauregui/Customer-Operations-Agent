# Customer-Operations-Agent

1. **V0 — Un agente + 3 tools**

   * Python
   * FastAPI
   * OpenAI API
   * Pydantic
   * Tools ficticias:

     ```python
     get_order(order_id)
     cancel_order(order_id)
     change_delivery_date(order_id, date)
     ```
   * Casos:

     ```text
     "¿Dónde está mi pedido 123?"
     "Cancelame el pedido 123"
     "Pasá la entrega del pedido 123 para el lunes"
     ```
   * Objetivo: entender bien **tool calling, structured outputs y control del LLM**.

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
