# Event-Driven Architecture and Transactional Outbox Pattern

Event-driven architecture decouples producers from consumers using asynchronous pub/sub messaging channels.

## Transactional Outbox Pattern

When a service modifies relational database state and publishes domain events, dual-write anomalies can occur if one operation succeeds and the other fails.
The Transactional Outbox pattern guarantees at-least-once message delivery:
1. The business mutation and the outgoing event are written to the database inside the same local ACID transaction.
2. The event is stored in a dedicated `outbox` table.
3. An asynchronous message relay (or Change Data Capture tool like Debezium reading the database WAL) polls the `outbox` table and publishes the messages to the broker (Apache Kafka or RabbitMQ).
4. Upon broker acknowledgment, the outbox record is marked published or deleted.

## Idempotent Consumer Pattern

Because at-least-once delivery may result in duplicate event delivery, consumers must be idempotent:
- Consumers track processed event IDs in an idempotency cache or database table.
- Duplicate events are acknowledged without repeating side effects.
