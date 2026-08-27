"""Optional Kafka event bus used to wake background workers.

Kafka is treated as the dispatch layer. PostgreSQL remains the source of truth
for queue state, retries, and idempotency.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: list[str]
    client_id: str
    security_protocol: str
    enabled: bool


def _split_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def load_kafka_config() -> KafkaConfig:
    bootstrap_servers = _split_csv(os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""))
    enabled = bool(bootstrap_servers)
    return KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id=os.environ.get("KAFKA_CLIENT_ID", "jobfinder").strip() or "jobfinder",
        security_protocol=os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").strip() or "PLAINTEXT",
        enabled=enabled,
    )


def kafka_enabled() -> bool:
    return load_kafka_config().enabled


def publish_json(topic: str, payload: dict[str, Any], *, key: str | None = None) -> bool:
    """Publish a JSON payload to Kafka if Kafka is configured.

    Returns True when the message was published and False when Kafka is disabled.
    """
    config = load_kafka_config()
    if not config.enabled:
        return False

    try:
        from kafka import KafkaProducer
    except Exception as exc:  # pragma: no cover - import guard for optional dependency
        raise RuntimeError("kafka-python is not installed.") from exc

    producer = KafkaProducer(
        bootstrap_servers=config.bootstrap_servers,
        client_id=config.client_id,
        security_protocol=config.security_protocol,
        value_serializer=lambda value: json.dumps(value, default=str).encode("utf-8"),
        key_serializer=lambda value: value.encode("utf-8") if value is not None else None,
        linger_ms=10,
        retries=3,
    )
    try:
        producer.send(topic, value=payload, key=key)
        producer.flush(timeout=10)
        return True
    finally:
        producer.close(timeout=10)
