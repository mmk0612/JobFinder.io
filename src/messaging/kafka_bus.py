"""Optional Kafka event bus used to wake background workers.

Kafka is treated as the dispatch layer. PostgreSQL remains the source of truth
for queue state, retries, and idempotency.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


# ── Canonical Topic Names ───────────────────────────────────────────────────
TOPIC_RESUME_ANALYSIS_REQUESTED  = "resume-analysis-requested"
TOPIC_RESUME_ANALYSIS_COMPLETED  = "resume-analysis-completed"
TOPIC_JOB_SCRAPE_REQUESTED        = "job-scrape-requested"
TOPIC_JOB_SCRAPE_COMPLETED        = "job-scrape-completed"
TOPIC_JOB_PROCESSING_REQUESTED    = "job-processing-requested"
TOPIC_JOB_PROCESSING_COMPLETED    = "job-processing-completed"
TOPIC_JOB_MATCHING_REQUESTED      = "job-matching-requested"
TOPIC_JOB_MATCHING_COMPLETED      = "job-matching-completed"
TOPIC_RECOMMENDATION_REQUESTED    = "recommendation-requested"
TOPIC_RECOMMENDATION_COMPLETED    = "recommendation-completed"


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: list[str]
    client_id: str
    security_protocol: str
    sasl_mechanism: str | None
    sasl_username: str | None
    sasl_password: str | None
    enabled: bool


def _split_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def load_kafka_config() -> KafkaConfig:
    bootstrap_servers = _split_csv(os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""))
    enabled = bool(bootstrap_servers)
    security_protocol = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").strip() or "PLAINTEXT"
    sasl_mechanism = os.environ.get("KAFKA_SASL_MECHANISM", "").strip() or None
    sasl_password = os.environ.get("KAFKA_SASL_PASSWORD", "").strip() or None
    sasl_username = os.environ.get("KAFKA_SASL_USERNAME", "").strip() or None

    if security_protocol.upper() == "SASL_SSL":
        sasl_mechanism = sasl_mechanism or "PLAIN"
        sasl_username = sasl_username or "$ConnectionString"

    if sasl_password and not sasl_username:
        sasl_username = "$ConnectionString"

    return KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id=os.environ.get("KAFKA_CLIENT_ID", "jobfinder").strip() or "jobfinder",
        security_protocol=security_protocol,
        sasl_mechanism=sasl_mechanism,
        sasl_username=sasl_username,
        sasl_password=sasl_password,
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
        sasl_mechanism=config.sasl_mechanism,
        sasl_plain_username=config.sasl_username,
        sasl_plain_password=config.sasl_password,
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
