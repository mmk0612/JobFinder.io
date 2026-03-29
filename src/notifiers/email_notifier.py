"""
src/notifiers/email_notifier.py
-------------------------------
Simple SMTP email sender for digest notifications.
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

import requests


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _parse_recipients(raw: str) -> list[str]:
    recipients = [value.strip() for value in raw.split(",") if value.strip()]
    if not recipients:
        raise EnvironmentError("NOTIFY_EMAIL_TO must contain at least one recipient.")
    return recipients


def _send_via_smtp(*, subject: str, body: str, sender: str, recipients: list[str]) -> None:
    smtp_host = os.environ.get("NOTIFY_EMAIL_SMTP_HOST", "").strip()
    smtp_port = int(os.environ.get("NOTIFY_EMAIL_SMTP_PORT", "587") or "587")
    smtp_user = os.environ.get("NOTIFY_EMAIL_SMTP_USER", "").strip()
    smtp_password = os.environ.get("NOTIFY_EMAIL_SMTP_PASSWORD", "").strip()
    use_ssl = _env_bool("NOTIFY_EMAIL_SMTP_SSL", False)

    if not smtp_host:
        raise EnvironmentError("NOTIFY_EMAIL_SMTP_HOST is not set.")

    use_tls = _env_bool("NOTIFY_EMAIL_USE_TLS", True)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    if use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            if smtp_user:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        if use_tls:
            server.starttls()
        if smtp_user:
            server.login(smtp_user, smtp_password)
        server.send_message(msg)


def _send_via_mailgun(*, subject: str, body: str, sender: str, recipients: list[str]) -> None:
    domain = os.environ.get("NOTIFY_MAILGUN_DOMAIN", "").strip()
    api_key = os.environ.get("NOTIFY_MAILGUN_API_KEY", "").strip()
    region = os.environ.get("NOTIFY_MAILGUN_REGION", "us").strip().lower() or "us"

    if not domain:
        raise EnvironmentError("NOTIFY_MAILGUN_DOMAIN is not set.")
    if not api_key:
        raise EnvironmentError("NOTIFY_MAILGUN_API_KEY is not set.")

    base_url = "https://api.mailgun.net"
    if region == "eu":
        base_url = "https://api.eu.mailgun.net"

    url = f"{base_url}/v3/{domain}/messages"
    data = {
        "from": sender,
        "to": recipients,
        "subject": subject,
        "text": body,
    }

    response = requests.post(
        url,
        auth=("api", api_key),
        data=data,
        timeout=30,
    )
    response.raise_for_status()


def send_email(*, subject: str, body: str) -> None:
    """Send plain-text email using configured provider (`smtp` or `mailgun`)."""
    provider = os.environ.get("NOTIFY_EMAIL_PROVIDER", "smtp").strip().lower() or "smtp"
    smtp_user = os.environ.get("NOTIFY_EMAIL_SMTP_USER", "").strip()
    sender = os.environ.get("NOTIFY_EMAIL_FROM", "").strip() or smtp_user
    recipients_raw = os.environ.get("NOTIFY_EMAIL_TO", "").strip()

    if not sender:
        raise EnvironmentError("NOTIFY_EMAIL_FROM (or NOTIFY_EMAIL_SMTP_USER) is not set.")
    if not recipients_raw:
        raise EnvironmentError("NOTIFY_EMAIL_TO is not set.")

    recipients = _parse_recipients(recipients_raw)

    if provider == "smtp":
        _send_via_smtp(subject=subject, body=body, sender=sender, recipients=recipients)
        return

    if provider == "mailgun":
        _send_via_mailgun(subject=subject, body=body, sender=sender, recipients=recipients)
        return

    raise ValueError("NOTIFY_EMAIL_PROVIDER must be one of: smtp, mailgun")
