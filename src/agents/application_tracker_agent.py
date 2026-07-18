"""
src/agents/application_tracker_agent.py
---------------------------------------
Tracks application records and follow-up actions in PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.context_helpers import compact_job_summary, resolve_target_job
from src.db.db import (
    list_application_records,
    update_application_record_status,
    upsert_application_record,
)


class ApplicationTrackerAgent(BaseAgent):
    name = "application_tracker"

    def run(self, context: dict[str, Any]) -> AgentResult:
        action = str(context.get("application_action", "list")).strip().lower()

        if action in {"track", "upsert", "save"}:
            return self._track(context)
        if action in {"update", "update_status"}:
            return self._update(context)
        if action == "list":
            return self._list(context)

        raise ValueError(
            "application_action must be one of: list, track, upsert, save, update, update_status."
        )

    def _track(self, context: dict[str, Any]) -> AgentResult:
        email = str(context.get("email") or "").strip().lower()
        if not email:
            raise ValueError("application tracking requires context['email'].")

        target_job = resolve_target_job(context)
        if not target_job:
            raise ValueError(
                "application tracking requires resolvable target_job/target_job_url/job_keyword."
            )

        status = str(context.get("application_status", "saved")).strip().lower() or "saved"
        follow_up_due_at = _parse_optional_datetime(context.get("follow_up_due_at"))
        notes = str(context.get("application_notes") or "").strip()

        record_id = upsert_application_record(
            email=email,
            job_url=str(target_job.get("url") or ""),
            company=str(target_job.get("company") or ""),
            job_title=str(target_job.get("job_title") or ""),
            status=status,
            follow_up_due_at=follow_up_due_at,
            notes=notes,
        )

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=f"Tracked application record #{record_id} for {email}.",
            data={
                "application_record_id": record_id,
                "application_record": {
                    "email": email,
                    "status": status,
                    "follow_up_due_at": follow_up_due_at,
                    "notes": notes,
                    "target_job": compact_job_summary(target_job),
                },
            },
            next_actions=[
                "Update status to interviewing/offer/rejected as process moves.",
                "Set follow_up_due_at for pending recruiter responses.",
            ],
        )

    def _update(self, context: dict[str, Any]) -> AgentResult:
        record_id_raw = context.get("application_record_id")
        if record_id_raw is None:
            raise ValueError("update_status requires context['application_record_id'].")
        record_id = int(record_id_raw)
        status = str(context.get("application_status") or "").strip().lower()
        if not status:
            raise ValueError("update_status requires context['application_status'].")

        notes = str(context.get("application_notes") or "").strip()
        follow_up_due_at = _parse_optional_datetime(context.get("follow_up_due_at"))
        updated = update_application_record_status(
            record_id=record_id,
            status=status,
            notes=notes,
            follow_up_due_at=follow_up_due_at,
        )

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=f"Updated {updated} application record(s) for record_id={record_id}.",
            data={"updated_records": updated},
            next_actions=["List records to confirm next follow-up queue."],
        )

    def _list(self, context: dict[str, Any]) -> AgentResult:
        email = str(context.get("email") or "").strip().lower() or None
        status = str(context.get("application_status") or "").strip().lower() or None
        limit = max(1, int(context.get("application_limit", 50)))
        records = list_application_records(email=email, status=status, limit=limit)

        return AgentResult(
            agent=self.name,
            status="completed",
            summary=f"Fetched {len(records)} tracked application record(s).",
            data={"application_records": records},
            next_actions=["Use record_id + update_status to progress each application stage."],
        )


def _parse_optional_datetime(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    datetime.fromisoformat(raw)
    return raw
