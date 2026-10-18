"""import_chat: parse -> store messages -> extract pending windows -> store records.

Re-importing the same (or a longer) export is safe:
- known messages are not re-sent to the LLM,
- duplicate records are merged in the database,
- messages from a failed window stay "pending" and are retried on the next import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gcl.config import Settings, get_settings
from gcl.db import Database
from gcl.extractor import LLMClient, extract
from gcl.parser import DateOrder, parse_chat_file


@dataclass
class ImportReport:
    chat_id: int
    messages_total: int
    messages_new: int
    windows_sent: int = 0
    failed_windows: list[int] = field(default_factory=list)
    records_new: int = 0
    records_merged: int = 0
    possible_duplicates: int = 0
    rejected: int = 0
    rejection_reasons: list[str] = field(default_factory=list)


def import_chat(
    path: str | Path,
    *,
    chat_name: str,
    db: Database,
    client: LLMClient,
    settings: Settings | None = None,
    date_order: DateOrder = "auto",
) -> ImportReport:
    settings = settings or get_settings()
    parsed = parse_chat_file(path, settings=settings, date_order=date_order)  # raises before any write

    chat_id = db.get_or_create_chat(chat_name)
    new_ids = db.add_messages(chat_id, parsed.messages)
    report = ImportReport(chat_id=chat_id, messages_total=len(parsed.messages), messages_new=len(new_ids))

    pending = db.pending_message_ids(chat_id)
    if pending:
        result = extract(parsed, client, tz=settings.tz, pending_ids=pending)
        for kind, records in (
            ("expense", result.expenses), ("settlement", result.settlements),
            ("commitment", result.commitments), ("event", result.events),
        ):
            stats = db.save_records(chat_id, kind, records)
            report.records_new += stats.inserted
            report.records_merged += stats.merged
            report.possible_duplicates += stats.possible_duplicates
        # Done = everything pending except messages whose window failed (retried next import).
        # Includes system/deleted/bare-media messages, which are never sent to the LLM.
        db.mark_extracted(chat_id, pending - result.failed_ids)
        report.windows_sent = result.windows_sent
        report.failed_windows = result.failed_windows
        report.rejected = len(result.rejected)
        report.rejection_reasons = [f"{r.kind}: {r.reason}" for r in result.rejected]

    db.record_import(
        chat_id,
        messages_total=report.messages_total, messages_new=report.messages_new,
        windows_sent=report.windows_sent, windows_failed=len(report.failed_windows),
        records_new=report.records_new, records_merged=report.records_merged,
        rejected=report.rejected,
    )
    return report