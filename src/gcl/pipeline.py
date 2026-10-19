"""import_chat(): the whole import, from export file to stored records.

  1. Parse the file.
  2. Store the messages (messages we already have are kept as they are).
  3. Find the windows that still contain messages the LLM hasn't read ("pending").
  4. Send those windows to the LLM, 10 at a time. After each group, save the records and
     mark those messages as read. So if the run stops halfway, nothing finished is lost.

Running it again on the same (or a longer) export is safe:
  - messages already read are never sent again,
  - duplicate records are merged by the database,
  - messages from a failed window stay pending and are tried again next time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from gcl.config import Settings, get_settings
from gcl.db import Database
from gcl.extractor import LLMClient, build_windows, extract
from gcl.parser import parse_chat_file


@dataclass
class ImportReport:
    chat_id: int
    messages_total: int
    messages_new: int
    windows_sent: int = 0
    failed_windows: list[int] = field(default_factory=list)
    records_new: int = 0
    records_merged: int = 0
    rejected: int = 0
    quota_exhausted: bool = False


def import_chat(
    path: str | Path,
    *,
    chat_name: str,
    db: Database,
    client: LLMClient,
    settings: Settings | None = None,
    batch_windows: int = 10,
    progress: Callable[[int, int], None] | None = None,
) -> ImportReport:
    """Import a chat export. `progress(done, total)` is called after each group of windows."""
    settings = settings or get_settings()

    # 1. Parse first: a bad file raises an error before anything is written.
    parsed = parse_chat_file(path, settings=settings)

    # 2. Store the messages.
    chat_id = db.get_or_create_chat(chat_name)
    new_ids = db.add_messages(chat_id, parsed.messages)
    report = ImportReport(
        chat_id=chat_id,
        messages_total=len(parsed.messages),
        messages_new=len(new_ids),
    )

    # 3. Which windows still have unread messages?
    pending = db.pending_message_ids(chat_id)
    windows_to_send = []
    for window in build_windows(parsed.messages):
        if any(m.msg_id in pending for m in window.messages):
            windows_to_send.append(window)

    # 4. Send them in groups, saving after each group.
    total = len(windows_to_send)
    for start in range(0, total, batch_windows):
        group = windows_to_send[start:start + batch_windows]
        group_ids = {m.msg_id for window in group for m in window.messages}

        result = extract(parsed, client, tz=settings.tz, windows=group)

        for kind, records in [
            ("expense", result.expenses),
            ("settlement", result.settlements),
            ("commitment", result.commitments),
            ("event", result.events),
        ]:
            stats = db.save_records(chat_id, kind, records)
            report.records_new += stats.inserted
            report.records_merged += stats.merged

        # Done = this group's messages, except those in a failed window (tried again later).
        db.mark_extracted(chat_id, group_ids - result.failed_ids)

        report.windows_sent += result.windows_sent
        report.failed_windows += result.failed_windows
        report.rejected += len(result.rejected)

        if progress:
            progress(min(start + batch_windows, total), total)

        if result.quota_exhausted:
            # Stop here. Everything not sent yet stays pending for next time.
            report.quota_exhausted = True
            for window in windows_to_send[start + batch_windows:]:
                report.failed_windows.append(window.index)
            break

    # System, deleted and caption-less media messages are never sent to the LLM.
    # There's nothing to extract from them, so mark them as done too.
    in_some_window = {m.msg_id for window in windows_to_send for m in window.messages}
    db.mark_extracted(chat_id, pending - in_some_window)

    return report
