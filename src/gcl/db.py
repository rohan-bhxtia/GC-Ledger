"""SQLite storage: chats, messages, extracted records and their source citations.

Design:
- `messages` keeps the parsed chat (needed for citations and Q&A).
- `records` holds every validated record. Key fields (kind, status, person, amount) are real
  columns with CHECK constraints; the full validated model is kept as JSON in `data`.
- `record_sources` links each record to the exact messages that support it.
- Deleting a chat cascades to everything derived from it; `secure_delete` overwrites the
  freed pages so deleted chat text does not linger inside the database file.

All SQL is parameterised: chat text never becomes part of a query.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generic, Iterable, Literal, TypeVar

from pydantic import BaseModel

from gcl.models import Commitment, Event, Expense, Message, MessageKind, Settlement

RecordKind = Literal["expense", "settlement", "commitment", "event"]
_MODELS: dict[str, type[BaseModel]] = {
    "expense": Expense, "settlement": Settlement, "commitment": Commitment, "event": Event,
}
POSSIBLE_DUPLICATE_WINDOW = timedelta(days=3)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    chat_id     INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    msg_id      TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    line_no     INTEGER NOT NULL,
    ts_utc      TEXT NOT NULL,              -- ISO 8601, UTC, sortable
    sender      TEXT,
    text        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('text','media','deleted','system')),
    edited      INTEGER NOT NULL,
    extracted   INTEGER NOT NULL DEFAULT 0, -- 1 once a successful LLM window covered it
    PRIMARY KEY (chat_id, msg_id)
);
CREATE INDEX IF NOT EXISTS ix_messages_time ON messages(chat_id, ts_utc, seq);

CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY,
    chat_id       INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('expense','settlement','commitment','event')),
    status        TEXT NOT NULL,
    person        TEXT,                     -- payer / from_person / owner
    amount_paise  INTEGER CHECK (amount_paise IS NULL OR amount_paise > 0),
    dedup_key     TEXT NOT NULL,
    data          TEXT NOT NULL,            -- validated model as JSON (without sources)
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_records_dedup ON records(chat_id, kind, dedup_key);

CREATE TABLE IF NOT EXISTS record_sources (
    record_id   INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    chat_id     INTEGER NOT NULL,
    msg_id      TEXT NOT NULL,
    PRIMARY KEY (record_id, msg_id),
    FOREIGN KEY (chat_id, msg_id) REFERENCES messages(chat_id, msg_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS imports (
    id              INTEGER PRIMARY KEY,
    chat_id         INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    imported_at     TEXT NOT NULL,
    messages_total  INTEGER NOT NULL,
    messages_new    INTEGER NOT NULL,
    windows_sent    INTEGER NOT NULL,
    windows_failed  INTEGER NOT NULL,
    records_new     INTEGER NOT NULL,
    records_merged  INTEGER NOT NULL,
    rejected        INTEGER NOT NULL
);
"""

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Stored(Generic[T]):
    id: int
    record: T


@dataclass(frozen=True)
class ChatInfo:
    id: int
    name: str
    messages: int
    records: int


@dataclass
class SaveStats:
    inserted: int = 0
    merged: int = 0
    possible_duplicates: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dedup_key(kind: str, record: BaseModel) -> str:
    if isinstance(record, Expense):
        return f"{record.payer}|{record.amount_paise}"
    if isinstance(record, Settlement):
        return f"{record.from_person}|{record.to_person}|{record.amount_paise}"
    if isinstance(record, Commitment):
        return record.owner
    if isinstance(record, Event):
        return str(record.on_date)
    raise TypeError(kind)


def _person(record: BaseModel) -> str | None:
    return getattr(record, "payer", None) or getattr(record, "from_person", None) or getattr(record, "owner", None)


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA secure_delete = ON")
        self.conn.executescript(SCHEMA)
        if self.path != ":memory:":
            os.chmod(self.path, 0o600)  # chat data: owner read/write only

    # --- lifecycle ------------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- chats ----------------------------------------------------------------------------

    def get_or_create_chat(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("chat name must not be empty")
        with self.conn:
            self.conn.execute(
                "INSERT INTO chats(name, created_at) VALUES (?, ?) ON CONFLICT(name) DO NOTHING",
                (name, _now()),
            )
        return self.conn.execute("SELECT id FROM chats WHERE name = ?", (name,)).fetchone()["id"]

    def find_chat(self, name: str) -> int | None:
        row = self.conn.execute("SELECT id FROM chats WHERE name = ?", (name.strip(),)).fetchone()
        return row["id"] if row else None

    def list_chats(self) -> list[ChatInfo]:
        rows = self.conn.execute(
            """SELECT c.id, c.name,
                      (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS messages,
                      (SELECT COUNT(*) FROM records r WHERE r.chat_id = c.id) AS records
               FROM chats c ORDER BY c.name"""
        ).fetchall()
        return [ChatInfo(r["id"], r["name"], r["messages"], r["records"]) for r in rows]

    def delete_chat(self, chat_id: int) -> bool:
        """Remove a chat and everything derived from it (messages, records, sources, imports)."""
        with self.conn:
            deleted = self.conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount
        if deleted:
            self.conn.execute("VACUUM")  # shrink the file; secure_delete already zeroed pages
        return bool(deleted)

    # --- messages -------------------------------------------------------------------------

    def add_messages(self, chat_id: int, messages: Iterable[Message]) -> set[str]:
        """Upsert messages. Returns the ids that were not already stored."""
        messages = list(messages)
        existing = self._message_ids(chat_id)
        with self.conn:
            self.conn.executemany(
                """INSERT INTO messages(chat_id, msg_id, seq, line_no, ts_utc, sender, text, kind, edited)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chat_id, msg_id) DO UPDATE SET seq = excluded.seq,
                                                              line_no = excluded.line_no""",
                [
                    (chat_id, m.msg_id, m.seq, m.line_no,
                     m.timestamp.astimezone(timezone.utc).isoformat(), m.sender, m.text,
                     m.kind.value, int(m.edited))
                    for m in messages
                ],
            )
        return {m.msg_id for m in messages} - existing

    def pending_message_ids(self, chat_id: int) -> set[str]:
        rows = self.conn.execute(
            "SELECT msg_id FROM messages WHERE chat_id = ? AND extracted = 0", (chat_id,)
        )
        return {r["msg_id"] for r in rows}

    def mark_extracted(self, chat_id: int, msg_ids: Iterable[str]) -> None:
        with self.conn:
            self.conn.executemany(
                "UPDATE messages SET extracted = 1 WHERE chat_id = ? AND msg_id = ?",
                [(chat_id, i) for i in msg_ids],
            )

    def get_messages(self, chat_id: int, msg_ids: Iterable[str] | None = None) -> list[Message]:
        sql = "SELECT * FROM messages WHERE chat_id = ?"
        params: list = [chat_id]
        if msg_ids is not None:
            ids = list(msg_ids)
            sql += f" AND msg_id IN ({','.join('?' * len(ids))})" if ids else " AND 0"
            params += ids
        rows = self.conn.execute(sql + " ORDER BY ts_utc, seq", params).fetchall()
        return [
            Message(
                msg_id=r["msg_id"], seq=r["seq"], line_no=r["line_no"],
                timestamp=datetime.fromisoformat(r["ts_utc"]), sender=r["sender"],
                text=r["text"], kind=MessageKind(r["kind"]), edited=bool(r["edited"]),
            )
            for r in rows
        ]

    def _message_ids(self, chat_id: int) -> set[str]:
        rows = self.conn.execute("SELECT msg_id FROM messages WHERE chat_id = ?", (chat_id,))
        return {r["msg_id"] for r in rows}

    # --- records --------------------------------------------------------------------------

    def save_records(self, chat_id: int, kind: RecordKind, records: Iterable[BaseModel]) -> SaveStats:
        """Insert records, merging duplicates.

        Duplicate = same chat, kind and key fields (e.g. payer + amount) AND at least one
        shared source message. A merge unions the sources; the record's content is replaced
        only if the new version is backed by a later message (it saw more of the conversation).
        Same key but no shared source within 3 days -> kept, flagged "possible_duplicate".
        """
        stats = SaveStats()
        with self.conn:
            for record in records:
                key = _dedup_key(kind, record)
                sources = list(dict.fromkeys(record.source_msg_ids))
                match = self._find_duplicate(chat_id, kind, key, sources)
                if match is not None:
                    self._merge(chat_id, match, record, sources)
                    stats.merged += 1
                    continue
                if kind in ("expense", "settlement") and self._near_same_key(chat_id, kind, key, sources):
                    record = record.model_copy(update={"flags": [*record.flags, "possible_duplicate"]})
                    stats.possible_duplicates += 1
                cur = self.conn.execute(
                    """INSERT INTO records(chat_id, kind, status, person, amount_paise, dedup_key, data, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (chat_id, kind, record.status.value, _person(record),
                     getattr(record, "amount_paise", None), key,
                     record.model_dump_json(exclude={"source_msg_ids"}), _now()),
                )
                self._add_sources(cur.lastrowid, chat_id, sources)
                stats.inserted += 1
        return stats

    def records(self, chat_id: int, kind: RecordKind) -> list[Stored]:
        model = _MODELS[kind]
        rows = self.conn.execute(
            "SELECT id, data FROM records WHERE chat_id = ? AND kind = ? ORDER BY id", (chat_id, kind)
        ).fetchall()
        result = []
        for row in rows:
            data = json.loads(row["data"])
            data["source_msg_ids"] = self._sources(row["id"])
            result.append(Stored(row["id"], model.model_validate(data)))  # re-validated on read
        return result

    def record_import(self, chat_id: int, **counts: int) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO imports(chat_id, imported_at, messages_total, messages_new, windows_sent,
                                       windows_failed, records_new, records_merged, rejected)
                   VALUES (:chat_id, :at, :messages_total, :messages_new, :windows_sent,
                           :windows_failed, :records_new, :records_merged, :rejected)""",
                {"chat_id": chat_id, "at": _now(), **counts},
            )

    # --- internals ------------------------------------------------------------------------

    def _find_duplicate(self, chat_id: int, kind: str, key: str, sources: list[str]) -> int | None:
        row = self.conn.execute(
            f"""SELECT r.id FROM records r JOIN record_sources s ON s.record_id = r.id
                WHERE r.chat_id = ? AND r.kind = ? AND r.dedup_key = ?
                  AND s.msg_id IN ({','.join('?' * len(sources))})
                ORDER BY r.id LIMIT 1""",
            [chat_id, kind, key, *sources],
        ).fetchone()
        return row["id"] if row else None

    def _near_same_key(self, chat_id: int, kind: str, key: str, sources: list[str]) -> bool:
        first, _ = self._time_span(chat_id, sources)
        rows = self.conn.execute(
            "SELECT id FROM records WHERE chat_id = ? AND kind = ? AND dedup_key = ?", (chat_id, kind, key)
        ).fetchall()
        for row in rows:
            _, last = self._time_span(chat_id, self._sources(row["id"]))
            if abs(first - last) <= POSSIBLE_DUPLICATE_WINDOW:
                return True
        return False

    def _merge(self, chat_id: int, record_id: int, record: BaseModel, sources: list[str]) -> None:
        old_sources = self._sources(record_id)
        _, old_last = self._time_span(chat_id, old_sources)
        _, new_last = self._time_span(chat_id, sources)
        self._add_sources(record_id, chat_id, sources)
        if new_last > old_last:  # newer evidence wins (e.g. proposed -> confirmed)
            self.conn.execute(
                "UPDATE records SET status = ?, data = ?, updated_at = ? WHERE id = ?",
                (record.status.value, record.model_dump_json(exclude={"source_msg_ids"}), _now(), record_id),
            )

    def _add_sources(self, record_id: int, chat_id: int, sources: list[str]) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO record_sources(record_id, chat_id, msg_id) VALUES (?, ?, ?)",
            [(record_id, chat_id, s) for s in sources],
        )

    def _sources(self, record_id: int) -> list[str]:
        rows = self.conn.execute(
            """SELECT s.msg_id FROM record_sources s
               JOIN messages m ON m.chat_id = s.chat_id AND m.msg_id = s.msg_id
               WHERE s.record_id = ? ORDER BY m.ts_utc, m.seq""",
            (record_id,),
        )
        return [r["msg_id"] for r in rows]

    def _time_span(self, chat_id: int, msg_ids: list[str]) -> tuple[datetime, datetime]:
        row = self.conn.execute(
            f"""SELECT MIN(ts_utc) AS first, MAX(ts_utc) AS last FROM messages
                WHERE chat_id = ? AND msg_id IN ({','.join('?' * len(msg_ids))})""",
            [chat_id, *msg_ids],
        ).fetchone()
        if row["first"] is None:
            raise ValueError("record cites messages that are not stored")
        return datetime.fromisoformat(row["first"]), datetime.fromisoformat(row["last"])