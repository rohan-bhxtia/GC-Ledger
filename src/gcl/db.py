"""Store chats, messages and records in a SQLite database file (data/gcl.db).

Tables:
  chats           one row per chat ("Goa Trip")
  messages        every parsed message. `extracted` = 1 once the LLM has read it.
  records         every checked expense / settlement / commitment / event
  record_sources  which messages each record came from (used for citations)

Safety:
  - Every query uses "?" placeholders, so chat text can never change a query.
  - Deleting a chat deletes everything that came from it (ON DELETE CASCADE).
  - `secure_delete` overwrites deleted data, so old chat text doesn't stay hidden in the file.
  - The file is readable only by you (permissions 0600).
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generic, Iterable, Literal, TypeVar

from pydantic import BaseModel

from gcl.models import Commitment, Event, Expense, Message, MessageKind, Settlement

RecordKind = Literal["expense", "settlement", "commitment", "event"]

# Which model class each record kind is stored as.
MODEL_FOR_KIND: dict[str, type[BaseModel]] = {
    "expense": Expense,
    "settlement": Settlement,
    "commitment": Commitment,
    "event": Event,
}

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
    ts_utc      TEXT NOT NULL,              -- time in UTC, so text sorting = time sorting
    sender      TEXT,
    text        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('text','media','deleted','system')),
    edited      INTEGER NOT NULL,
    extracted   INTEGER NOT NULL DEFAULT 0, -- 1 once the LLM has read this message
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
    dedup_key     TEXT NOT NULL,            -- used to spot the same record twice
    data          TEXT NOT NULL,            -- the whole record as JSON (without its sources)
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
"""

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Stored(Generic[T]):
    """A record read back from the database, together with its row id."""

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
    inserted: int = 0   # new records
    merged: int = 0     # duplicates folded into an existing record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _placeholders(count: int) -> str:
    """'?,?,?' for an IN (...) list of `count` values."""
    return ",".join("?" * count)


def _dedup_key(record: BaseModel) -> str:
    """Records with the same key (and a shared source message) are the same record."""
    if isinstance(record, Expense):
        return f"{record.payer}|{record.amount_paise}"
    if isinstance(record, Settlement):
        return f"{record.from_person}|{record.to_person}|{record.amount_paise}"
    if isinstance(record, Commitment):
        return record.owner
    if isinstance(record, Event):
        return str(record.on_date)
    raise TypeError(type(record).__name__)


def _main_person(record: BaseModel) -> str | None:
    """The payer, sender or owner of a record (whichever it has)."""
    for attribute in ("payer", "from_person", "owner"):
        value = getattr(record, attribute, None)
        if value:
            return value
    return None


class Database:
    """Open with `with Database(path) as db:` so it is always closed."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        in_memory = self.path == ":memory:"  # used by the tests

        if not in_memory:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row             # rows can be read like dicts
        self.conn.execute("PRAGMA foreign_keys = ON")   # needed for ON DELETE CASCADE
        self.conn.execute("PRAGMA secure_delete = ON")  # overwrite deleted data
        self.conn.executescript(SCHEMA)

        if not in_memory:
            os.chmod(self.path, 0o600)  # only you can read or write the file

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Chats
    # -------------------------------------------------------------------------

    def get_or_create_chat(self, name: str) -> int:
        """Return the chat's id, creating the chat if it's new."""
        name = name.strip()
        if not name:
            raise ValueError("chat name must not be empty")
        with self.conn:
            self.conn.execute(
                "INSERT INTO chats(name, created_at) VALUES (?, ?) ON CONFLICT(name) DO NOTHING",
                (name, _now()),
            )
        return self.find_chat(name)

    def find_chat(self, name: str) -> int | None:
        row = self.conn.execute("SELECT id FROM chats WHERE name = ?", (name.strip(),)).fetchone()
        if row is None:
            return None
        return row["id"]

    def list_chats(self) -> list[ChatInfo]:
        rows = self.conn.execute(
            """SELECT c.id, c.name,
                      (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS messages,
                      (SELECT COUNT(*) FROM records r WHERE r.chat_id = c.id) AS records
               FROM chats c ORDER BY c.name"""
        ).fetchall()
        return [ChatInfo(row["id"], row["name"], row["messages"], row["records"]) for row in rows]

    def delete_chat(self, chat_id: int) -> bool:
        """Delete a chat and everything that came from it. Returns False if it didn't exist."""
        with self.conn:
            deleted = self.conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount
        if deleted:
            self.conn.execute("VACUUM")  # shrink the file and drop the freed pages
        return bool(deleted)

    # -------------------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------------------

    def add_messages(self, chat_id: int, messages: Iterable[Message]) -> set[str]:
        """Store messages (already-stored ones are kept). Returns the ids that were new."""
        messages = list(messages)
        already_stored = self._message_ids(chat_id)

        rows = []
        for m in messages:
            utc_time = m.timestamp.astimezone(timezone.utc).isoformat()
            rows.append((chat_id, m.msg_id, m.seq, m.line_no, utc_time,
                         m.sender, m.text, m.kind.value, int(m.edited)))

        with self.conn:
            # If a message is already stored, only update its position (a longer export
            # shifts line numbers). Its `extracted` flag is left alone.
            self.conn.executemany(
                """INSERT INTO messages(chat_id, msg_id, seq, line_no, ts_utc, sender, text, kind, edited)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chat_id, msg_id) DO UPDATE SET seq = excluded.seq,
                                                              line_no = excluded.line_no""",
                rows,
            )

        all_ids = {m.msg_id for m in messages}
        return all_ids - already_stored

    def pending_message_ids(self, chat_id: int) -> set[str]:
        """Messages the LLM hasn't read yet."""
        rows = self.conn.execute(
            "SELECT msg_id FROM messages WHERE chat_id = ? AND extracted = 0", (chat_id,)
        )
        return {row["msg_id"] for row in rows}

    def mark_extracted(self, chat_id: int, msg_ids: Iterable[str]) -> None:
        with self.conn:
            self.conn.executemany(
                "UPDATE messages SET extracted = 1 WHERE chat_id = ? AND msg_id = ?",
                [(chat_id, msg_id) for msg_id in msg_ids],
            )

    def get_messages(self, chat_id: int) -> list[Message]:
        """All messages of a chat, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY ts_utc, seq", (chat_id,)
        ).fetchall()

        messages = []
        for row in rows:
            messages.append(Message(
                msg_id=row["msg_id"],
                seq=row["seq"],
                line_no=row["line_no"],
                timestamp=datetime.fromisoformat(row["ts_utc"]),
                sender=row["sender"],
                text=row["text"],
                kind=MessageKind(row["kind"]),
                edited=bool(row["edited"]),
            ))
        return messages

    def _message_ids(self, chat_id: int) -> set[str]:
        rows = self.conn.execute("SELECT msg_id FROM messages WHERE chat_id = ?", (chat_id,))
        return {row["msg_id"] for row in rows}

    # -------------------------------------------------------------------------
    # Records
    # -------------------------------------------------------------------------

    def save_records(self, chat_id: int, kind: RecordKind, records: Iterable[BaseModel]) -> SaveStats:
        """Store records, merging duplicates.

        A duplicate has the same key (e.g. same payer and amount) AND shares at least one
        source message with a stored record. This happens when two overlapping windows find
        the same expense. The two are merged into one: the sources are combined, and the
        content is replaced only if the new copy is backed by a later message
        (e.g. first "proposed", later "paid").
        """
        stats = SaveStats()

        with self.conn:
            for record in records:
                key = _dedup_key(record)
                sources = list(dict.fromkeys(record.source_msg_ids))  # no repeats

                existing_id = self._find_duplicate(chat_id, kind, key, sources)
                if existing_id is not None:
                    self._merge(chat_id, existing_id, record, sources)
                    stats.merged += 1
                    continue

                cursor = self.conn.execute(
                    """INSERT INTO records(chat_id, kind, status, person, amount_paise,
                                           dedup_key, data, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chat_id,
                        kind,
                        record.status.value,
                        _main_person(record),
                        getattr(record, "amount_paise", None),
                        key,
                        record.model_dump_json(exclude={"source_msg_ids"}),
                        _now(),
                    ),
                )
                self._add_sources(cursor.lastrowid, chat_id, sources)
                stats.inserted += 1

        return stats

    def records(self, chat_id: int, kind: RecordKind) -> list[Stored]:
        """All records of one kind, in the order they were saved."""
        model = MODEL_FOR_KIND[kind]
        rows = self.conn.execute(
            "SELECT id, data FROM records WHERE chat_id = ? AND kind = ? ORDER BY id",
            (chat_id, kind),
        ).fetchall()

        result = []
        for row in rows:
            data = json.loads(row["data"])
            data["source_msg_ids"] = self._sources(row["id"])
            record = model.model_validate(data)  # checked again when read back
            result.append(Stored(row["id"], record))
        return result

    # -------------------------------------------------------------------------
    # Helpers for save_records
    # -------------------------------------------------------------------------

    def _find_duplicate(self, chat_id: int, kind: str, key: str, sources: list[str]) -> int | None:
        """The id of a stored record with the same key that shares a source message."""
        row = self.conn.execute(
            f"""SELECT r.id FROM records r
                JOIN record_sources s ON s.record_id = r.id
                WHERE r.chat_id = ? AND r.kind = ? AND r.dedup_key = ?
                  AND s.msg_id IN ({_placeholders(len(sources))})
                ORDER BY r.id LIMIT 1""",
            [chat_id, kind, key, *sources],
        ).fetchone()
        if row is None:
            return None
        return row["id"]

    def _merge(self, chat_id: int, record_id: int, record: BaseModel, sources: list[str]) -> None:
        """Fold a duplicate into an existing record."""
        old_latest = self._latest_time(chat_id, self._sources(record_id))
        new_latest = self._latest_time(chat_id, sources)

        self._add_sources(record_id, chat_id, sources)

        # The copy backed by the newer message wins (e.g. "proposed" -> "confirmed").
        if new_latest > old_latest:
            self.conn.execute(
                "UPDATE records SET status = ?, data = ?, updated_at = ? WHERE id = ?",
                (record.status.value, record.model_dump_json(exclude={"source_msg_ids"}),
                 _now(), record_id),
            )

    def _add_sources(self, record_id: int, chat_id: int, sources: list[str]) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO record_sources(record_id, chat_id, msg_id) VALUES (?, ?, ?)",
            [(record_id, chat_id, msg_id) for msg_id in sources],
        )

    def _sources(self, record_id: int) -> list[str]:
        """A record's source message ids, oldest message first."""
        rows = self.conn.execute(
            """SELECT s.msg_id FROM record_sources s
               JOIN messages m ON m.chat_id = s.chat_id AND m.msg_id = s.msg_id
               WHERE s.record_id = ? ORDER BY m.ts_utc, m.seq""",
            (record_id,),
        )
        return [row["msg_id"] for row in rows]

    def _latest_time(self, chat_id: int, msg_ids: list[str]) -> datetime:
        """The time of the newest message in the list."""
        row = self.conn.execute(
            f"""SELECT MAX(ts_utc) AS latest FROM messages
                WHERE chat_id = ? AND msg_id IN ({_placeholders(len(msg_ids))})""",
            [chat_id, *msg_ids],
        ).fetchone()
        if row["latest"] is None:
            raise ValueError("record cites messages that are not stored")
        return datetime.fromisoformat(row["latest"])
