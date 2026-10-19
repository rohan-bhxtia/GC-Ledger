import sqlite3
import stat
from zoneinfo import ZoneInfo

import pytest

from gcl.db import Database
from gcl.models import (
    Commitment, CommitmentStatus, Event, EventStatus, Expense, ExpenseStatus, Settlement,
    SettlementStatus, SplitType,
)
from gcl.parser import parse_chat_text

IST = ZoneInfo("Asia/Kolkata")

CHAT = """\
14/10/2026, 9:00 pm - Aman: Dinner ₹3,600 kal?
14/10/2026, 9:05 pm - Riya: Haan
14/10/2026, 10:30 pm - Aman: Paid ₹3,600 for dinner
15/10/2026, 9:00 am - Aman: Paid ₹3,600 for dinner (reminder)
20/10/2026, 9:00 am - Aman: Paid ₹3,600 for movie
20/10/2026, 9:05 am - Karan: '); DROP TABLE records; --
"""


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def chat(db):
    msgs = parse_chat_text(CHAT, tz=IST).messages
    chat_id = db.get_or_create_chat("Goa Trip")
    db.add_messages(chat_id, msgs)
    return chat_id, [m.msg_id for m in msgs], msgs


def expense(ids, status=ExpenseStatus.CONFIRMED, **kw):
    base = dict(payer="Aman", amount_paise=360_000, description="dinner", participants=["Aman", "Riya"],
                split=SplitType.EQUAL, status=status, source_msg_ids=ids)
    return Expense(**{**base, **kw})


# --- Setup & security -------------------------------------------------------------------

def test_file_is_private_and_constraints_are_on(tmp_path):
    path = tmp_path / "sub" / "gcl.db"
    with Database(path) as database:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert database.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert database.conn.execute("PRAGMA secure_delete").fetchone()[0] == 1


def test_amount_check_constraint(db, chat):
    chat_id, _, _ = chat
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO records(chat_id, kind, status, amount_paise, dedup_key, data, updated_at)"
            " VALUES (?, 'expense', 'confirmed', -5, 'k', '{}', 'now')", (chat_id,))


def test_sql_in_chat_text_is_just_data(db, chat):
    chat_id, ids, _ = chat
    stored = next(m for m in db.get_messages(chat_id) if m.msg_id == ids[5])
    assert stored.text == "'); DROP TABLE records; --"
    assert db.records(chat_id, "expense") == []  # table still exists


# --- Chats & messages ---------------------------------------------------------------------

def test_chat_names_are_unique(db):
    assert db.get_or_create_chat("Goa") == db.get_or_create_chat("  Goa ")
    with pytest.raises(ValueError):
        db.get_or_create_chat("   ")


def test_messages_round_trip_and_reimport_adds_nothing(db, chat):
    chat_id, ids, msgs = chat
    assert db.get_messages(chat_id) == msgs                 # timestamps equal as instants
    assert db.add_messages(chat_id, msgs) == set()          # re-import: nothing new
    assert len(db.get_messages(chat_id)) == len(msgs)


def test_pending_until_marked(db, chat):
    chat_id, ids, _ = chat
    assert db.pending_message_ids(chat_id) == set(ids)
    db.mark_extracted(chat_id, ids[:4])
    assert db.pending_message_ids(chat_id) == set(ids[4:])


# --- Records & dedup -----------------------------------------------------------------------

def test_records_round_trip_with_sources(db, chat):
    chat_id, ids, _ = chat
    db.save_records(chat_id, "expense", [expense([ids[2]])])
    db.save_records(chat_id, "settlement", [Settlement(
        from_person="Riya", to_person="Aman", amount_paise=180_000,
        status=SettlementStatus.CLAIMED, source_msg_ids=[ids[1]])])
    db.save_records(chat_id, "commitment", [Commitment(
        owner="Riya", action="pay", status=CommitmentStatus.PROMISED, source_msg_ids=[ids[1]])])
    db.save_records(chat_id, "event", [Event(
        title="Dinner", status=EventStatus.PROPOSED, source_msg_ids=[ids[0]])])

    e = db.records(chat_id, "expense")[0]
    assert e.record.amount_paise == 360_000 and e.record.source_msg_ids == [ids[2]]
    assert [len(db.records(chat_id, k)) for k in ("settlement", "commitment", "event")] == [1, 1, 1]


def test_overlap_duplicate_is_merged(db, chat):
    chat_id, ids, _ = chat
    db.save_records(chat_id, "expense", [expense([ids[2]])])
    stats = db.save_records(chat_id, "expense", [expense([ids[2], ids[3]])])  # same msg + one more
    assert (stats.inserted, stats.merged) == (0, 1)
    stored = db.records(chat_id, "expense")
    assert len(stored) == 1
    assert stored[0].record.source_msg_ids == [ids[2], ids[3]]


def test_newer_evidence_updates_status_older_does_not(db, chat):
    chat_id, ids, _ = chat
    db.save_records(chat_id, "expense", [expense([ids[0]], status=ExpenseStatus.PROPOSED)])
    db.save_records(chat_id, "expense", [expense([ids[0], ids[2]], status=ExpenseStatus.CONFIRMED)])
    assert db.records(chat_id, "expense")[0].record.status is ExpenseStatus.CONFIRMED

    db.save_records(chat_id, "expense", [expense([ids[0]], status=ExpenseStatus.DISPUTED)])
    assert db.records(chat_id, "expense")[0].record.status is ExpenseStatus.CONFIRMED


def test_same_amount_without_shared_source_is_kept_separately(db, chat):
    chat_id, ids, _ = chat
    db.save_records(chat_id, "expense", [expense([ids[2]])])
    second = db.save_records(chat_id, "expense", [expense([ids[3]])])  # same payer and amount
    assert (second.inserted, second.merged) == (1, 0)
    assert len(db.records(chat_id, "expense")) == 2


# --- Delete -----------------------------------------------------------------------------------

def test_delete_chat_removes_everything_and_only_that_chat(db, chat):
    chat_id, ids, msgs = chat
    db.save_records(chat_id, "expense", [expense([ids[2]])])
    other = db.get_or_create_chat("Family")
    db.add_messages(other, msgs)

    assert db.delete_chat(chat_id) is True
    for table in ("records", "record_sources"):
        assert db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)).fetchone()[0] == 0
    assert len(db.get_messages(other)) == len(msgs)
    assert [c.name for c in db.list_chats()] == ["Family"]
    assert db.delete_chat(chat_id) is False


def test_deleted_text_is_not_left_in_the_file(tmp_path):
    path = tmp_path / "gcl.db"
    secret = "SECRET-VILLA-CODE-4242"
    with Database(path) as database:
        chat_id = database.get_or_create_chat("x")
        database.add_messages(chat_id, parse_chat_text(f"14/10/2026, 9:00 pm - A: {secret}\n", tz=IST).messages)
        database.delete_chat(chat_id)
    assert secret.encode() not in path.read_bytes()