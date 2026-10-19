import json
from pathlib import Path

import pytest

from gcl.config import Settings
from gcl.db import Database
from gcl.extractor import LLMError
from gcl.parser import ParseError
from gcl.pipeline import import_chat

FIXTURE = Path(__file__).parent / "fixtures" / "android_dmy_12h.txt"
VILLA = "Booked the villa, paid ₹12,000\nsplit 3 ways pls"


class FakeLLM:
    """Returns the villa expense whenever that message is in the window; can fail on demand."""

    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self.fail_first = fail_first

    def generate_json(self, *, system, user, schema):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise LLMError("simulated outage")
        ids = {json.loads(l)["text"]: json.loads(l)["id"] for l in user.splitlines() if l.startswith("{")}
        if VILLA not in ids:
            return "{}"
        return json.dumps({"expenses": [{
            "payer": "Karan", "amount_text": "₹12,000", "split": "equal", "split_count": "3",
            "status": "confirmed", "source_msg_ids": [ids[VILLA]],
        }]})


@pytest.fixture
def settings():
    return Settings(_env_file=None)


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


def test_first_import(db, settings):
    llm = FakeLLM()
    report = import_chat(FIXTURE, chat_name="Goa", db=db, client=llm, settings=settings)
    assert (report.messages_total, report.messages_new) == (11, 11)
    assert report.windows_sent == 3 and llm.calls == 3
    assert report.records_new == 1 and report.failed_windows == []
    expense = db.records(report.chat_id, "expense")[0].record
    assert expense.participants == ["Aman Verma", "Karan", "Riya"]  # Option A in action
    assert db.pending_message_ids(report.chat_id) == set()


def test_reimport_same_file_calls_llm_zero_times(db, settings):
    import_chat(FIXTURE, chat_name="Goa", db=db, client=FakeLLM(), settings=settings)
    llm = FakeLLM()
    report = import_chat(FIXTURE, chat_name="Goa", db=db, client=llm, settings=settings)
    assert (report.messages_new, report.windows_sent, llm.calls) == (0, 0, 0)
    assert len(db.records(report.chat_id, "expense")) == 1


def test_longer_export_only_sends_new_windows(db, settings, tmp_path):
    import_chat(FIXTURE, chat_name="Goa", db=db, client=FakeLLM(), settings=settings)
    longer = tmp_path / "chat.txt"
    longer.write_text(FIXTURE.read_text(encoding="utf-8")
                      + "20/10/2026, 8:00 pm - Riya: Sent Karan 4000\n", encoding="utf-8")
    llm = FakeLLM()
    report = import_chat(longer, chat_name="Goa", db=db, client=llm, settings=settings)
    assert report.messages_new == 1
    assert llm.calls == 1   # only the new window, not the 3 old ones


def test_failed_window_is_retried_on_next_import(db, settings):
    first = import_chat(FIXTURE, chat_name="Goa", db=db, client=FakeLLM(fail_first=1), settings=settings)
    assert first.failed_windows == [0] and first.records_new == 0
    assert db.pending_message_ids(first.chat_id)            # villa window still pending

    llm = FakeLLM()
    second = import_chat(FIXTURE, chat_name="Goa", db=db, client=llm, settings=settings)
    assert llm.calls == 1 and second.records_new == 1
    assert db.pending_message_ids(second.chat_id) == set()


def test_bad_file_writes_nothing(db, settings, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("just some notes\n", encoding="utf-8")
    with pytest.raises(ParseError):
        import_chat(bad, chat_name="Goa", db=db, client=FakeLLM(), settings=settings)
    assert db.list_chats() == []


def test_import_report_counts(db, settings):
    report = import_chat(FIXTURE, chat_name="Goa", db=db, client=FakeLLM(), settings=settings)
    assert (report.messages_new, report.windows_sent, report.records_new) == (11, 3, 1)

def test_interrupted_run_keeps_finished_batches(db, settings):
    """Villa is in window 0; stop (Ctrl+C) during window 1: batch 0 is already saved."""
    class StopAfterFirst(FakeLLM):
        def generate_json(self, **kwargs):
            if self.calls == 1:
                raise KeyboardInterrupt
            return super().generate_json(**kwargs)

    seen = []
    with pytest.raises(KeyboardInterrupt):
        import_chat(FIXTURE, chat_name="Goa", db=db, client=StopAfterFirst(), settings=settings,
                    batch_windows=1, progress=lambda done, total: seen.append((done, total)))
    chat_id = db.find_chat("Goa")
    assert len(db.records(chat_id, "expense")) == 1         # villa saved before the stop
    assert seen == [(1, 3)]

    llm = FakeLLM()
    import_chat(FIXTURE, chat_name="Goa", db=db, client=llm, settings=settings)
    assert llm.calls == 2                                    # only the 2 unfinished windows
    assert db.pending_message_ids(chat_id) == set()
