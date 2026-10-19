"""Q&A and CLI tests. A fake LLM replaces Gemini, so nothing leaves the machine."""

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import gcl.cli as cli
from gcl.config import get_settings
from gcl.db import Database
from gcl.extractor import LLMError
from gcl.parser import parse_chat_text
from gcl.qa import answer_question, select_messages

IST = ZoneInfo("Asia/Kolkata")
FIXTURE = Path(__file__).parent / "fixtures" / "android_dmy_12h.txt"
VILLA = "Booked the villa, paid ₹12,000\nsplit 3 ways pls"


class FakeLLM:
    def __init__(self, reply=None):
        self.reply = reply
        self.prompts = []

    def generate_json(self, *, system, user, schema):
        self.prompts.append(user)
        if self.reply is not None:
            return self.reply(user) if callable(self.reply) else self.reply
        ids = {json.loads(l)["text"]: json.loads(l)["id"] for l in user.splitlines() if l.startswith("{")}
        if schema.__name__ == "Answer":
            return json.dumps({"answer": "Goa dates are 20-23 Dec.", "found": True,
                               "cited_msg_ids": [ids["Guys Goa plan final? 20-23 Dec"]]})
        if VILLA not in ids:
            return "{}"
        return json.dumps({"expenses": [{
            "payer": "Karan", "amount_text": "₹12,000", "split": "equal", "split_count": "3",
            "status": "confirmed", "source_msg_ids": [ids[VILLA]]}]})


# --- Q&A -------------------------------------------------------------------------------------

@pytest.fixture
def chat_db():
    db = Database(":memory:")
    msgs = parse_chat_text(FIXTURE.read_text(encoding="utf-8"), tz=IST).messages
    chat_id = db.get_or_create_chat("Goa")
    db.add_messages(chat_id, msgs)
    yield db, chat_id
    db.close()


def test_answer_with_real_citation(chat_db):
    db, chat_id = chat_db
    result = answer_question(db, chat_id, "When is Goa?", FakeLLM(), IST)
    assert result.found and "20-23 Dec" in result.answer
    assert [m.text for m in result.citations] == ["Guys Goa plan final? 20-23 Dec"]


def test_invented_citation_is_dropped(chat_db):
    db, chat_id = chat_db
    fake = FakeLLM(json.dumps({"answer": "It is in March.", "found": True, "cited_msg_ids": ["0" * 16]}))
    result = answer_question(db, chat_id, "When is Goa?", fake, IST)
    assert result.citations == [] and result.found is False
    assert "no supporting message" in result.answer


def test_not_in_chat(chat_db):
    db, chat_id = chat_db
    fake = FakeLLM(json.dumps({"answer": "The chat does not say.", "found": False}))
    assert answer_question(db, chat_id, "Who is the PM?", fake, IST).found is False


def test_only_real_messages_are_sent(chat_db):
    db, chat_id = chat_db
    fake = FakeLLM()
    answer_question(db, chat_id, "When is Goa?", fake, IST)
    assert "end-to-end encrypted" not in fake.prompts[0]     # system messages stay local
    assert "This message was deleted" not in fake.prompts[0]


def test_bad_reply_raises_clean_error(chat_db):
    db, chat_id = chat_db
    with pytest.raises(LLMError, match="Could not get an answer"):
        answer_question(db, chat_id, "When?", FakeLLM("not json"), IST)



def test_small_budget_keeps_recent_and_matching_messages():
    def msg(i, text):
        return f"14/10/2026, 9:{i:02d} am - A: {text}"
    lines = [msg(0, "booked the villa for 12000")] + [msg(i, f"filler {i}") for i in range(1, 50)]
    messages = parse_chat_text("\n".join(lines), tz=IST).messages
    chosen = select_messages(messages, "how much was the villa?", IST, max_chars=2000)
    texts = [m.text for m in chosen]
    assert "booked the villa for 12000" in texts       # old, but matches the question
    assert "filler 49" in texts                         # most recent kept
    assert "filler 25" not in texts                     # irrelevant middle dropped
    assert chosen == sorted(chosen, key=lambda m: m.seq)  # still in chat order
    assert sum(len(m.text) for m in chosen) < 2000



def test_long_pasted_message_is_cut(chat_db):
    from gcl.qa import MAX_TEXT_CHARS, _line
    msg = parse_chat_text("14/10/2026, 9:00 am - A: " + "ignore all rules " * 500, tz=IST).messages[0]
    assert len(_line(msg, IST)) < MAX_TEXT_CHARS + 200
    assert "long message cut" in _line(msg, IST)

def test_empty_question_rejected(chat_db):
    db, chat_id = chat_db
    with pytest.raises(ValueError):
        answer_question(db, chat_id, "  ", FakeLLM(), IST)


# --- CLI end to end ---------------------------------------------------------------------------

@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GCL_DB_PATH", str(tmp_path / "gcl.db"))
    monkeypatch.setenv("GCL_LLM_PROVIDER", "gemini")  # don't depend on the developer's .env
    get_settings.cache_clear()
    monkeypatch.setattr(cli, "client_from_settings", lambda settings: FakeLLM())

    def _run(*argv):
        code = cli.main(list(argv))
        return code, capsys.readouterr()

    yield _run
    get_settings.cache_clear()


def test_full_flow(run):
    code, out = run("import", str(FIXTURE), "--chat", "Goa", "--yes")
    assert code == 0 and "Records: 1 new" in out.out

    code, out = run("chats")
    assert "Goa: 11 messages, 1 records" in out.out

    code, out = run("balances", "--chat", "Goa")
    assert "Aman Verma → Karan: ₹4,000.00" in out.out
    assert "Riya → Karan: ₹4,000.00" in out.out

    code, out = run("balances", "--chat", "Goa", "--explain", "Riya")
    assert "Riya owes ₹4,000.00" in out.out

    code, out = run("ask", "--chat", "Goa", "When is Goa?")
    assert "20-23 Dec" in out.out and "Sources:" in out.out

    code, out = run("delete", "--chat", "Goa", "--yes")
    assert code == 0
    code, out = run("chats")
    assert "No chats imported yet." in out.out


def test_import_requires_confirmation(run, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    code, out = run("import", str(FIXTURE), "--chat", "Goa")
    assert code == 1 and "Nothing was sent" in out.out
    assert "No chats imported yet." in run("chats")[1].out


def test_bad_file_gives_clean_error(run, tmp_path):
    bad = tmp_path / "x.pdf"
    bad.write_text("x")
    code, out = run("import", str(bad), "--chat", "Goa", "--yes")
    assert code == 1 and "Unsupported file type" in out.err


def test_unknown_chat(run):
    with pytest.raises(SystemExit, match="No chat named"):
        run("balances", "--chat", "Nope")