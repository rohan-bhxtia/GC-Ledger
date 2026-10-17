import logging
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from gcl.config import Settings
from gcl.models import MessageKind
from gcl.parser import ParseError, parse_chat_file, parse_chat_text

FIXTURES = Path(__file__).parent / "fixtures"
IST = ZoneInfo("Asia/Kolkata")


def parse(text: str, **kwargs):
    return parse_chat_text(text, tz=IST, **kwargs)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, max_import_mb=1)


# --- Android fixture -------------------------------------------------------------

@pytest.fixture
def android(settings):
    return parse_chat_file(FIXTURES / "android_dmy_12h.txt", settings=settings)


def test_android_counts(android):
    s = android.stats
    assert (s.messages, s.text_messages, s.media_messages, s.deleted_messages, s.system_messages) == (
        11, 5, 1, 1, 4,
    )
    assert android.date_order == "DMY"
    assert android.participants == ["Aman Verma", "Karan", "Riya"]


def test_android_timestamps_are_ist(android):
    first_text = next(m for m in android.messages if m.kind is MessageKind.TEXT)
    assert first_text.timestamp == datetime(2026, 10, 14, 21, 2, tzinfo=IST)
    assert first_text.timestamp.utcoffset().total_seconds() == 5.5 * 3600


def test_multiline_message_joined(android):
    villa = next(m for m in android.messages if "villa" in m.text)
    assert villa.text == "Booked the villa, paid ₹12,000\nsplit 3 ways pls"
    assert villa.sender == "Karan"
    assert villa.line_no == 6


def test_edited_marker_stripped(android):
    cab = next(m for m in android.messages if m.text.startswith("Cab"))
    assert cab.edited is True
    assert cab.text == "Cab kaun book karega?"


def test_subject_change_with_colon_is_system(android):
    subject = next(m for m in android.messages if "changed the subject" in m.text)
    assert subject.kind is MessageKind.SYSTEM
    assert subject.sender is None


# --- iOS -----------------------------------------------------------------------------

def test_ios_fixture(settings):
    result = parse_chat_file(FIXTURES / "ios_dmy.txt", settings=settings)
    assert result.stats.messages == 5
    assert result.messages[0].timestamp == datetime(2026, 10, 14, 21, 2, 11, tzinfo=IST)
    assert result.messages[-1].text.endswith("\nto finalise the budget")


def test_ios_invisible_marks_and_group_system_lines():
    text = (
        "‎[14/10/26, 9:00:00 PM] Goa Trip: ‎Aman created this group\n"
        "[14/10/26, 9:00:05 PM] Goa Trip: ‎Messages and calls are end-to-end encrypted.\n"
        "[14/10/26, 9:01:00 PM] Aman: ‎image omitted\n"
        "[14/10/26, 9:02:00 PM] ‪+91 98765 43210‬: hello\n"
        "[14/10/26, 9:03:00 PM] Riya: ‎This message was deleted.\n"
    )
    kinds = [m.kind for m in parse(text).messages]
    assert kinds == [
        MessageKind.SYSTEM, MessageKind.SYSTEM, MessageKind.MEDIA,
        MessageKind.TEXT, MessageKind.DELETED,
    ]
    result = parse(text)
    assert result.participants == ["+91 98765 43210", "Aman", "Riya"]  # group name excluded


# --- Formats and dates -----------------------------------------------------------------

def test_24_hour_and_narrow_space():
    result = parse(
        "14/10/2026, 21:15 - Aman: x\n"
        "14/10/2026, 9:16 pm - Riya: y\n"
    )
    assert [m.timestamp.hour for m in result.messages] == [21, 21]


def test_midnight_and_noon():
    result = parse("14/10/2026, 12:05 am - A: x\n14/10/2026, 12:05 pm - A: y\n")
    assert [m.timestamp.hour for m in result.messages] == [0, 12]


def test_month_first_detected():
    result = parse("10/14/26, 9:00 PM - Aman: hi\n10/15/26, 9:01 PM - Riya: yo\n")
    assert result.date_order == "MDY"
    assert result.messages[0].timestamp.date() == datetime(2026, 10, 14).date()


def test_ambiguous_dates_default_day_first_and_override():
    text = "05/06/2026, 10:00 - A: x\n"
    assert parse(text).messages[0].timestamp.month == 6
    assert parse(text, date_order="MDY").messages[0].timestamp.month == 5


def test_inconsistent_date_order_rejected():
    with pytest.raises(ParseError, match="Inconsistent"):
        parse("14/10/2026, 10:00 - A: x\n10/14/2026, 10:00 - A: y\n")


def test_media_with_caption_keeps_caption():
    result = parse("14/10/2026, 9:00 pm - Karan: IMG-20261014-WA0001.jpg (file attached)\nVilla pics\n")
    msg = result.messages[0]
    assert msg.kind is MessageKind.MEDIA
    assert "Villa pics" in msg.text


# --- Malformed input ----------------------------------------------------------------------

def test_malformed_lines_reported_not_crashing(settings):
    result = parse_chat_file(FIXTURES / "malformed.txt", settings=settings)
    assert [m.text for m in result.messages] == ["valid message", "another valid one"]
    assert result.stats.orphan_lines == 2
    assert result.stats.invalid_header_lines == [4, 5]


@pytest.mark.parametrize("text", ["", "   \n\n", "hello world\njust some text\n"])
def test_not_a_chat_rejected(text):
    with pytest.raises(ParseError):
        parse(text)


# --- Message ids -----------------------------------------------------------------------------

def test_ids_stable_and_unique(settings):
    a = parse_chat_file(FIXTURES / "ios_dmy.txt", settings=settings)
    b = parse_chat_file(FIXTURES / "ios_dmy.txt", settings=settings)
    ids = [m.msg_id for m in a.messages]
    assert ids == [m.msg_id for m in b.messages]
    assert len(set(ids)) == len(ids)  # identical duplicate lines still get distinct ids


def test_ids_unchanged_when_chat_grows():
    base = "14/10/2026, 9:00 pm - A: one\n14/10/2026, 9:01 pm - B: two\n"
    longer = base + "14/10/2026, 9:02 pm - A: three\n"
    assert [m.msg_id for m in parse(base).messages] == [m.msg_id for m in parse(longer).messages][:2]


# --- File validation -------------------------------------------------------------------------

def test_wrong_extension(tmp_path, settings):
    f = tmp_path / "chat.pdf"
    f.write_text("x")
    with pytest.raises(ParseError, match="Unsupported"):
        parse_chat_file(f, settings=settings)


def test_missing_file(tmp_path, settings):
    with pytest.raises(ParseError, match="not found"):
        parse_chat_file(tmp_path / "nope.txt", settings=settings)


def test_empty_file(tmp_path, settings):
    f = tmp_path / "chat.txt"
    f.write_bytes(b"")
    with pytest.raises(ParseError, match="empty"):
        parse_chat_file(f, settings=settings)


def test_too_large(tmp_path, settings):
    f = tmp_path / "chat.txt"
    f.write_bytes(b"a" * (1024 * 1024 + 1))
    with pytest.raises(ParseError, match="limit"):
        parse_chat_file(f, settings=settings)


@pytest.mark.parametrize("payload", [b"\xff\xfe\xfa\xfb", b"14/10/2026, 9:00 pm - A: x\x00\x00"])
def test_binary_rejected(tmp_path, settings, payload):
    f = tmp_path / "chat.txt"
    f.write_bytes(payload)
    with pytest.raises(ParseError):
        parse_chat_file(f, settings=settings)


# --- Zip exports ------------------------------------------------------------------------------

def make_zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


def test_zip_export(tmp_path, settings):
    chat = (FIXTURES / "ios_dmy.txt").read_text(encoding="utf-8")
    z = make_zip(tmp_path / "WhatsApp Chat - Goa.zip", {"_chat.txt": chat, "IMG-0001.jpg": "fake"})
    assert parse_chat_file(z, settings=settings).stats.messages == 5


def test_zip_without_chat(tmp_path, settings):
    z = make_zip(tmp_path / "x.zip", {"photo.jpg": "fake"})
    with pytest.raises(ParseError, match="does not contain"):
        parse_chat_file(z, settings=settings)


def test_zip_with_several_chats(tmp_path, settings):
    z = make_zip(tmp_path / "x.zip", {"a.txt": "x", "b.txt": "y"})
    with pytest.raises(ParseError, match="several"):
        parse_chat_file(z, settings=settings)


def test_zip_bomb_guard(tmp_path, settings):
    z = make_zip(tmp_path / "x.zip", {"_chat.txt": "a" * (2 * 1024 * 1024)})
    assert z.stat().st_size < settings.max_import_bytes  # small on disk...
    with pytest.raises(ParseError, match="size limit"):    # ...but huge uncompressed
        parse_chat_file(z, settings=settings)


def test_corrupt_zip(tmp_path, settings):
    f = tmp_path / "x.zip"
    f.write_bytes(b"not a zip")
    with pytest.raises(ParseError, match="zip"):
        parse_chat_file(f, settings=settings)


# --- Privacy ------------------------------------------------------------------------------------

def test_logs_contain_no_chat_content(caplog, settings):
    with caplog.at_level(logging.DEBUG, logger="gcl"):
        parse_chat_file(FIXTURES / "android_dmy_12h.txt", settings=settings)
    assert "Parsed 11 messages" in caplog.text
    for secret in ("villa", "12,000", "Karan", "speaker"):
        assert secret not in caplog.text