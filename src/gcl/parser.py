"""Read a WhatsApp export file and turn it into a list of Message objects.

WhatsApp writes one line per message, in one of two formats:

    Android:  14/10/2026, 9:15 pm - Riya: text
    iOS:      [14/10/26, 9:15:30 PM] Riya: text

A message can continue over several lines; those extra lines have no date at the start.

The input can be the exported .txt file or the exported .zip. A zip is read in memory and
never unpacked onto the disk.

Privacy: this module only logs counts, never message text.
"""

from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from gcl.config import Settings, get_settings
from gcl.models import Message, MessageKind, ParseResult

logger = logging.getLogger(__name__)


class ParseError(ValueError):
    """The file is not a usable WhatsApp export."""


# -----------------------------------------------------------------------------
# Patterns
# -----------------------------------------------------------------------------

# A date like 14/10/2026, 14.10.26 or 10-14-2026
DATE = r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
# A time like 9:15, 21:15:30 or 9:15 pm
TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]\.?\s?m\.?)?"

# The start of a message line. "rest" is "Name: text" (or a system line).
ANDROID_LINE = re.compile(rf"^(?P<date>{DATE}),?\s(?P<time>{TIME})\s[-–]\s(?P<rest>.*)$", re.I)
IOS_LINE = re.compile(rf"^\[(?P<date>{DATE}),?\s(?P<time>{TIME})\]\s(?P<rest>.*)$", re.I)

# WhatsApp hides invisible "text direction" characters around names and numbers.
INVISIBLE_CHARS = re.compile("[‎‏‪-‮⁦-⁩﻿]")
# Unusual spaces WhatsApp uses (narrow and non-breaking); we turn them into normal spaces.
ODD_SPACES = str.maketrans({" ": " ", " ": " "})

# Placeholder text for photos, videos, documents and so on.
MEDIA_TEXT = re.compile(
    r"<media omitted>"
    r"|(?:image|video|audio|sticker|gif|document|contact card) omitted"
    r"|<attached: [^>]+>"
    r"|.+ \(file attached\)"
    r"|.* document omitted",
    re.I,
)
DELETED_TEXT = re.compile(r"(?:this message was deleted|you deleted this message)\.?", re.I)
EDITED_MARK = re.compile(r"\s*<this message was edited>\s*$", re.I)
ENCRYPTION_NOTICE = "messages and calls are end-to-end encrypted"

MAX_SENDER_LENGTH = 60


# -----------------------------------------------------------------------------
# Step 1: group the file's lines into messages
# -----------------------------------------------------------------------------

def _split_into_entries(text: str) -> list[dict]:
    """Return one dict per message: its line number, date, time, first line and extra lines."""
    entries: list[dict] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.translate(ODD_SPACES)
        start = line.lstrip("‎‏﻿")
        match = IOS_LINE.match(start) or ANDROID_LINE.match(start)

        if match:
            # A new message starts on this line.
            entries.append({
                "line_no": line_no,
                "date": match["date"],
                "time": match["time"],
                "first_line": match["rest"],
                "extra_lines": [],
            })
        elif entries:
            # No date at the start: this line continues the previous message.
            entries[-1]["extra_lines"].append(line)
        # Lines before the first message (rare) are ignored.

    return entries


# -----------------------------------------------------------------------------
# Step 2: dates and times
# -----------------------------------------------------------------------------

def _date_numbers(date_text: str) -> tuple[int, int, int]:
    """'14/10/2026' -> (14, 10, 2026). We don't know yet which is the day and which the month."""
    first, second, year = re.split(r"[/.\-]", date_text)
    return int(first), int(second), int(year)


def _detect_date_order(entries: list[dict]) -> str:
    """Decide whether dates are day-first ("DMY", 14/10) or month-first ("MDY", 10/14).

    A number above 12 can only be a day. If we never see one, we assume day-first,
    which is the normal format in India.
    """
    first_is_sometimes_day = False
    second_is_sometimes_day = False
    for entry in entries:
        first, second, _ = _date_numbers(entry["date"])
        if first > 12:
            first_is_sometimes_day = True
        if second > 12:
            second_is_sometimes_day = True

    if first_is_sometimes_day and second_is_sometimes_day:
        raise ParseError("Inconsistent date format: both DD/MM and MM/DD dates found.")
    if second_is_sometimes_day:
        return "MDY"
    return "DMY"


def _to_datetime(date_text: str, time_text: str, order: str, tz: ZoneInfo) -> datetime | None:
    """Combine the date and time text into a datetime. Returns None if it's impossible (31/02)."""
    first, second, year = _date_numbers(date_text)
    if order == "DMY":
        day, month = first, second
    else:
        day, month = second, first
    if year < 100:
        year += 2000  # "26" means 2026

    # Clean the time: "9:15 p.m." -> "9:15pm"
    clean_time = time_text.lower().replace(".", "").replace(" ", "")
    am_or_pm = None
    if clean_time.endswith("am") or clean_time.endswith("pm"):
        am_or_pm = clean_time[-2:]
        clean_time = clean_time[:-2]

    parts = [int(p) for p in clean_time.split(":")]
    hour = parts[0]
    minute = parts[1]
    second_of_minute = parts[2] if len(parts) == 3 else 0

    if am_or_pm:
        if hour < 1 or hour > 12:
            return None
        # 12 am is 0:00, 12 pm is 12:00, 1 pm is 13:00
        hour = hour % 12
        if am_or_pm == "pm":
            hour += 12

    try:
        return datetime(year, month, day, hour, minute, second_of_minute, tzinfo=tz)
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# Step 3: who sent it, and what kind of message is it?
# -----------------------------------------------------------------------------

def _looks_like_a_name(name: str) -> bool:
    if len(name) == 0 or len(name) > MAX_SENDER_LENGTH:
        return False
    for quote in '"“”':
        if quote in name:
            return False
    return True


def _sender_text_and_kind(first_line: str) -> tuple[str | None, str, MessageKind]:
    """Split 'Name: text' and decide the message kind."""
    sender, separator, body = first_line.partition(": ")
    sender = INVISIBLE_CHARS.sub("", sender).strip()

    # Android system lines have no "Name: " part, e.g. "Aman added Riya".
    if not separator or not _looks_like_a_name(sender):
        return None, INVISIBLE_CHARS.sub("", first_line).strip(), MessageKind.SYSTEM

    # iOS marks system and media text with an invisible character at the start.
    marked_by_ios = body.startswith("‎")
    clean = INVISIBLE_CHARS.sub("", body).strip()

    if MEDIA_TEXT.fullmatch(clean):
        return sender, clean, MessageKind.MEDIA
    if DELETED_TEXT.fullmatch(clean):
        return sender, clean, MessageKind.DELETED
    if marked_by_ios or clean.lower().startswith(ENCRYPTION_NOTICE):
        # iOS system messages show the group name as the "sender".
        return None, clean, MessageKind.SYSTEM
    return sender, clean, MessageKind.TEXT


def _make_message_id(timestamp: datetime, sender: str | None, text: str, seen: dict[str, int]) -> str:
    """A short id that stays the same every time the same chat is imported.

    It is a hash of time + sender + text. If the very same message appears twice
    (same time, sender and text), the second one gets a different id.
    """
    key = f"{timestamp.isoformat()}\x1f{sender or ''}\x1f{text}"
    times_seen_before = seen.get(key, 0)
    seen[key] = times_seen_before + 1
    return hashlib.sha256(f"{key}\x1f{times_seen_before}".encode()).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def parse_chat_text(text: str, *, tz: ZoneInfo) -> ParseResult:
    """Parse the text of an export into messages."""
    if not text.strip():
        raise ParseError("Chat export is empty.")
    if "\x00" in text:
        raise ParseError("File does not look like a text chat export.")

    entries = _split_into_entries(text)
    if not entries:
        raise ParseError("No WhatsApp messages found. Is this an exported chat?")

    order = _detect_date_order(entries)
    messages: list[Message] = []
    seen_keys: dict[str, int] = {}

    for entry in entries:
        timestamp = _to_datetime(entry["date"], entry["time"], order, tz)
        if timestamp is None:
            continue  # impossible date or time: skip this message

        sender, first_text, kind = _sender_text_and_kind(entry["first_line"])

        # Join the first line with any continuation lines.
        extra = [INVISIBLE_CHARS.sub("", line).rstrip() for line in entry["extra_lines"]]
        body = "\n".join([first_text] + extra).strip()

        # Remove WhatsApp's "<This message was edited>" mark, but remember it.
        edited = bool(EDITED_MARK.search(body))
        if edited:
            body = EDITED_MARK.sub("", body)

        messages.append(Message(
            msg_id=_make_message_id(timestamp, sender, body, seen_keys),
            seq=len(messages),
            line_no=entry["line_no"],
            timestamp=timestamp,
            sender=sender,
            text=body,
            kind=kind,
            edited=edited,
        ))

    if not messages:
        raise ParseError("No messages with a valid date/time were found.")

    logger.info("Parsed %d messages (date order %s)", len(messages), order)
    return ParseResult(messages=messages, date_order=order)


def parse_chat_file(path: str | Path, *, settings: Settings | None = None) -> ParseResult:
    """Check and parse an exported chat file (.txt or .zip)."""
    settings = settings or get_settings()
    path = Path(path)

    suffix = path.suffix.lower()
    if suffix not in (".txt", ".zip"):
        raise ParseError(f"Unsupported file type {suffix or '(none)'!r}; expected .txt or .zip.")
    if not path.is_file():
        raise ParseError(f"File not found: {path}")

    size = path.stat().st_size
    if size == 0:
        raise ParseError("File is empty.")
    if size > settings.max_import_bytes:
        raise ParseError(f"File is {size / 1_048_576:.1f} MB; limit is {settings.max_import_mb} MB.")

    if suffix == ".zip":
        data = _read_chat_from_zip(path, settings.max_import_bytes)
    else:
        data = path.read_bytes()

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ParseError("File is not valid UTF-8 text.") from None

    return parse_chat_text(text, tz=settings.tz)


def _read_chat_from_zip(path: Path, max_bytes: int) -> bytes:
    """Read the chat .txt out of an export zip, in memory.

    Nothing is unpacked to disk, so a malicious zip can't write files anywhere.
    Size checks stop "zip bombs" (tiny zips that unpack to huge files).
    """
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ParseError("File is not a valid zip archive.") from None

    with archive:
        # Find the .txt files (ignoring folders and macOS's hidden __MACOSX copies).
        text_files = []
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            if info.filename.lower().endswith(".txt"):
                text_files.append(info)

        if not text_files:
            raise ParseError("Zip does not contain a chat .txt file.")
        if len(text_files) > 1:
            # iOS calls the chat "_chat.txt"; use that one if it's there.
            text_files = [i for i in text_files if PurePosixPath(i.filename).name == "_chat.txt"]
            if len(text_files) != 1:
                raise ParseError("Zip contains several chat files; export one chat at a time.")

        info = text_files[0]
        if info.file_size > max_bytes:
            raise ParseError("Chat inside the zip exceeds the size limit.")

        # Read at most one byte more than allowed, in case the zip lies about the size.
        with archive.open(info) as handle:
            data = handle.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ParseError("Chat inside the zip exceeds the size limit.")
        return data
