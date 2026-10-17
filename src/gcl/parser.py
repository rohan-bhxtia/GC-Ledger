"""WhatsApp chat export -> list[Message].

Supported formats:
  Android: 14/10/2026, 9:15 pm - Name: text
  iOS:     [14/10/26, 9:15:30 PM] Name: text
Input may be the exported .txt or the exported .zip (read in memory, never extracted).

Privacy: this module logs counts only, never message content.
"""

from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from zoneinfo import ZoneInfo

from gcl.config import Settings, get_settings
from gcl.models import Message, MessageKind, ParseResult, ParseStats

logger = logging.getLogger(__name__)

DateOrder = Literal["auto", "DMY", "MDY"]
ALLOWED_SUFFIXES = {".txt", ".zip"}


class ParseError(ValueError):
    """The input is not a usable WhatsApp export."""


# --- Line patterns ----------------------------------------------------------

_DATE = r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]\.?\s?m\.?)?"
_ANDROID = re.compile(rf"^(?P<date>{_DATE}),?\s(?P<time>{_TIME})\s[-–]\s(?P<rest>.*)$", re.I)
_IOS = re.compile(rf"^\[(?P<date>{_DATE}),?\s(?P<time>{_TIME})\]\s(?P<rest>.*)$", re.I)

# Invisible direction marks WhatsApp inserts around names, numbers and system text.
_BIDI = re.compile("[‎‏‪-‮⁦-⁩﻿]")
_SPACES = str.maketrans({" ": " ", " ": " "})  # narrow/no-break spaces

_MEDIA = re.compile(
    r"<media omitted>"
    r"|(?:image|video|audio|sticker|gif|document|contact card) omitted"
    r"|<attached: [^>]+>"
    r"|.+ \(file attached\)"
    r"|.* document omitted",
    re.I,
)
_DELETED = re.compile(r"(?:this message was deleted|you deleted this message)\.?", re.I)
_EDITED_SUFFIX = re.compile(r"\s*<this message was edited>\s*$", re.I)
_ENCRYPTION_NOTICE = "messages and calls are end-to-end encrypted"
_QUOTES = set('"“”')
_MAX_SENDER_LEN = 60


@dataclass
class _RawEntry:
    line_no: int
    date: str
    time: str
    rest: str
    extra: list[str] = field(default_factory=list)  # continuation lines


# --- Pass 1: group lines into entries ---------------------------------------

def _split_entries(text: str) -> tuple[list[_RawEntry], int, int]:
    entries: list[_RawEntry] = []
    orphans = 0
    lines = text.splitlines()
    for line_no, raw in enumerate(lines, start=1):
        line = raw.translate(_SPACES)
        match = _IOS.match(line.lstrip("‎‏﻿")) or _ANDROID.match(
            line.lstrip("‎‏﻿")
        )
        if match:
            entries.append(_RawEntry(line_no, match["date"], match["time"], match["rest"]))
        elif entries:
            entries[-1].extra.append(line)
        elif line.strip():
            orphans += 1
    return entries, orphans, len(lines)


# --- Dates and times ---------------------------------------------------------

def _date_parts(date: str) -> tuple[int, int, int]:
    a, b, year = (int(p) for p in re.split(r"[/.\-]", date))
    return a, b, year


def _detect_date_order(entries: list[_RawEntry], requested: DateOrder) -> Literal["DMY", "MDY"]:
    if requested != "auto":
        return requested
    parts = [_date_parts(e.date) for e in entries]
    first_over_12 = any(a > 12 for a, _, _ in parts)
    second_over_12 = any(b > 12 for _, b, _ in parts)
    if first_over_12 and second_over_12:
        raise ParseError("Inconsistent date format: both DD/MM and MM/DD dates found.")
    if second_over_12:
        return "MDY"
    return "DMY"  # day-first is the Indian default; also used when every date is ambiguous


def _to_datetime(date: str, time: str, order: str, tz: ZoneInfo) -> datetime | None:
    a, b, year = _date_parts(date)
    day, month = (a, b) if order == "DMY" else (b, a)
    if year < 100:
        year += 2000

    t = time.lower().replace(".", "").replace(" ", "")
    meridiem = None
    if t.endswith(("am", "pm")):
        meridiem, t = t[-2:], t[:-2]
    pieces = [int(p) for p in t.split(":")]
    hour, minute = pieces[0], pieces[1]
    second = pieces[2] if len(pieces) == 3 else 0
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=tz)
    except ValueError:
        return None


# --- Message classification ---------------------------------------------------

def _plausible_sender(name: str) -> bool:
    return 0 < len(name) <= _MAX_SENDER_LEN and not (_QUOTES & set(name))


def _classify(rest: str) -> tuple[str | None, str, MessageKind]:
    """Split 'Name: text' and decide the message kind (from the first line only)."""
    sender, sep, body = rest.partition(": ")
    sender = _BIDI.sub("", sender).strip()
    if not sep or not _plausible_sender(sender):
        # Android system lines have no 'Name: ' part, e.g. 'Aman added Riya'.
        return None, _BIDI.sub("", rest).strip(), MessageKind.SYSTEM

    lrm_prefixed = body.startswith("‎")  # iOS marks system/media text this way
    clean = _BIDI.sub("", body).strip()
    if _MEDIA.fullmatch(clean):
        return sender, clean, MessageKind.MEDIA
    if _DELETED.fullmatch(clean):
        return sender, clean, MessageKind.DELETED
    if lrm_prefixed or clean.lower().startswith(_ENCRYPTION_NOTICE):
        # iOS system messages appear with the group name as "sender".
        return None, clean, MessageKind.SYSTEM
    return sender, clean, MessageKind.TEXT


def _make_id(timestamp: datetime, sender: str | None, text: str, counter: Counter) -> str:
    """Stable across re-imports of the same chat; identical repeats get distinct ids."""
    key = f"{timestamp.isoformat()}\x1f{sender or ''}\x1f{text}"
    occurrence = counter[key]
    counter[key] += 1
    return hashlib.sha256(f"{key}\x1f{occurrence}".encode()).hexdigest()[:16]


# --- Public API ----------------------------------------------------------------

def parse_chat_text(text: str, *, tz: ZoneInfo, date_order: DateOrder = "auto") -> ParseResult:
    if not text.strip():
        raise ParseError("Chat export is empty.")
    if "\x00" in text:
        raise ParseError("File does not look like a text chat export.")

    entries, orphans, total_lines = _split_entries(text)
    if not entries:
        raise ParseError("No WhatsApp messages found. Is this an exported chat?")

    order = _detect_date_order(entries, date_order)
    messages: list[Message] = []
    invalid_lines: list[int] = []
    id_counter: Counter = Counter()

    for entry in entries:
        timestamp = _to_datetime(entry.date, entry.time, order, tz)
        if timestamp is None:
            invalid_lines.append(entry.line_no)
            continue

        sender, first_line, kind = _classify(entry.rest)
        extra = [_BIDI.sub("", line).rstrip() for line in entry.extra]
        body = "\n".join([first_line, *extra]).strip()
        edited = bool(_EDITED_SUFFIX.search(body))
        if edited:
            body = _EDITED_SUFFIX.sub("", body)

        messages.append(
            Message(
                msg_id=_make_id(timestamp, sender, body, id_counter),
                seq=len(messages),
                line_no=entry.line_no,
                timestamp=timestamp,
                sender=sender,
                text=body,
                kind=kind,
                edited=edited,
            )
        )

    if not messages:
        raise ParseError("No messages with a valid date/time were found.")

    kinds = Counter(m.kind for m in messages)
    stats = ParseStats(
        total_lines=total_lines,
        messages=len(messages),
        text_messages=kinds[MessageKind.TEXT],
        media_messages=kinds[MessageKind.MEDIA],
        deleted_messages=kinds[MessageKind.DELETED],
        system_messages=kinds[MessageKind.SYSTEM],
        orphan_lines=orphans,
        invalid_header_lines=invalid_lines,
    )
    logger.info(
        "Parsed %d messages (%d system, %d invalid headers, %d orphan lines, order=%s)",
        stats.messages, stats.system_messages, len(invalid_lines), orphans, order,
    )
    return ParseResult(messages=messages, date_order=order, stats=stats)


def parse_chat_file(
    path: str | Path,
    *,
    settings: Settings | None = None,
    date_order: DateOrder = "auto",
) -> ParseResult:
    """Validate and parse an exported chat (.txt or .zip)."""
    settings = settings or get_settings()
    path = Path(path)

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ParseError(f"Unsupported file type {suffix or '(none)'!r}; expected .txt or .zip.")
    if not path.is_file():
        raise ParseError(f"File not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ParseError("File is empty.")
    if size > settings.max_import_bytes:
        raise ParseError(f"File is {size / 1_048_576:.1f} MB; limit is {settings.max_import_mb} MB.")

    data = _read_zip(path, settings.max_import_bytes) if suffix == ".zip" else path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ParseError("File is not valid UTF-8 text.") from None
    return parse_chat_text(text, tz=settings.tz, date_order=date_order)


def _read_zip(path: Path, max_bytes: int) -> bytes:
    """Read the chat .txt from an export zip in memory (no extraction -> no path traversal)."""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ParseError("File is not a valid zip archive.") from None

    with archive:
        texts = [
            info for info in archive.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(".txt")
            and not info.filename.startswith("__MACOSX/")
        ]
        if not texts:
            raise ParseError("Zip does not contain a chat .txt file.")
        if len(texts) > 1:
            # iOS exports name the chat "_chat.txt"; prefer it if present.
            texts = [i for i in texts if PurePosixPath(i.filename).name == "_chat.txt"]
            if len(texts) != 1:
                raise ParseError("Zip contains several chat files; export one chat at a time.")

        info = texts[0]
        if info.file_size > max_bytes:  # zip-bomb guard (declared size)
            raise ParseError("Chat inside the zip exceeds the size limit.")
        with archive.open(info) as handle:
            data = handle.read(max_bytes + 1)  # guard against a lying header
        if len(data) > max_bytes:
            raise ParseError("Chat inside the zip exceeds the size limit.")
        return data