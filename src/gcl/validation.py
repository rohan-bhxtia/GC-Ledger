"""Deterministic checks on LLM output. Nothing the LLM says is trusted until it passes here.

Hard failures -> Rejected (record dropped, reason kept).
Soft problems -> record kept with `flags` so the ledger/UI can treat it as uncertain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from gcl.models import (
    Commitment, Event, Expense, Message, RawCommitment, RawEvent, RawExpense,
    RawExtraction, RawSettlement, Rejected, Settlement, SplitType,
)

MAX_AMOUNT_PAISE = 1_00_00_000 * 100  # ₹1 crore: anything above is almost certainly an error

# --- Amounts ------------------------------------------------------------------------

_NUM = r"(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d{1,2}))?"
_SUFFIX = r"(k|lakhs?|lacs?)"
_AMOUNT_EXACT = re.compile(
    rf"^(?:₹|rs\.?|inr)?\s*{_NUM}\s*{_SUFFIX}?\s*(?:/-)?\s*(?:rs\.?|rupees?|inr)?$", re.I
)
_AMOUNT_ANY = re.compile(rf"(?<![\d.,]){_NUM}(?:\s?{_SUFFIX}(?![a-z]))?", re.I)
_MULTIPLIER = {"k": 1_000, "lakh": 1_00_000, "lakhs": 1_00_000, "lac": 1_00_000, "lacs": 1_00_000}


def _to_paise(integer: str, fraction: str | None, suffix: str | None) -> int | None:
    try:
        value = Decimal(integer.replace(",", ""))
        if fraction:
            value += Decimal(f"0.{fraction}")
    except InvalidOperation:
        return None
    paise = value * _MULTIPLIER.get((suffix or "").lower(), 1) * 100
    if paise != paise.to_integral_value():
        return None
    return int(paise)


def parse_amount(text: str | None) -> int | None:
    """'₹12,000' -> 1200000 paise. Returns None if not a plain amount."""
    if not text:
        return None
    match = _AMOUNT_EXACT.match(text.strip())
    return _to_paise(*match.groups()) if match else None


def amounts_in(text: str) -> set[int]:
    """Every amount (in paise) written anywhere in `text`."""
    found = set()
    for match in _AMOUNT_ANY.finditer(text):
        value = _to_paise(*match.groups())
        if value is not None:
            found.add(value)
    return found


# --- Head counts ("split 3 ways") -------------------------------------------------------

_COUNT_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "teen": 3, "char": 4, "chaar": 4, "paanch": 5,
}


def parse_count(text: str | None) -> int | None:
    if not text:
        return None
    value = text.strip().casefold()
    count = int(value) if value.isdigit() else _COUNT_WORDS.get(value)
    return count if count is not None and 2 <= count <= 50 else None


def count_in(text: str, count: int) -> bool:
    """Is `count` written in `text`, as digits or a word?"""
    if re.search(rf"(?<![\d.,]){count}(?![\d.,])", text):
        return True
    words = [w for w, n in _COUNT_WORDS.items() if n == count]
    return any(re.search(rf"\b{w}\b", text, re.I) for w in words)


# --- People --------------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def resolve_person(name: str, participants: list[str]) -> str | None:
    """Exact (case-insensitive) match, else a unique first-name match, else None."""
    wanted = _norm(name)
    if not wanted:
        return None
    for person in participants:
        if _norm(person) == wanted:
            return person
    first_name = [p for p in participants if _norm(p).split()[0] == wanted]
    return first_name[0] if len(first_name) == 1 else None


# --- Context & helpers -------------------------------------------------------------------

_INJECTION_HINTS = re.compile(
    r"ignore (all |the )?(previous|above|prior) instructions|system prompt|you are an? (ai|assistant|llm)"
    r"|disregard .{0,20}instructions|^\s*(system|assistant)\s*:",
    re.I | re.M,
)


class _Reject(Exception):
    pass


class _AlreadyCovered(Exception):
    """Record cites only overlap/context messages: the previous window owns it."""


@dataclass
class WindowContext:
    messages_by_id: dict[str, Message]
    new_ids: set[str]                 # ids this window is responsible for (excludes overlap)
    participants: list[str]
    tz: ZoneInfo


@dataclass
class ValidatedBatch:
    expenses: list[Expense] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    commitments: list[Commitment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    skipped_overlap: int = 0


def _sources(ids: list[str], ctx: WindowContext) -> tuple[list[str], str, list[str]]:
    """Validate citations; return (ids, combined source text, flags)."""
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise _Reject("no source messages cited")
    if any(i not in ctx.messages_by_id for i in ids):
        raise _Reject("cites a message id that is not in this window")
    if not set(ids) & ctx.new_ids:
        raise _AlreadyCovered
    text = "\n".join(ctx.messages_by_id[i].text for i in ids)
    flags = ["suspicious_instruction_text"] if _INJECTION_HINTS.search(text) else []
    return ids, text, flags


def _person(name: str, ctx: WindowContext, role: str) -> str:
    person = resolve_person(name, ctx.participants)
    if person is None:
        raise _Reject(f"{role} is not a chat participant")
    return person


def _grounded_amount(amount_text: str, source_text: str, role: str) -> int:
    paise = parse_amount(amount_text)
    if paise is None:
        raise _Reject(f"{role} is not a recognisable amount")
    if paise <= 0 or paise > MAX_AMOUNT_PAISE:
        raise _Reject(f"{role} is out of range")
    if paise not in amounts_in(source_text):
        raise _Reject(f"{role} does not appear in the cited messages")
    return paise


def _when(
    date_s: str | None, time_s: str | None, when_text: str | None,
    ids: list[str], source_text: str, ctx: WindowContext,
) -> tuple[date | None, time | None, list[str]]:
    flags: list[str] = []
    on_date = at_time = None
    if date_s:
        try:
            on_date = date.fromisoformat(date_s)
        except ValueError:
            flags.append("invalid_date")
        else:
            anchor = min(ctx.messages_by_id[i].timestamp for i in ids).astimezone(ctx.tz).date()
            if not anchor - timedelta(days=366) <= on_date <= anchor + timedelta(days=732):
                flags.append("date_out_of_range")
                on_date = None
    if time_s:
        try:
            at_time = time.fromisoformat(time_s)
        except ValueError:
            flags.append("invalid_time")
        if at_time is not None and on_date is None:
            flags.append("time_without_date")
            at_time = None
    if when_text and _norm(when_text) not in _norm(source_text):
        flags.append("when_text_not_found")
    return on_date, at_time, flags


# --- Per-record validators -----------------------------------------------------------------

def _expense(raw: RawExpense, ctx: WindowContext) -> Expense:
    ids, text, flags = _sources(raw.source_msg_ids, ctx)
    payer = _person(raw.payer, ctx, "payer")

    participants = None
    if raw.participants:
        participants = list(dict.fromkeys(_person(p, ctx, "participant") for p in raw.participants))

    shares: dict[str, int] = {}
    for share in raw.shares:
        person = _person(share.person, ctx, "share person")
        if person in shares:
            raise _Reject("same person has two shares")
        shares[person] = _grounded_amount(share.amount_text, text, "share amount")

    split = raw.split
    if shares and split is not SplitType.EXACT:
        raise _Reject("shares given but split is not 'exact'")
    if split is SplitType.EXACT and not shares:
        # The expense itself is still grounded; only the split is unclear. Keep it for review.
        split = SplitType.UNSPECIFIED
        flags.append("exact_split_without_shares")

    if raw.amount_text:
        total = _grounded_amount(raw.amount_text, text, "amount")
    elif shares:
        total = sum(shares.values())  # code does the maths, not the LLM
    else:
        raise _Reject("missing amount")

    if shares:
        if sum(shares.values()) != total:
            raise _Reject("shares do not add up to the total")
        if participants is not None and set(participants) != set(shares):
            raise _Reject("participants and shares disagree")
        participants = list(shares)

    if split is SplitType.EQUAL and raw.split_count:
        count = parse_count(raw.split_count)
        if count is None or not count_in(text, count):
            flags.append("split_count_not_found")
        elif participants is None:
            # "split 3 ways" in a chat with exactly 3 people: code fills them in, and says so.
            if count == len(ctx.participants):
                participants = list(ctx.participants)
                flags.append("participants_inferred")
            else:
                flags.append("split_count_mismatch")
        elif count != len(participants):
            flags.append("split_count_mismatch")

    if split is SplitType.EQUAL and participants is None:
        flags.append("participants_unspecified")
    if split is SplitType.UNSPECIFIED:
        flags.append("split_unspecified")

    return Expense(
        payer=payer, amount_paise=total, description=raw.description.strip(),
        participants=participants, split=split, shares_paise=shares,
        status=raw.status, source_msg_ids=ids, flags=flags,
    )


def _settlement(raw: RawSettlement, ctx: WindowContext) -> Settlement:
    ids, text, flags = _sources(raw.source_msg_ids, ctx)
    sender = _person(raw.from_person, ctx, "from_person")
    receiver = _person(raw.to_person, ctx, "to_person")
    if sender == receiver:
        raise _Reject("settlement to oneself")
    amount = _grounded_amount(raw.amount_text, text, "amount")
    return Settlement(
        from_person=sender, to_person=receiver, amount_paise=amount,
        status=raw.status, source_msg_ids=ids, flags=flags,
    )


def _commitment(raw: RawCommitment, ctx: WindowContext) -> Commitment:
    ids, text, flags = _sources(raw.source_msg_ids, ctx)
    owner = _person(raw.owner, ctx, "owner")
    if not raw.action.strip():
        raise _Reject("empty action")
    due_date, due_time, when_flags = _when(raw.due_date, raw.due_time, raw.when_text, ids, text, ctx)
    return Commitment(
        owner=owner, action=raw.action.strip(), status=raw.status,
        due_date=due_date, due_time=due_time, when_text=raw.when_text,
        source_msg_ids=ids, flags=flags + when_flags,
    )


def _event(raw: RawEvent, ctx: WindowContext) -> Event:
    ids, text, flags = _sources(raw.source_msg_ids, ctx)
    if not raw.title.strip():
        raise _Reject("empty title")
    on_date, at_time, when_flags = _when(raw.on_date, raw.at_time, raw.when_text, ids, text, ctx)
    end_date = None
    if raw.end_date:
        try:
            end = date.fromisoformat(raw.end_date)
        except ValueError:
            end = None
        if end is None or on_date is None or not on_date <= end <= on_date + timedelta(days=366):
            when_flags.append("invalid_end_date")
        else:
            end_date = end
    return Event(
        title=raw.title.strip(), status=raw.status, on_date=on_date, end_date=end_date, at_time=at_time,
        when_text=raw.when_text, source_msg_ids=ids, flags=flags + when_flags,
    )


def validate_extraction(raw: RawExtraction, ctx: WindowContext) -> ValidatedBatch:
    batch = ValidatedBatch()
    groups = [
        ("expense", raw.expenses, _expense, batch.expenses),
        ("settlement", raw.settlements, _settlement, batch.settlements),
        ("commitment", raw.commitments, _commitment, batch.commitments),
        ("event", raw.events, _event, batch.events),
    ]
    for kind, items, validator, accepted in groups:
        for item in items:
            try:
                accepted.append(validator(item, ctx))
            except _AlreadyCovered:
                batch.skipped_overlap += 1
            except _Reject as exc:
                batch.rejected.append(
                    Rejected(kind=kind, reason=str(exc), source_msg_ids=item.source_msg_ids)
                )
    return batch