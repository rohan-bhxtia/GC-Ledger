"""Check everything the LLM returns before we trust it.

The LLM reads the chat and suggests records ("Karan paid ₹12,000"). This file checks each
suggestion against the real messages it cites.

  - A serious problem (the amount isn't in the message, the person isn't in the chat, ...)
    REJECTS the record. It is dropped, and the reason is kept.
  - A smaller doubt (the split is unclear, a date looks wrong, ...) keeps the record but adds
    a FLAG to it. The ledger won't count flagged expenses until a person has looked at them.

No LLM is used here. It's plain Python rules, so it always gives the same answer.
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

# Anything above ₹1 crore is almost certainly a mistake.
MAX_AMOUNT_PAISE = 1_00_00_000 * 100


# =============================================================================
# Amounts: "₹12,000", "1.5k", "2 lakh"  ->  whole paise
# =============================================================================

# A number: "12000", "12,000", "1,00,000", optionally with up to 2 decimals.
NUMBER = r"(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d{1,2}))?"
# An optional multiplier word after the number.
MULTIPLIER_WORD = r"(k|lakhs?|lacs?)"
MULTIPLIERS = {"k": 1_000, "lakh": 1_00_000, "lakhs": 1_00_000, "lac": 1_00_000, "lacs": 1_00_000}

# The whole text is one amount, e.g. "₹12,000" or "Rs. 500/-" or "2 lakh rupees".
WHOLE_AMOUNT = re.compile(
    rf"^(?:₹|rs\.?|inr)?\s*{NUMBER}\s*{MULTIPLIER_WORD}?\s*(?:/-)?\s*(?:rs\.?|rupees?|inr)?$", re.I
)
# An amount somewhere inside a longer message.
AMOUNT_IN_TEXT = re.compile(rf"(?<![\d.,]){NUMBER}(?:\s?{MULTIPLIER_WORD}(?![a-z]))?", re.I)


def _to_paise(whole_part: str, decimal_part: str | None, multiplier: str | None) -> int | None:
    """("12,000", None, None) -> 1200000.   ("1", "5", "k") -> 150000."""
    try:
        rupees = Decimal(whole_part.replace(",", ""))
        if decimal_part:
            rupees += Decimal(f"0.{decimal_part}")
    except InvalidOperation:
        return None

    rupees = rupees * MULTIPLIERS.get((multiplier or "").lower(), 1)
    paise = rupees * 100
    if paise != paise.to_integral_value():
        return None  # fractions of a paisa make no sense
    return int(paise)


def parse_amount(text: str | None) -> int | None:
    """Turn one amount like '₹12,000' into paise (1200000). None if it isn't an amount."""
    if not text:
        return None
    match = WHOLE_AMOUNT.match(text.strip())
    if not match:
        return None
    return _to_paise(*match.groups())


def amounts_in(text: str) -> set[int]:
    """Every amount (in paise) written anywhere in a message."""
    found = set()
    for match in AMOUNT_IN_TEXT.finditer(text):
        paise = _to_paise(*match.groups())
        if paise is not None:
            found.add(paise)
    return found


# =============================================================================
# Head counts: "split 3 ways", "teen log"
# =============================================================================

COUNT_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "teen": 3, "char": 4, "chaar": 4, "paanch": 5,
}


def parse_count(text: str | None) -> int | None:
    """'3' or 'three' or 'teen' -> 3. Only 2 to 50 people make sense."""
    if not text:
        return None
    word = text.strip().casefold()
    if word.isdigit():
        count = int(word)
    else:
        count = COUNT_WORDS.get(word)
    if count is None or count < 2 or count > 50:
        return None
    return count


def count_in(text: str, count: int) -> bool:
    """Is this number of people written in the message, as digits or as a word?"""
    if re.search(rf"(?<![\d.,]){count}(?![\d.,])", text):
        return True
    for word, number in COUNT_WORDS.items():
        if number == count and re.search(rf"\b{word}\b", text, re.I):
            return True
    return False


# =============================================================================
# People
# =============================================================================

def _normalise(text: str) -> str:
    """Lower case, single spaces: '  Riya   Sharma ' -> 'riya sharma'."""
    return " ".join(text.casefold().split())


def resolve_person(name: str, participants: list[str]) -> str | None:
    """Match a name from the LLM to a real chat participant.

    1. The full name matches (ignoring case)  -> that person.
    2. Only one participant has that first name -> that person.
    3. Otherwise -> None (we don't guess).
    """
    wanted = _normalise(name)
    if not wanted:
        return None

    for person in participants:
        if _normalise(person) == wanted:
            return person

    same_first_name = [p for p in participants if _normalise(p).split()[0] == wanted]
    if len(same_first_name) == 1:
        return same_first_name[0]
    return None


# =============================================================================
# The context of one window, and the result of checking it
# =============================================================================

# Text that looks like someone trying to give the LLM orders ("prompt injection").
INSTRUCTION_LIKE_TEXT = re.compile(
    r"ignore (all |the )?(previous|above|prior) instructions|system prompt|you are an? (ai|assistant|llm)"
    r"|disregard .{0,20}instructions|^\s*(system|assistant)\s*:",
    re.I | re.M,
)


class _Reject(Exception):
    """Raised inside a checker to throw the record away. The message is the reason."""


class _BelongsToPreviousWindow(Exception):
    """The record only cites the overlap messages, which the previous window already handled."""


@dataclass
class WindowContext:
    """What a checker needs to know about the window the record came from."""

    messages_by_id: dict[str, Message]
    new_ids: set[str]          # messages this window is responsible for (not the overlap)
    participants: list[str]
    tz: ZoneInfo


@dataclass
class ValidatedBatch:
    """The records that passed, the ones that didn't, and how many were overlap repeats."""

    expenses: list[Expense] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    commitments: list[Commitment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    skipped_overlap: int = 0


# =============================================================================
# Small checks shared by all record types
# =============================================================================

def _check_sources(cited_ids: list[str], ctx: WindowContext) -> tuple[list[str], str, list[str]]:
    """Check the cited message ids.

    Returns (ids without repeats, the text of those messages joined together, flags).
    """
    ids = list(dict.fromkeys(cited_ids))  # remove repeats, keep the order
    if not ids:
        raise _Reject("no source messages cited")
    for msg_id in ids:
        if msg_id not in ctx.messages_by_id:
            raise _Reject("cites a message id that is not in this window")

    cites_a_new_message = any(msg_id in ctx.new_ids for msg_id in ids)
    if not cites_a_new_message:
        raise _BelongsToPreviousWindow

    source_text = "\n".join(ctx.messages_by_id[msg_id].text for msg_id in ids)
    flags = []
    if INSTRUCTION_LIKE_TEXT.search(source_text):
        flags.append("suspicious_instruction_text")
    return ids, source_text, flags


def _check_person(name: str, ctx: WindowContext, role: str) -> str:
    person = resolve_person(name, ctx.participants)
    if person is None:
        raise _Reject(f"{role} is not a chat participant")
    return person


def _check_amount(amount_text: str, source_text: str, role: str) -> int:
    """The amount must be a real amount, in range, and written in the cited messages."""
    paise = parse_amount(amount_text)
    if paise is None:
        raise _Reject(f"{role} is not a recognisable amount")
    if paise <= 0 or paise > MAX_AMOUNT_PAISE:
        raise _Reject(f"{role} is out of range")
    if paise not in amounts_in(source_text):
        raise _Reject(f"{role} does not appear in the cited messages")
    return paise


def _check_date_and_time(
    date_text: str | None,
    time_text: str | None,
    when_text: str | None,
    ids: list[str],
    source_text: str,
    ctx: WindowContext,
) -> tuple[date | None, time | None, list[str]]:
    """Check a date and time from the LLM. Bad ones are dropped (set to None) and flagged."""
    flags: list[str] = []
    the_date = None
    the_time = None

    if date_text:
        try:
            the_date = date.fromisoformat(date_text)
        except ValueError:
            flags.append("invalid_date")
        else:
            # Plans are usually within a year before and two years after the message.
            sent_on = min(ctx.messages_by_id[i].timestamp for i in ids).astimezone(ctx.tz).date()
            earliest = sent_on - timedelta(days=366)
            latest = sent_on + timedelta(days=732)
            if the_date < earliest or the_date > latest:
                flags.append("date_out_of_range")
                the_date = None

    if time_text:
        try:
            the_time = time.fromisoformat(time_text)
        except ValueError:
            flags.append("invalid_time")
        if the_time is not None and the_date is None:
            # A time without a date can't be placed on a calendar.
            flags.append("time_without_date")
            the_time = None

    if when_text and _normalise(when_text) not in _normalise(source_text):
        flags.append("when_text_not_found")

    return the_date, the_time, flags


# =============================================================================
# One checker per record type
# =============================================================================

def _check_expense(raw: RawExpense, ctx: WindowContext) -> Expense:
    ids, text, flags = _check_sources(raw.source_msg_ids, ctx)
    payer = _check_person(raw.payer, ctx, "payer")

    # Who shares the cost (if the chat says).
    participants = None
    if raw.participants:
        participants = []
        for name in raw.participants:
            person = _check_person(name, ctx, "participant")
            if person not in participants:
                participants.append(person)

    # Individual shares (only for "exact" splits).
    shares: dict[str, int] = {}
    for share in raw.shares:
        person = _check_person(share.person, ctx, "share person")
        if person in shares:
            raise _Reject("same person has two shares")
        shares[person] = _check_amount(share.amount_text, text, "share amount")

    split = raw.split
    if shares and split is not SplitType.EXACT:
        raise _Reject("shares given but split is not 'exact'")
    if split is SplitType.EXACT and not shares:
        # "Exact" but no shares: the expense is still real, only the split is unclear.
        # Keep it, but mark it for review.
        split = SplitType.UNSPECIFIED
        flags.append("exact_split_without_shares")

    # The total amount.
    if raw.amount_text:
        total = _check_amount(raw.amount_text, text, "amount")
    elif shares:
        total = sum(shares.values())  # our code adds it up, never the LLM
    else:
        raise _Reject("missing amount")

    if shares:
        if sum(shares.values()) != total:
            raise _Reject("shares do not add up to the total")
        if participants is not None and set(participants) != set(shares):
            raise _Reject("participants and shares disagree")
        participants = list(shares)

    # "split 3 ways": check the number is really in the message.
    if split is SplitType.EQUAL and raw.split_count:
        count = parse_count(raw.split_count)
        if count is None or not count_in(text, count):
            flags.append("split_count_not_found")
        elif participants is None:
            # Nobody is named. If the chat has exactly that many people, it must be all of them.
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
        payer=payer,
        amount_paise=total,
        description=raw.description.strip(),
        participants=participants,
        split=split,
        shares_paise=shares,
        status=raw.status,
        source_msg_ids=ids,
        flags=flags,
    )


def _check_settlement(raw: RawSettlement, ctx: WindowContext) -> Settlement:
    ids, text, flags = _check_sources(raw.source_msg_ids, ctx)
    sender = _check_person(raw.from_person, ctx, "from_person")
    receiver = _check_person(raw.to_person, ctx, "to_person")
    if sender == receiver:
        raise _Reject("settlement to oneself")
    amount = _check_amount(raw.amount_text, text, "amount")
    return Settlement(
        from_person=sender,
        to_person=receiver,
        amount_paise=amount,
        status=raw.status,
        source_msg_ids=ids,
        flags=flags,
    )


def _check_commitment(raw: RawCommitment, ctx: WindowContext) -> Commitment:
    ids, text, flags = _check_sources(raw.source_msg_ids, ctx)
    owner = _check_person(raw.owner, ctx, "owner")
    if not raw.action.strip():
        raise _Reject("empty action")

    due_date, due_time, date_flags = _check_date_and_time(
        raw.due_date, raw.due_time, raw.when_text, ids, text, ctx
    )
    return Commitment(
        owner=owner,
        action=raw.action.strip(),
        status=raw.status,
        due_date=due_date,
        due_time=due_time,
        when_text=raw.when_text,
        source_msg_ids=ids,
        flags=flags + date_flags,
    )


def _check_event(raw: RawEvent, ctx: WindowContext) -> Event:
    ids, text, flags = _check_sources(raw.source_msg_ids, ctx)
    if not raw.title.strip():
        raise _Reject("empty title")

    on_date, at_time, date_flags = _check_date_and_time(
        raw.on_date, raw.at_time, raw.when_text, ids, text, ctx
    )

    # Multi-day events: the end date must be on or after the start, within a year.
    end_date = None
    if raw.end_date:
        try:
            end = date.fromisoformat(raw.end_date)
        except ValueError:
            end = None
        end_is_valid = (
            end is not None
            and on_date is not None
            and on_date <= end <= on_date + timedelta(days=366)
        )
        if end_is_valid:
            end_date = end
        else:
            date_flags.append("invalid_end_date")

    return Event(
        title=raw.title.strip(),
        status=raw.status,
        on_date=on_date,
        end_date=end_date,
        at_time=at_time,
        when_text=raw.when_text,
        source_msg_ids=ids,
        flags=flags + date_flags,
    )


# =============================================================================
# Check a whole LLM reply
# =============================================================================

def validate_extraction(raw: RawExtraction, ctx: WindowContext) -> ValidatedBatch:
    """Run the right checker on every record and sort the results."""
    batch = ValidatedBatch()

    # (kind, the LLM's records, the checker, where accepted ones go)
    groups = [
        ("expense", raw.expenses, _check_expense, batch.expenses),
        ("settlement", raw.settlements, _check_settlement, batch.settlements),
        ("commitment", raw.commitments, _check_commitment, batch.commitments),
        ("event", raw.events, _check_event, batch.events),
    ]

    for kind, raw_records, checker, accepted in groups:
        for raw_record in raw_records:
            try:
                accepted.append(checker(raw_record, ctx))
            except _BelongsToPreviousWindow:
                batch.skipped_overlap += 1
            except _Reject as problem:
                batch.rejected.append(Rejected(
                    kind=kind,
                    reason=str(problem),
                    source_msg_ids=raw_record.source_msg_ids,
                ))

    return batch
