"""The data shapes used everywhere in the project.

There are three groups:

1. Chat messages  - what the parser reads out of the WhatsApp export.
2. Raw* records   - what the LLM sends back. NOT trusted: names, ids and amounts may be wrong.
3. Clean records  - Expense, Settlement, Commitment, Event. Only validation.py creates these,
                    after checking the raw records against the real messages.

Money is always stored as whole paise (₹1 = 100 paise), never as a float.
"""

from datetime import date, datetime, time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# 1. Chat messages
# =============================================================================

class MessageKind(str, Enum):
    TEXT = "text"
    MEDIA = "media"        # photo / video / document (any caption is kept in the text)
    DELETED = "deleted"    # "This message was deleted"
    SYSTEM = "system"      # "Aman added Riya", encryption notice, ...


class Message(BaseModel):
    """One chat message. It cannot be changed after it is created."""

    model_config = ConfigDict(frozen=True)

    msg_id: str = Field(min_length=16, max_length=16)  # stable id, used for citations
    seq: int = Field(ge=0)                              # position in the chat (0, 1, 2, ...)
    line_no: int = Field(ge=1)                          # line number in the export file
    timestamp: datetime                                 # always has a timezone
    sender: str | None                                  # None only for system messages
    text: str
    kind: MessageKind
    edited: bool = False

    @field_validator("timestamp")
    @classmethod
    def _must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _sender_matches_kind(self) -> "Message":
        is_system = self.kind is MessageKind.SYSTEM
        if is_system and self.sender is not None:
            raise ValueError("system messages have no sender")
        if not is_system and not self.sender:
            raise ValueError("non-system messages need a sender")
        return self


class ParseResult(BaseModel):
    """Everything the parser found in one export file."""

    messages: list[Message]
    date_order: Literal["DMY", "MDY"]  # day-first (14/10) or month-first (10/14)

    @property
    def participants(self) -> list[str]:
        """Everyone who sent at least one message, sorted by name."""
        names = set()
        for message in self.messages:
            if message.sender:
                names.add(message.sender)
        return sorted(names)


# =============================================================================
# Statuses and split types (shared by raw and clean records)
# =============================================================================

class SplitType(str, Enum):
    EQUAL = "equal"              # everyone pays the same
    EXACT = "exact"              # each person's share is written in the chat
    UNSPECIFIED = "unspecified"  # the chat doesn't say


class ExpenseStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    CANCELLED = "cancelled"


class SettlementStatus(str, Enum):
    CLAIMED = "claimed"          # the sender says "paid you back"
    CONFIRMED = "confirmed"      # the receiver said they got it


class CommitmentStatus(str, Enum):
    SUGGESTED = "suggested"
    PROMISED = "promised"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class EventStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


# =============================================================================
# 2. Raw records: exactly what the LLM returns (not trusted yet)
# =============================================================================

class RawShare(BaseModel):
    person: str
    amount_text: str


class RawExpense(BaseModel):
    payer: str
    amount_text: str | None = None          # copied word for word from the chat, e.g. "₹12,000"
    description: str = ""
    participants: list[str] | None = None   # None = the chat doesn't say who shares it
    split: SplitType = SplitType.UNSPECIFIED
    split_count: str | None = None          # e.g. "3" from "split 3 ways"
    shares: list[RawShare] = []
    status: ExpenseStatus
    source_msg_ids: list[str] = []


class RawSettlement(BaseModel):
    from_person: str
    to_person: str
    amount_text: str
    status: SettlementStatus
    source_msg_ids: list[str] = []


class RawCommitment(BaseModel):
    owner: str
    action: str
    status: CommitmentStatus
    due_date: str | None = None    # YYYY-MM-DD
    due_time: str | None = None    # HH:MM
    when_text: str | None = None   # the original words, e.g. "kal tak"
    source_msg_ids: list[str] = []


class RawEvent(BaseModel):
    title: str
    status: EventStatus
    on_date: str | None = None     # YYYY-MM-DD (first day)
    end_date: str | None = None    # YYYY-MM-DD (last day, multi-day events only)
    at_time: str | None = None     # HH:MM
    when_text: str | None = None
    source_msg_ids: list[str] = []


class RawExtraction(BaseModel):
    """The whole LLM reply for one window of messages."""

    expenses: list[RawExpense] = []
    settlements: list[RawSettlement] = []
    commitments: list[RawCommitment] = []
    events: list[RawEvent] = []


# =============================================================================
# 3. Clean records: checked by validation.py, safe to store and count
# =============================================================================

class Expense(BaseModel):
    payer: str
    amount_paise: int = Field(gt=0)
    description: str
    participants: list[str] | None
    split: SplitType
    shares_paise: dict[str, int] = {}
    status: ExpenseStatus
    source_msg_ids: list[str] = Field(min_length=1)
    flags: list[str] = []          # warnings, e.g. "split_unspecified"


class Settlement(BaseModel):
    from_person: str
    to_person: str
    amount_paise: int = Field(gt=0)
    status: SettlementStatus
    source_msg_ids: list[str] = Field(min_length=1)
    flags: list[str] = []


class Commitment(BaseModel):
    owner: str
    action: str
    status: CommitmentStatus
    due_date: date | None = None
    due_time: time | None = None
    when_text: str | None = None
    source_msg_ids: list[str] = Field(min_length=1)
    flags: list[str] = []


class Event(BaseModel):
    title: str
    status: EventStatus
    on_date: date | None = None
    end_date: date | None = None
    at_time: time | None = None
    when_text: str | None = None
    source_msg_ids: list[str] = Field(min_length=1)
    flags: list[str] = []


class Rejected(BaseModel):
    """A record the LLM suggested that failed the checks. Holds no chat text."""

    kind: Literal["expense", "settlement", "commitment", "event"]
    reason: str
    source_msg_ids: list[str]
