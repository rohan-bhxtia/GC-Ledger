"""Pydantic schemas shared across the pipeline.

Phase 2: parsed chat messages.
Phase 3: extraction schemas (raw LLM output + validated records).
"""

from datetime import date, datetime, time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MessageKind(str, Enum):
    TEXT = "text"
    MEDIA = "media"        # photo/video/document placeholder (caption kept in text)
    DELETED = "deleted"    # "This message was deleted"
    SYSTEM = "system"      # group created, member added, encryption notice...


class Message(BaseModel):
    """One normalised chat message. Immutable once parsed."""

    model_config = ConfigDict(frozen=True)

    msg_id: str = Field(min_length=16, max_length=16)  # stable hash, used for citations/dedup
    seq: int = Field(ge=0)                              # position in this import
    line_no: int = Field(ge=1)                          # line in the source file
    timestamp: datetime
    sender: str | None                                  # None only for system messages
    text: str
    kind: MessageKind
    edited: bool = False

    @field_validator("timestamp")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _sender_matches_kind(self) -> "Message":
        if self.kind is MessageKind.SYSTEM and self.sender is not None:
            raise ValueError("system messages have no sender")
        if self.kind is not MessageKind.SYSTEM and not self.sender:
            raise ValueError("non-system messages need a sender")
        return self


class ParseStats(BaseModel):
    total_lines: int
    messages: int
    text_messages: int
    media_messages: int
    deleted_messages: int
    system_messages: int
    orphan_lines: int                 # non-empty lines before the first message header
    invalid_header_lines: list[int]   # line numbers with an impossible date/time (skipped)


class ParseResult(BaseModel):
    messages: list[Message]
    date_order: Literal["DMY", "MDY"]
    stats: ParseStats

    @property
    def participants(self) -> list[str]:
        return sorted({m.sender for m in self.messages if m.sender})


# =============================================================================
# Extraction (Phase 3)
#
# Raw*  = what the LLM returns. UNTRUSTED: names, ids and amounts may be wrong.
# Plain = validated records, built only by gcl.validation. Amounts in paise.
# =============================================================================

class SplitType(str, Enum):
    EQUAL = "equal"
    EXACT = "exact"              # individual shares written in the chat
    UNSPECIFIED = "unspecified"


class ExpenseStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    CANCELLED = "cancelled"


class SettlementStatus(str, Enum):
    CLAIMED = "claimed"          # "sending now" / "paid you back"
    CONFIRMED = "confirmed"      # the receiver acknowledged it


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


# --- What the LLM returns ------------------------------------------------------

class RawShare(BaseModel):
    person: str
    amount_text: str


class RawExpense(BaseModel):
    payer: str
    amount_text: str | None = None          # copied verbatim from the chat, never computed
    description: str = ""
    participants: list[str] | None = None   # None = not stated
    split: SplitType = SplitType.UNSPECIFIED
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
    when_text: str | None = None   # original wording, e.g. "kal tak"
    source_msg_ids: list[str] = []


class RawEvent(BaseModel):
    title: str
    status: EventStatus
    on_date: str | None = None     # YYYY-MM-DD
    at_time: str | None = None     # HH:MM
    when_text: str | None = None
    source_msg_ids: list[str] = []


class RawExtraction(BaseModel):
    expenses: list[RawExpense] = []
    settlements: list[RawSettlement] = []
    commitments: list[RawCommitment] = []
    events: list[RawEvent] = []


# --- Validated records -----------------------------------------------------------

class Expense(BaseModel):
    payer: str
    amount_paise: int = Field(gt=0)
    description: str
    participants: list[str] | None
    split: SplitType
    shares_paise: dict[str, int] = {}
    status: ExpenseStatus
    source_msg_ids: list[str] = Field(min_length=1)
    flags: list[str] = []


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
    at_time: time | None = None
    when_text: str | None = None
    source_msg_ids: list[str] = Field(min_length=1)
    flags: list[str] = []


class Rejected(BaseModel):
    """A record the LLM proposed that failed validation. Holds no chat text."""
    kind: Literal["expense", "settlement", "commitment", "event"]
    reason: str
    source_msg_ids: list[str]