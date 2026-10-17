"""Pydantic schemas shared across the pipeline.

Phase 2: parsed chat messages. Extraction schemas (expenses, commitments,
events) are added in Phase 3.
"""

from datetime import datetime
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