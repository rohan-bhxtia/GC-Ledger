"""Balances from validated records. Pure Python, integer paise, no LLM.

Rules:
- Only CONFIRMED expenses count. Proposed/cancelled are ignored; disputed or flagged
  expenses are listed as "needs review" and never silently added.
- Settlements count when CLAIMED or CONFIRMED (the explanation shows which).
- Equal splits divide in paise; leftover paise go one each to participants in name order,
  so totals always add up exactly.
- Invariant: all net balances sum to zero (checked on every computation).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from gcl.db import Database, Stored
from gcl.models import Expense, ExpenseStatus, Message, Settlement, SettlementStatus, SplitType

# Flags that make an expense unsafe to count without a human looking at it.
BLOCKING_FLAGS = {
    "possible_duplicate": "may be a duplicate of another expense",
    "participants_unspecified": "who shares the cost is not stated",
    "split_unspecified": "how to split it is not stated",
    "split_count_mismatch": "head count does not match the people named",
    "split_count_not_found": "head count not found in the message",
    "suspicious_instruction_text": "message contains instruction-like text",
}


@dataclass(frozen=True)
class Entry:
    """One line of a person's balance: positive = they are owed, negative = they owe."""
    person: str
    delta_paise: int
    kind: str              # "paid" | "share" | "sent" | "received"
    record_id: int
    description: str
    source_msg_ids: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class Transfer:
    debtor: str
    creditor: str
    amount_paise: int


@dataclass(frozen=True)
class Excluded:
    record_id: int
    kind: str              # "expense" | "settlement"
    description: str
    amount_paise: int
    reason: str
    needs_review: bool     # True = a human should decide; False = correctly ignored


@dataclass
class Ledger:
    balances: dict[str, int] = field(default_factory=dict)
    entries: list[Entry] = field(default_factory=list)
    transfers: list[Transfer] = field(default_factory=list)
    excluded: list[Excluded] = field(default_factory=list)

    def entries_for(self, person: str) -> list[Entry]:
        return [e for e in self.entries if e.person == person]

    @property
    def needs_review(self) -> list[Excluded]:
        return [x for x in self.excluded if x.needs_review]


# --- Formatting --------------------------------------------------------------------------

def rupees(paise: int) -> str:
    sign = "-" if paise < 0 else ""
    whole, part = divmod(abs(paise), 100)
    return f"{sign}₹{_indian_grouping(whole)}.{part:02d}"


def _indian_grouping(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


# --- Core computation ------------------------------------------------------------------------

def split_equally(amount_paise: int, people: list[str]) -> dict[str, int]:
    """Exact integer split; leftover paise go to the first names alphabetically."""
    ordered = sorted(people)
    base, remainder = divmod(amount_paise, len(ordered))
    return {p: base + (1 if i < remainder else 0) for i, p in enumerate(ordered)}


def _expense_exclusion(expense: Expense) -> tuple[str, bool] | None:
    if expense.status is ExpenseStatus.PROPOSED:
        return "not confirmed (only proposed)", False
    if expense.status is ExpenseStatus.CANCELLED:
        return "cancelled", False
    if expense.status is ExpenseStatus.DISPUTED:
        return "disputed in the chat", True
    blocking = [BLOCKING_FLAGS[f] for f in expense.flags if f in BLOCKING_FLAGS]
    if blocking:
        return "; ".join(blocking), True
    if not expense.participants:
        return "who shares the cost is not stated", True
    return None


def compute_ledger(expenses: list[Stored[Expense]], settlements: list[Stored[Settlement]]) -> Ledger:
    ledger = Ledger()
    net: dict[str, int] = defaultdict(int)

    for stored in expenses:
        e = stored.record
        exclusion = _expense_exclusion(e)
        if exclusion:
            reason, review = exclusion
            ledger.excluded.append(Excluded(stored.id, "expense", e.description, e.amount_paise, reason, review))
            continue

        if e.split is SplitType.EXACT:
            shares, how = e.shares_paise, "exact share"
        else:
            shares = split_equally(e.amount_paise, e.participants)
            how = f"{rupees(e.amount_paise)} ÷ {len(shares)}"
        note = "participants inferred from head count" if "participants_inferred" in e.flags else ""
        sources = tuple(e.source_msg_ids)

        net[e.payer] += e.amount_paise
        ledger.entries.append(Entry(e.payer, e.amount_paise, "paid", stored.id, e.description, sources, note))
        for person, share in shares.items():
            net[person] -= share
            ledger.entries.append(Entry(person, -share, "share", stored.id, e.description, sources, how))

    for stored in settlements:
        s = stored.record
        if s.flags and any(f in BLOCKING_FLAGS for f in s.flags):
            reason = "; ".join(BLOCKING_FLAGS[f] for f in s.flags if f in BLOCKING_FLAGS)
            ledger.excluded.append(Excluded(stored.id, "settlement",
                                            f"{s.from_person} → {s.to_person}", s.amount_paise, reason, True))
            continue
        label = "claimed by sender" if s.status is SettlementStatus.CLAIMED else "confirmed"
        sources = tuple(s.source_msg_ids)
        net[s.from_person] += s.amount_paise
        net[s.to_person] -= s.amount_paise
        ledger.entries.append(Entry(s.from_person, s.amount_paise, "sent", stored.id,
                                    f"payment to {s.to_person}", sources, label))
        ledger.entries.append(Entry(s.to_person, -s.amount_paise, "received", stored.id,
                                    f"payment from {s.from_person}", sources, label))

    if sum(net.values()) != 0:  # can only happen through a bug; never show wrong money
        raise AssertionError("ledger does not balance")

    ledger.balances = dict(sorted(net.items()))
    ledger.transfers = settle_up(ledger.balances)
    return ledger


def settle_up(balances: dict[str, int]) -> list[Transfer]:
    """Few transfers that clear every balance: largest debtor pays largest creditor, repeat."""
    debtors = sorted(((-v, p) for p, v in balances.items() if v < 0), key=lambda x: (-x[0], x[1]))
    creditors = sorted(((v, p) for p, v in balances.items() if v > 0), key=lambda x: (-x[0], x[1]))
    debtors = [list(d) for d in debtors]
    creditors = [list(c) for c in creditors]
    transfers: list[Transfer] = []
    i = j = 0
    while i < len(debtors) and j < len(creditors):
        amount = min(debtors[i][0], creditors[j][0])
        transfers.append(Transfer(debtors[i][1], creditors[j][1], amount))
        debtors[i][0] -= amount
        creditors[j][0] -= amount
        if debtors[i][0] == 0:
            i += 1
        if creditors[j][0] == 0:
            j += 1
    return transfers


# --- Explanations -----------------------------------------------------------------------------

def explain(ledger: Ledger, person: str, messages: dict[str, Message], tz: ZoneInfo) -> str:
    """Step-by-step: every line that makes up `person`'s balance, with the source messages."""
    balance = ledger.balances.get(person, 0)
    status = "is owed" if balance > 0 else "owes" if balance < 0 else "is settled"
    lines = [f"{person} {status} {rupees(abs(balance))}" if balance else f"{person} is settled"]

    for entry in ledger.entries_for(person):
        sign = "+" if entry.delta_paise > 0 else "−"
        what = {
            "paid": f'paid for "{entry.description}"',
            "share": f'share of "{entry.description}" ({entry.note})',
            "sent": f"{entry.description} ({entry.note})",
            "received": f"{entry.description} ({entry.note})",
        }[entry.kind]
        if entry.kind == "paid" and entry.note:
            what += f" [{entry.note}]"
        lines.append(f"  {sign} {rupees(abs(entry.delta_paise)):>14}  {what}")
        for msg_id in entry.source_msg_ids:
            m = messages.get(msg_id)
            if m:
                when = m.timestamp.astimezone(tz).strftime("%d %b %H:%M")
                lines.append(f"      ↳ {when} {m.sender}: \"{_short(m.text)}\" [{msg_id[:8]}]")
    lines.append(f"  = {rupees(balance):>15}")
    return "\n".join(lines)


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- Convenience ------------------------------------------------------------------------------

def ledger_for_chat(db: Database, chat_id: int) -> Ledger:
    return compute_ledger(db.records(chat_id, "expense"), db.records(chat_id, "settlement"))