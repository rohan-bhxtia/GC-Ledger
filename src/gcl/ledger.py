"""Work out who owes whom. Plain Python maths in whole paise, no LLM.

Rules:
  - Only CONFIRMED expenses count. Proposed and cancelled ones are ignored.
  - Disputed or flagged expenses go on a "needs review" list. They are never added silently.
  - Settlements ("paid you back") count whether the sender claimed them or the receiver confirmed.
  - An equal split is done in paise. Leftover paise go one each to the first names
    alphabetically, so the shares always add up to exactly the total.
  - Safety check: everyone's balances must add up to zero. If they don't, we stop
    rather than show wrong money.

A positive balance means the person is owed money. A negative balance means they owe money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from gcl.db import Database, Stored
from gcl.models import Expense, ExpenseStatus, Message, Settlement, SettlementStatus, SplitType

# Flags that mean "a person should look at this before we count it", with a plain reason.
BLOCKING_FLAGS = {
    "participants_unspecified": "who shares the cost is not stated",
    "split_unspecified": "how to split it is not stated",
    "split_count_mismatch": "head count does not match the people named",
    "split_count_not_found": "head count not found in the message",
    "suspicious_instruction_text": "message contains instruction-like text",
}


@dataclass(frozen=True)
class Entry:
    """One line in a person's balance. Positive = they are owed; negative = they owe."""

    person: str
    delta_paise: int
    kind: str                       # "paid", "share", "sent" or "received"
    record_id: int
    description: str
    source_msg_ids: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class Transfer:
    """One payment that settles up: `debtor` pays `creditor`."""

    debtor: str
    creditor: str
    amount_paise: int


@dataclass(frozen=True)
class Excluded:
    """A record we did not count, and why."""

    record_id: int
    kind: str              # "expense" or "settlement"
    description: str
    amount_paise: int
    reason: str
    needs_review: bool     # True = a person should decide; False = rightly ignored


@dataclass
class Ledger:
    balances: dict[str, int] = field(default_factory=dict)
    entries: list[Entry] = field(default_factory=list)
    transfers: list[Transfer] = field(default_factory=list)
    excluded: list[Excluded] = field(default_factory=list)

    def entries_for(self, person: str) -> list[Entry]:
        return [entry for entry in self.entries if entry.person == person]

    @property
    def needs_review(self) -> list[Excluded]:
        return [item for item in self.excluded if item.needs_review]


# =============================================================================
# Showing money
# =============================================================================

def rupees(paise: int) -> str:
    """123456789 paise -> '₹12,34,567.89' (Indian digit grouping)."""
    sign = "-" if paise < 0 else ""
    whole, part = divmod(abs(paise), 100)
    return f"{sign}₹{_indian_grouping(whole)}.{part:02d}"


def _indian_grouping(number: int) -> str:
    """1234567 -> '12,34,567': the last 3 digits, then groups of 2."""
    digits = str(number)
    if len(digits) <= 3:
        return digits

    last_three = digits[-3:]
    rest = digits[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups + [last_three])


# =============================================================================
# The calculation
# =============================================================================

def split_equally(amount_paise: int, people: list[str]) -> dict[str, int]:
    """Split exactly. ₹100.00 between 3 -> 3334, 3333, 3333 paise (first name gets the extra)."""
    ordered = sorted(people)
    each, leftover = divmod(amount_paise, len(ordered))
    shares = {}
    for position, person in enumerate(ordered):
        extra = 1 if position < leftover else 0
        shares[person] = each + extra
    return shares


def _why_not_counted(expense: Expense) -> tuple[str, bool] | None:
    """If this expense shouldn't be counted, return (reason, needs_review). Else None."""
    if expense.status is ExpenseStatus.PROPOSED:
        return "not confirmed (only proposed)", False
    if expense.status is ExpenseStatus.CANCELLED:
        return "cancelled", False
    if expense.status is ExpenseStatus.DISPUTED:
        return "disputed in the chat", True

    blocking_reasons = [BLOCKING_FLAGS[flag] for flag in expense.flags if flag in BLOCKING_FLAGS]
    if blocking_reasons:
        return "; ".join(blocking_reasons), True

    if not expense.participants:
        return "who shares the cost is not stated", True
    return None


def compute_ledger(expenses: list[Stored[Expense]], settlements: list[Stored[Settlement]]) -> Ledger:
    ledger = Ledger()
    balance: dict[str, int] = {}

    def add(person: str, paise: int) -> None:
        balance[person] = balance.get(person, 0) + paise

    # --- Expenses: the payer is owed the total; each participant owes their share.
    for stored in expenses:
        expense = stored.record

        problem = _why_not_counted(expense)
        if problem:
            reason, needs_review = problem
            ledger.excluded.append(Excluded(
                stored.id, "expense", expense.description, expense.amount_paise, reason, needs_review,
            ))
            continue

        if expense.split is SplitType.EXACT:
            shares = expense.shares_paise
            how = "exact share"
        else:
            shares = split_equally(expense.amount_paise, expense.participants)
            how = f"{rupees(expense.amount_paise)} ÷ {len(shares)}"

        note = ""
        if "participants_inferred" in expense.flags:
            note = "participants inferred from head count"
        sources = tuple(expense.source_msg_ids)

        add(expense.payer, expense.amount_paise)
        ledger.entries.append(Entry(
            expense.payer, expense.amount_paise, "paid", stored.id, expense.description, sources, note,
        ))
        for person, share in shares.items():
            add(person, -share)
            ledger.entries.append(Entry(
                person, -share, "share", stored.id, expense.description, sources, how,
            ))

    # --- Settlements: paying someone back moves money the other way.
    for stored in settlements:
        settlement = stored.record

        blocking_reasons = [BLOCKING_FLAGS[f] for f in settlement.flags if f in BLOCKING_FLAGS]
        if blocking_reasons:
            ledger.excluded.append(Excluded(
                stored.id, "settlement",
                f"{settlement.from_person} → {settlement.to_person}",
                settlement.amount_paise, "; ".join(blocking_reasons), True,
            ))
            continue

        if settlement.status is SettlementStatus.CLAIMED:
            label = "claimed by sender"
        else:
            label = "confirmed"
        sources = tuple(settlement.source_msg_ids)

        add(settlement.from_person, settlement.amount_paise)
        add(settlement.to_person, -settlement.amount_paise)
        ledger.entries.append(Entry(
            settlement.from_person, settlement.amount_paise, "sent", stored.id,
            f"payment to {settlement.to_person}", sources, label,
        ))
        ledger.entries.append(Entry(
            settlement.to_person, -settlement.amount_paise, "received", stored.id,
            f"payment from {settlement.from_person}", sources, label,
        ))

    # --- Safety check: money can't appear or disappear.
    if sum(balance.values()) != 0:
        raise AssertionError("ledger does not balance")

    ledger.balances = dict(sorted(balance.items()))
    ledger.transfers = settle_up(ledger.balances)
    return ledger


def settle_up(balances: dict[str, int]) -> list[Transfer]:
    """Find a few payments that bring everyone to zero.

    Repeatedly: the person who owes the most pays the person who is owed the most.
    """
    # [amount, name] lists, biggest amount first (ties: alphabetical).
    owes = sorted([[-amount, person] for person, amount in balances.items() if amount < 0],
                  key=lambda item: (-item[0], item[1]))
    owed = sorted([[amount, person] for person, amount in balances.items() if amount > 0],
                  key=lambda item: (-item[0], item[1]))

    transfers: list[Transfer] = []
    i = 0
    j = 0
    while i < len(owes) and j < len(owed):
        amount = min(owes[i][0], owed[j][0])
        transfers.append(Transfer(debtor=owes[i][1], creditor=owed[j][1], amount_paise=amount))
        owes[i][0] -= amount
        owed[j][0] -= amount
        if owes[i][0] == 0:
            i += 1
        if owed[j][0] == 0:
            j += 1
    return transfers


# =============================================================================
# Explaining one person's balance
# =============================================================================

def explain(ledger: Ledger, person: str, messages: dict[str, Message], tz: ZoneInfo) -> str:
    """Every line that makes up a person's balance, with the messages it came from."""
    balance = ledger.balances.get(person, 0)
    if balance > 0:
        lines = [f"{person} is owed {rupees(balance)}"]
    elif balance < 0:
        lines = [f"{person} owes {rupees(-balance)}"]
    else:
        lines = [f"{person} is settled"]

    for entry in ledger.entries_for(person):
        sign = "+" if entry.delta_paise > 0 else "−"

        if entry.kind == "paid":
            what = f'paid for "{entry.description}"'
            if entry.note:
                what += f" [{entry.note}]"
        elif entry.kind == "share":
            what = f'share of "{entry.description}" ({entry.note})'
        else:  # "sent" or "received"
            what = f"{entry.description} ({entry.note})"

        lines.append(f"  {sign} {rupees(abs(entry.delta_paise)):>14}  {what}")

        for msg_id in entry.source_msg_ids:
            message = messages.get(msg_id)
            if message:
                when = message.timestamp.astimezone(tz).strftime("%d %b %H:%M")
                lines.append(f'      ↳ {when} {message.sender}: "{_shorten(message.text)}" [{msg_id[:8]}]')

    lines.append(f"  = {rupees(balance):>15}")
    return "\n".join(lines)


def _shorten(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def ledger_for_chat(db: Database, chat_id: int) -> Ledger:
    """The ledger for a stored chat."""
    return compute_ledger(db.records(chat_id, "expense"), db.records(chat_id, "settlement"))
