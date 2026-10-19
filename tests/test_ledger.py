"""Ledger tests. No LLM anywhere: balances must be correct from records alone."""

import random
from zoneinfo import ZoneInfo

import pytest

from gcl.db import Database, Stored
from gcl.ledger import compute_ledger, explain, ledger_for_chat, rupees, settle_up, split_equally
from gcl.models import Expense, ExpenseStatus, Settlement, SettlementStatus, SplitType
from gcl.parser import parse_chat_text

IST = ZoneInfo("Asia/Kolkata")
GROUP = ["Aman", "Karan", "Riya"]
_ids = iter(range(1, 10_000))


def exp(payer, amount, participants=GROUP, status=ExpenseStatus.CONFIRMED, split=SplitType.EQUAL,
        shares=None, flags=(), desc="x"):
    return Stored(next(_ids), Expense(
        payer=payer, amount_paise=amount, description=desc, participants=participants,
        split=split, shares_paise=shares or {}, status=status,
        source_msg_ids=["a" * 16], flags=list(flags)))


def pay(sender, receiver, amount, status=SettlementStatus.CONFIRMED, flags=()):
    return Stored(next(_ids), Settlement(
        from_person=sender, to_person=receiver, amount_paise=amount, status=status,
        source_msg_ids=["b" * 16], flags=list(flags)))


# --- Money basics -------------------------------------------------------------------------

@pytest.mark.parametrize("paise,text", [
    (0, "₹0.00"), (5, "₹0.05"), (1_200_000, "₹12,000.00"), (12_000_000, "₹1,20,000.00"),
    (-40_000, "-₹400.00"), (123_456_789, "₹12,34,567.89"),
])
def test_rupees_indian_format(paise, text):
    assert rupees(paise) == text


def test_equal_split_never_loses_a_paisa():
    shares = split_equally(10_000, ["Riya", "Aman", "Karan"])   # ₹100 / 3
    assert shares == {"Aman": 3_334, "Karan": 3_333, "Riya": 3_333}
    assert sum(shares.values()) == 10_000


# --- Balances -------------------------------------------------------------------------------

def test_villa_split_three_ways():
    ledger = compute_ledger([exp("Karan", 1_200_000)], [])
    assert ledger.balances == {"Aman": -400_000, "Karan": 800_000, "Riya": -400_000}
    assert {(t.debtor, t.creditor, t.amount_paise) for t in ledger.transfers} == {
        ("Aman", "Karan", 400_000), ("Riya", "Karan", 400_000)}


def test_exact_shares():
    e = exp("Karan", 50_000, participants=["Aman", "Riya"], split=SplitType.EXACT,
            shares={"Aman": 20_000, "Riya": 30_000})
    assert compute_ledger([e], []).balances == {"Aman": -20_000, "Karan": 50_000, "Riya": -30_000}


def test_payer_not_sharing_the_cost():
    ledger = compute_ledger([exp("Aman", 60_000, participants=["Riya"])], [])
    assert ledger.balances == {"Aman": 60_000, "Riya": -60_000}


def test_settlement_clears_debt():
    ledger = compute_ledger([exp("Karan", 1_200_000)], [pay("Riya", "Karan", 400_000)])
    assert ledger.balances["Riya"] == 0
    assert [(t.debtor, t.amount_paise) for t in ledger.transfers] == [("Aman", 400_000)]


def test_claimed_settlement_counts_but_is_labelled():
    ledger = compute_ledger([exp("Karan", 1_200_000)],
                            [pay("Riya", "Karan", 400_000, status=SettlementStatus.CLAIMED)])
    assert ledger.balances["Riya"] == 0
    assert all(e.note == "claimed by sender" for e in ledger.entries if e.kind in ("sent", "received"))


def test_debts_are_simplified():
    # Aman owes Karan 100, Karan owes Riya 100  ->  Aman pays Riya directly.
    ledger = compute_ledger([
        exp("Karan", 10_000, participants=["Aman"]),
        exp("Riya", 10_000, participants=["Karan"]),
    ], [])
    assert [(t.debtor, t.creditor, t.amount_paise) for t in ledger.transfers] == [("Aman", "Riya", 10_000)]


# --- What does NOT count ---------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,review", [
    ({"status": ExpenseStatus.PROPOSED}, False),
    ({"status": ExpenseStatus.CANCELLED}, False),
    ({"status": ExpenseStatus.DISPUTED}, True),
    ({"flags": ["possible_duplicate"]}, True),
    ({"flags": ["participants_unspecified"], "participants": None}, True),
    ({"flags": ["split_unspecified"]}, True),
    ({"flags": ["suspicious_instruction_text"]}, True),
])
def test_uncertain_expenses_are_never_counted(kwargs, review):
    ledger = compute_ledger([exp("Karan", 1_200_000, **kwargs)], [])
    assert ledger.balances == {}
    assert len(ledger.excluded) == 1
    assert ledger.excluded[0].needs_review is review


def test_inferred_participants_count_and_are_noted():
    ledger = compute_ledger([exp("Karan", 1_200_000, flags=["participants_inferred"])], [])
    assert ledger.balances["Karan"] == 800_000
    assert ledger.entries_for("Karan")[0].note == "participants inferred from head count"


def test_flagged_settlement_needs_review():
    ledger = compute_ledger([], [pay("Riya", "Karan", 400_000, flags=["suspicious_instruction_text"])])
    assert ledger.balances == {} and ledger.needs_review[0].kind == "settlement"


# --- Invariants (randomised) ----------------------------------------------------------------------

def test_random_ledgers_always_balance_and_settle():
    rng = random.Random(42)
    people = ["A", "B", "C", "D", "E"]
    for _ in range(300):
        expenses = [
            exp(rng.choice(people), rng.randint(1, 5_000_000),
                participants=rng.sample(people, rng.randint(1, 5)))
            for _ in range(rng.randint(1, 8))
        ]
        settlements = [pay(*rng.sample(people, 2), rng.randint(1, 1_000_000)) for _ in range(rng.randint(0, 4))]
        ledger = compute_ledger(expenses, settlements)
        assert sum(ledger.balances.values()) == 0
        # Applying the suggested transfers leaves everyone at exactly zero.
        after = dict(ledger.balances)
        for t in ledger.transfers:
            after[t.debtor] += t.amount_paise
            after[t.creditor] -= t.amount_paise
        assert all(v == 0 for v in after.values())
        # Each person's entries add up to their balance (explanations reconcile).
        for person, balance in ledger.balances.items():
            assert sum(e.delta_paise for e in ledger.entries_for(person)) == balance


def test_settle_up_on_already_settled_group():
    assert settle_up({"A": 0, "B": 0}) == []


# --- Explanation & DB integration -------------------------------------------------------------

CHAT = """\
14/10/2026, 9:07 pm - Karan: Booked the villa, paid ₹12,000 split 3 ways pls
14/10/2026, 9:10 pm - Riya: Sent you 4000
14/10/2026, 9:11 pm - Aman: Riya hi
"""


def test_explanation_cites_messages_and_reconciles():
    msgs = parse_chat_text(CHAT, tz=IST).messages
    villa = Stored(1, Expense(payer="Karan", amount_paise=1_200_000, description="villa",
                              participants=GROUP, split=SplitType.EQUAL, status=ExpenseStatus.CONFIRMED,
                              source_msg_ids=[msgs[0].msg_id], flags=["participants_inferred"]))
    ledger = compute_ledger([villa], [])
    text = explain(ledger, "Riya", {m.msg_id: m for m in msgs}, IST)
    assert text.splitlines()[0] == "Riya owes ₹4,000.00"
    assert 'share of "villa" (₹12,000.00 ÷ 3)' in text
    assert "14 Oct 21:07 Karan" in text
    assert " ".join(text.splitlines()[-1].split()) == "= -₹4,000.00"


def test_ledger_from_database():
    with Database(":memory:") as db:
        msgs = parse_chat_text(CHAT, tz=IST).messages
        chat_id = db.get_or_create_chat("Goa")
        db.add_messages(chat_id, msgs)
        db.save_records(chat_id, "expense", [Expense(
            payer="Karan", amount_paise=1_200_000, description="villa", participants=GROUP,
            split=SplitType.EQUAL, status=ExpenseStatus.CONFIRMED, source_msg_ids=[msgs[0].msg_id])])
        db.save_records(chat_id, "settlement", [Settlement(
            from_person="Riya", to_person="Karan", amount_paise=400_000,
            status=SettlementStatus.CLAIMED, source_msg_ids=[msgs[1].msg_id])])
        ledger = ledger_for_chat(db, chat_id)
    assert ledger.balances == {"Aman": -400_000, "Karan": 400_000, "Riya": 0}
    assert [(t.debtor, t.creditor, t.amount_paise) for t in ledger.transfers] == [("Aman", "Karan", 400_000)]