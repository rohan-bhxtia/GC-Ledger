"""The `gcl` command, for running one step at a time:

    gcl import FILE --chat NAME     read a chat export and extract records
    gcl chats                       list imported chats
    gcl balances --chat NAME        who pays whom   (add --explain PERSON for the details)
    gcl plans --chat NAME           commitments and events
    gcl ask --chat NAME "question"  ask about the chat
    gcl delete --chat NAME          delete a chat and everything from it

run.py uses cmd_balances() and cmd_plans() from here to print the same output.
"""

from __future__ import annotations

import argparse
import logging
import sys

from gcl.config import get_settings
from gcl.db import Database
from gcl.extractor import LLMError
from gcl.ledger import explain, ledger_for_chat, rupees
from gcl.llm import client_from_settings
from gcl.parser import ParseError
from gcl.pipeline import import_chat
from gcl.qa import answer_question


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(f"{question} [y/N] ").strip().lower()
    return answer in ("y", "yes")


def _chat_id(db: Database, name: str) -> int:
    chat_id = db.find_chat(name)
    if chat_id is None:
        raise SystemExit(f"No chat named {name!r}. Run `gcl chats` to see imported chats.")
    return chat_id


def _time_text(value) -> str:
    """A time like 18:30:00 -> '18:30'."""
    return str(value)[:5]


# -----------------------------------------------------------------------------
# The commands
# -----------------------------------------------------------------------------

def cmd_import(args, settings, db) -> int:
    print(f"Message text from this chat will be read by {settings.llm_label}.")
    # Only Gemini sends the chat off this computer, so only Gemini needs a "yes".
    if not settings.is_local_llm and not _confirm("Continue?", args.yes):
        print("Cancelled. Nothing was sent or stored.")
        return 1

    report = import_chat(
        args.file,
        chat_name=args.chat,
        db=db,
        client=client_from_settings(settings),
        settings=settings,
    )
    print(f"Messages: {report.messages_total} ({report.messages_new} new)")
    print(f"Windows sent to the LLM: {report.windows_sent}, failed: {len(report.failed_windows)}")
    print(f"Records: {report.records_new} new, {report.records_merged} merged, {report.rejected} rejected")
    if report.failed_windows:
        print("Some windows failed; run the same import again later to retry them.")
    print("Tip: you can now delete the exported chat file; everything needed is stored.")
    return 0


def cmd_chats(args, settings, db) -> int:
    chats = db.list_chats()
    if not chats:
        print("No chats imported yet.")
    for chat in chats:
        print(f"{chat.name}: {chat.messages} messages, {chat.records} records")
    return 0


def cmd_balances(args, settings, db) -> int:
    chat_id = _chat_id(db, args.chat)
    ledger = ledger_for_chat(db, chat_id)

    # "--explain PERSON": show how one person's balance was worked out.
    if args.explain:
        messages = {m.msg_id: m for m in db.get_messages(chat_id)}
        print(explain(ledger, args.explain, messages, settings.tz))
        return 0

    if ledger.transfers:
        print("Who pays whom:")
    else:
        print("Everyone is settled.")
    for transfer in ledger.transfers:
        print(f"  {transfer.debtor} → {transfer.creditor}: {rupees(transfer.amount_paise)}")

    if ledger.needs_review:
        print("\nNeeds review (not counted):")
        for item in ledger.needs_review:
            print(f'  #{item.record_id} {item.kind} {rupees(item.amount_paise)} '
                  f'"{item.description}": {item.reason}')

    print("\nSee the calculation: gcl balances --chat NAME --explain PERSON")
    return 0


def cmd_plans(args, settings, db) -> int:
    chat_id = _chat_id(db, args.chat)

    print("Commitments:")
    for stored in db.records(chat_id, "commitment"):
        commitment = stored.record
        due = ""
        if commitment.due_date:
            due = f" (due {commitment.due_date}"
            if commitment.due_time:
                due += " " + _time_text(commitment.due_time)
            due += ")"
        print(f"  [{commitment.status.value}] {commitment.owner}: {commitment.action}{due}")

    print("Events:")
    for stored in db.records(chat_id, "event"):
        event = stored.record
        if event.on_date:
            when = str(event.on_date)
        else:
            when = "date unclear"
        if event.end_date:
            when += f" to {event.end_date}"
        if event.at_time:
            when += " " + _time_text(event.at_time)
        print(f"  [{event.status.value}] {event.title}: {when}")
    return 0


def cmd_ask(args, settings, db) -> int:
    chat_id = _chat_id(db, args.chat)
    print(f"Searching this chat with {settings.llm_label}...")
    result = answer_question(
        db, chat_id, args.question, client_from_settings(settings), settings.tz,
        max_chars=settings.qa_max_chars,
    )

    print(f"\n{result.answer}\n")
    if result.citations:
        print("Sources:")
        for message in result.citations:
            when = message.timestamp.astimezone(settings.tz).strftime("%d %b %Y %H:%M")
            print(f"  {when} {message.sender}: {message.text[:100]}")
    elif not result.found:
        print("(The chat does not seem to contain this information.)")
    return 0


def cmd_delete(args, settings, db) -> int:
    chat_id = _chat_id(db, args.chat)
    if not _confirm(f"Permanently delete {args.chat!r} and everything extracted from it?", args.yes):
        print("Cancelled.")
        return 1
    db.delete_chat(chat_id)
    print(f"Deleted {args.chat!r}.")
    return 0


# -----------------------------------------------------------------------------
# Reading the command line
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gcl", description="Group Chat Ledger")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("import", help="import a WhatsApp export (.txt or .zip)")
    p.add_argument("file")
    p.add_argument("--chat", required=True, help="a name for this chat, e.g. 'Goa Trip'")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_import)

    p = commands.add_parser("chats", help="list imported chats")
    p.set_defaults(func=cmd_chats)

    p = commands.add_parser("balances", help="who owes whom")
    p.add_argument("--chat", required=True)
    p.add_argument("--explain", metavar="PERSON", help="show how PERSON's balance was calculated")
    p.set_defaults(func=cmd_balances)

    p = commands.add_parser("plans", help="commitments and events")
    p.add_argument("--chat", required=True)
    p.set_defaults(func=cmd_plans)

    p = commands.add_parser("ask", help="ask a question about the chat")
    p.add_argument("--chat", required=True)
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = commands.add_parser("delete", help="delete a chat and all its data")
    p.add_argument("--chat", required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_delete)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    # These libraries log every web request; only show their warnings.
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        with Database(settings.db_path) as db:
            return args.func(args, settings, db)
    except (ParseError, LLMError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
