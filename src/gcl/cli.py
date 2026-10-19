"""Command line: gcl import | chats | balances | plans | ask | delete"""

from __future__ import annotations

import argparse
import logging
import sys

from gcl.config import get_settings
from gcl.db import Database
from gcl.extractor import LLMError
from gcl.llm import client_from_settings
from gcl.ledger import explain, ledger_for_chat, rupees
from gcl.parser import ParseError
from gcl.pipeline import import_chat
from gcl.qa import answer_question


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _chat_id(db: Database, name: str) -> int:
    chat_id = db.find_chat(name)
    if chat_id is None:
        raise SystemExit(f"No chat named {name!r}. Run `gcl chats` to see imported chats.")
    return chat_id


def cmd_import(args, settings, db) -> int:
    print(f"Message text from this chat will be read by {settings.llm_label}.")
    if not settings.is_local_llm and not _confirm("Continue?", args.yes):
        print("Cancelled. Nothing was sent or stored.")
        return 1
    report = import_chat(args.file, chat_name=args.chat, db=db, client=client_from_settings(settings),
                         settings=settings)
    print(f"Messages: {report.messages_total} ({report.messages_new} new)")
    print(f"Windows sent to the LLM: {report.windows_sent}, failed: {len(report.failed_windows)}")
    print(f"Records: {report.records_new} new, {report.records_merged} merged, "
          f"{report.possible_duplicates} possible duplicates, {report.rejected} rejected")
    if report.failed_windows:
        print("Some windows failed; run the same import again later to retry them.")
    print("Tip: you can now delete the exported chat file; everything needed is stored.")
    return 0


def cmd_chats(args, settings, db) -> int:
    chats = db.list_chats()
    if not chats:
        print("No chats imported yet.")
    for c in chats:
        print(f"{c.name}: {c.messages} messages, {c.records} records")
    return 0


def cmd_balances(args, settings, db) -> int:
    chat_id = _chat_id(db, args.chat)
    ledger = ledger_for_chat(db, chat_id)
    if args.explain:
        messages = {m.msg_id: m for m in db.get_messages(chat_id)}
        print(explain(ledger, args.explain, messages, settings.tz))
        return 0
    print("Who pays whom:" if ledger.transfers else "Everyone is settled.")
    for t in ledger.transfers:
        print(f"  {t.debtor} → {t.creditor}: {rupees(t.amount_paise)}")
    if ledger.needs_review:
        print("\nNeeds review (not counted):")
        for x in ledger.needs_review:
            print(f"  #{x.record_id} {x.kind} {rupees(x.amount_paise)} \"{x.description}\": {x.reason}")
    print("\nSee the calculation: gcl balances --chat NAME --explain PERSON")
    return 0


def cmd_plans(args, settings, db) -> int:
    chat_id = _chat_id(db, args.chat)
    print("Commitments:")
    for s in db.records(chat_id, "commitment"):
        c = s.record
        due = f" (due {c.due_date}{' ' + str(c.due_time)[:5] if c.due_time else ''})" if c.due_date else ""
        print(f"  [{c.status.value}] {c.owner}: {c.action}{due}")
    print("Events:")
    for s in db.records(chat_id, "event"):
        e = s.record
        when = str(e.on_date) if e.on_date else "date unclear"
        if e.end_date:
            when += f" to {e.end_date}"
        if e.at_time:
            when += f" {str(e.at_time)[:5]}"
        print(f"  [{e.status.value}] {e.title}: {when}")
    return 0


def cmd_ask(args, settings, db) -> int:
    chat_id = _chat_id(db, args.chat)
    print(f"Searching this chat with {settings.llm_label}...")
    result = answer_question(db, chat_id, args.question, client_from_settings(settings), settings.tz,
                             max_chars=settings.qa_max_chars)
    print(f"\n{result.answer}\n")
    if result.citations:
        print("Sources:")
        for m in result.citations:
            print(f"  {m.timestamp.astimezone(settings.tz):%d %b %Y %H:%M} {m.sender}: {m.text[:100]}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gcl", description="Group Chat Ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import", help="import a WhatsApp export (.txt or .zip)")
    p.add_argument("file")
    p.add_argument("--chat", required=True, help="a name for this chat, e.g. 'Goa Trip'")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("chats", help="list imported chats")
    p.set_defaults(func=cmd_chats)

    p = sub.add_parser("balances", help="who owes whom")
    p.add_argument("--chat", required=True)
    p.add_argument("--explain", metavar="PERSON", help="show how PERSON's balance was calculated")
    p.set_defaults(func=cmd_balances)

    p = sub.add_parser("plans", help="commitments and events")
    p.add_argument("--chat", required=True)
    p.set_defaults(func=cmd_plans)

    p = sub.add_parser("ask", help="ask a question about the chat")
    p.add_argument("--chat", required=True)
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("delete", help="delete a chat and all its data")
    p.add_argument("--chat", required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_delete)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        with Database(settings.db_path) as db:
            return args.func(args, settings, db)
    except (ParseError, LLMError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())