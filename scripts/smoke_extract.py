"""Run extraction on a chat and print what was accepted, flagged and rejected.

    python scripts/smoke_extract.py tests/fixtures/android_dmy_12h.txt

Sends message text to Google Gemini. Restricted to tests/fixtures/ unless you pass
--allow-real-chat, so a real conversation is never sent by accident.
"""

import argparse
import logging
import sys
from pathlib import Path

from gcl.config import get_settings
from gcl.extractor import build_windows, extract
from gcl.gemini import client_from_settings
from gcl.parser import ParseError, parse_chat_file

FIXTURES = (Path(__file__).resolve().parent.parent / "tests" / "fixtures").resolve()


def rupees(paise: int) -> str:
    return f"₹{paise / 100:,.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chat", type=Path)
    ap.add_argument("--allow-real-chat", action="store_true")
    args = ap.parse_args()

    path = args.chat.resolve()
    if not args.allow_real_chat and FIXTURES not in path.parents:
        print("Refusing: only files in tests/fixtures/ unless --allow-real-chat is given.")
        return 2

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "google_genai"):  # per-request SDK chatter
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        parsed = parse_chat_file(path, settings=settings)
        client = client_from_settings(settings)
    except (ParseError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    windows = build_windows(parsed.messages)
    sent = sum(len(w.all_messages) for w in windows)
    print(f"Sending {sent} messages in {len(windows)} windows to Gemini ({settings.gemini_model})...\n")
    result = extract(parsed, client, tz=settings.tz)
    text_of = {m.msg_id: m.text.replace("\n", " / ") for m in parsed.messages}

    def src(ids):
        return " | ".join(f'"{text_of[i][:60]}"' for i in ids)

    print("EXPENSES")
    for e in result.expenses:
        print(f"  {e.payer} paid {rupees(e.amount_paise)} [{e.status.value}, split={e.split.value}] "
              f"for {e.participants} flags={e.flags}\n    <- {src(e.source_msg_ids)}")
    print("SETTLEMENTS")
    for s in result.settlements:
        print(f"  {s.from_person} -> {s.to_person} {rupees(s.amount_paise)} [{s.status.value}]\n"
              f"    <- {src(s.source_msg_ids)}")
    print("COMMITMENTS")
    for c in result.commitments:
        print(f"  {c.owner}: {c.action} [{c.status.value}] due={c.due_date} {c.due_time or ''} "
              f"flags={c.flags}\n    <- {src(c.source_msg_ids)}")
    print("EVENTS")
    for ev in result.events:
        print(f"  {ev.title} [{ev.status.value}] {ev.on_date} {ev.at_time or ''} flags={ev.flags}\n"
              f"    <- {src(ev.source_msg_ids)}")
    print("REJECTED")
    for r in result.rejected:
        print(f"  {r.kind}: {r.reason}")
    print(f"\nwindows={result.windows} failed={result.failed_windows} skipped_overlap={result.skipped_overlap}")
    return 0 if not result.failed_windows else 1


if __name__ == "__main__":
    sys.exit(main())