"""One command to go from a WhatsApp export to asking questions about it.

    1. Put your export (.txt or .zip) in the chats/ folder
    2. python3 run.py

First run sets up .venv. With GCL_LLM_PROVIDER=ollama everything runs on this computer;
with gemini (the default) it asks for your API key if .env has none. Re-running is safe:
messages already imported are never sent to the model again.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHATS = ROOT / "chats"
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ENV_FILE = ROOT / ".env"

HELP = """\
Type a question, or one of:
  balances          who pays whom
  explain NAME      how NAME's balance was calculated
  plans             commitments and events
  help              show this again
  quit              exit"""

QUOTA_HINT = ("Your Gemini API quota is used up (429 RESOURCE_EXHAUSTED). "
              "Wait a minute, or until the daily quota resets, and try again.")


# --- Setup: run inside .venv with dependencies installed -------------------------------------

def ensure_environment() -> None:
    """Re-launch this script with .venv's Python, creating and installing it if needed."""
    if Path(sys.prefix).resolve() == VENV.resolve():
        try:
            import gcl, google.genai, pydantic_settings  # noqa: F401
            return
        except ImportError:
            pass  # venv exists but the project isn't installed yet
    elif not VENV_PY.exists():
        print("First run: creating .venv (one time only)...")
        venv.create(VENV, with_pip=True)

    if subprocess.run([str(VENV_PY), "-c", "import gcl, google.genai, pydantic_settings"],
                      capture_output=True).returncode != 0:
        print("Installing dependencies (one time only)...")
        subprocess.run([str(VENV_PY), "-m", "pip", "install", "-q", "-e", str(ROOT)], check=True)

    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])


def ensure_api_key() -> None:
    """Ask for the Gemini key once and save it to .env (never committed)."""
    from gcl.config import get_settings

    if get_settings().gemini_api_key and get_settings().gemini_api_key.get_secret_value().strip():
        return
    from getpass import getpass

    print("No Gemini API key found. Get one at https://aistudio.google.com/apikey")
    key = getpass("Paste your key (hidden): ").strip()
    if not key:
        raise SystemExit("No key given. Add GCL_GEMINI_API_KEY to .env and run again.")

    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [line for line in lines if not line.startswith("GCL_GEMINI_API_KEY=")]
    ENV_FILE.write_text("\n".join([f"GCL_GEMINI_API_KEY={key}", *lines]) + "\n", encoding="utf-8")
    ENV_FILE.chmod(0o600)
    get_settings.cache_clear()
    print("Saved to .env.\n")


# --- Choosing the chat -----------------------------------------------------------------------

def chat_name_for(path: Path) -> str:
    """'WhatsApp Chat with Goa Trip.txt' -> 'Goa Trip'."""
    name = path.stem
    for prefix in ("WhatsApp Chat with ", "WhatsApp Chat - "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.strip() or path.stem


def pick_chat_file() -> Path:
    CHATS.mkdir(exist_ok=True)
    # Also one folder deep: macOS unzips an export into "WhatsApp Chat with X/".
    candidates = [*CHATS.glob("*"), *CHATS.glob("*/*")]
    files = sorted(p for p in candidates if p.suffix.lower() in (".txt", ".zip") and p.is_file())
    if not files:
        raise SystemExit(
            f"No chat found. Put your WhatsApp export (.txt or .zip) in:\n  {CHATS}\n"
            "Then run: python3 run.py"
        )
    if len(files) == 1:
        return files[0]
    print("Chats found:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.relative_to(CHATS)}")
    while True:
        choice = input(f"Which one? [1-{len(files)}] ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]


# --- Main flow -------------------------------------------------------------------------------

def main() -> int:
    os.chdir(ROOT)  # .env and data/gcl.db are relative paths
    ensure_environment()

    import logging
    from types import SimpleNamespace

    from gcl.cli import cmd_balances, cmd_plans
    from gcl.config import get_settings
    from gcl.db import Database
    from gcl.extractor import LLMError, build_windows
    from gcl.llm import client_from_settings
    from gcl.parser import ParseError, parse_chat_file
    from gcl.pipeline import import_chat
    from gcl.qa import answer_question

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    path = pick_chat_file()
    name = chat_name_for(path)
    if not get_settings().is_local_llm:
        ensure_api_key()
    settings = get_settings()
    print(f"Model: {settings.llm_label}")

    try:
        parsed = parse_chat_file(path, settings=settings)  # local only, nothing sent yet
    except (ParseError, ValueError) as exc:
        print(f"Could not read {path.name}: {exc}")
        return 1

    with Database(settings.db_path) as db:
        if db.find_chat(name) is None:
            print(f"Chat: {path.name}  ->  saved as {name!r} ({len(parsed.messages)} messages)")
        if db.find_chat(name) is None and not settings.is_local_llm:
            print(f"Message text is sent to {settings.llm_label} to answer questions "
                  "and extract expenses and plans.")
            if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Cancelled. Nothing was sent or stored.")
                return 1

        chat_id = db.get_or_create_chat(name)
        new = db.add_messages(chat_id, parsed.messages)
        print(f"{len(parsed.messages)} messages stored ({len(new)} new).")

        pending = db.pending_message_ids(chat_id)
        calls = sum(1 for w in build_windows(parsed.messages) if any(m.msg_id in pending for m in w.messages))
        if calls:
            print(f"\nExpenses and plans not extracted yet for part of this chat: {calls} model call(s) needed.")
            if settings.is_local_llm:
                print("This runs on your computer and can take a long time for a big chat.")
            print("You can skip this and ask questions right away (1 call per question).")
            if input("Extract now? [y/N] ").strip().lower() in ("y", "yes"):
                print("Extracting. Progress is saved every 10 windows, so Ctrl+C is safe; "
                      "run again later to continue.", flush=True)
                started = time.monotonic()

                def show(done: int, total: int) -> None:
                    left = (time.monotonic() - started) / done * (total - done)
                    print(f"\r  {done}/{total} windows done, about {left / 60:.0f} min left   ",
                          end="", flush=True)

                try:
                    report = import_chat(path, chat_name=name, db=db, client=client_from_settings(settings),
                                         settings=settings, progress=show)
                    print()
                except KeyboardInterrupt:
                    print("\nStopped. Everything up to the last saved batch is kept; run again to continue.")
                else:
                    print(f"{report.records_new} new records, {report.rejected} rejected.")
                    if report.quota_exhausted:
                        print(QUOTA_HINT)
                    if report.failed_windows:
                        print(f"{len(report.failed_windows)} part(s) still pending; run again later to finish.")

        chat = SimpleNamespace(chat=name, explain=None)
        print(f"\n=== {name} ===")
        if db.pending_message_ids(chat_id):
            print("(Part of this chat is not extracted yet, so balances and plans below are incomplete.)")
        cmd_balances(chat, settings, db)
        print()
        cmd_plans(chat, settings, db)
        print(f"\n{HELP}\n")

        client = client_from_settings(settings)
        while True:
            try:
                text = input("Ask> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            command = text.lower()
            if not text:
                continue
            if command in ("quit", "exit", "q"):
                break
            if command == "help":
                print(HELP)
            elif command == "balances":
                cmd_balances(chat, settings, db)
            elif command == "plans":
                cmd_plans(chat, settings, db)
            elif command.startswith("explain "):
                cmd_balances(SimpleNamespace(chat=name, explain=text[8:].strip()), settings, db)
            else:
                try:
                    print("Thinking...", flush=True)
                    result = answer_question(db, chat_id, text, client, settings.tz,
                                             max_chars=settings.qa_max_chars)
                except LLMError as exc:
                    print(f"Error: {exc}")
                    if "429" in str(exc):
                        print(QUOTA_HINT)
                    continue
                print(f"\n{result.answer}")
                for m in result.citations:
                    print(f"  - {m.timestamp.astimezone(settings.tz):%d %b %H:%M} {m.sender}: {m.text[:100]}")
                print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
