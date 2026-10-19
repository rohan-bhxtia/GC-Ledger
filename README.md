# Group Chat Ledger

Turn a WhatsApp group chat export into a structured list of **expenses, settlements, commitments and events**, e.g. "Aman paid ₹1,200 for dinner, split equally" or "Riya will book the cab by Friday".

An LLM (Google Gemini) reads the chat, and **deterministic code checks every answer** before it is accepted. Amounts must appear in the cited messages, people must be chat participants, and shares must add up. Anything that fails is rejected with a reason. Anything uncertain is kept but flagged.

> **Status:** work in progress. Parsing, extraction, validation and the Gemini client are done and tested. Storage (SQLite), balance calculation and the `gcl` CLI are not built yet.

---

## How it works

```
WhatsApp export (.txt / .zip)
        │
        ▼
  parser.py      → list of Message (sender, time, text, kind)
        │
        ▼
  extractor.py   → splits messages into conversation windows
        │            and builds a prompt for each window
        ▼
  gemini.py      → sends the prompt to Gemini, gets structured JSON back
        │
        ▼
  validation.py  → checks every record against the original messages
        │ 
        ▼
  Expenses · Settlements · Commitments · Events · Rejected
```

| Module | What it does |
|---|---|
| [`config.py`](src/gcl/config.py) | All settings, loaded from environment variables / `.env` (prefix `GCL_`). |
| [`models.py`](src/gcl/models.py) | Pydantic schemas: parsed messages, raw LLM output, validated records. |
| [`parser.py`](src/gcl/parser.py) | Reads Android and iOS exports (`.txt` or `.zip`), with DD/MM vs MM/DD auto-detection, multi-line messages, media/deleted/system messages and edited markers. |
| [`extractor.py`](src/gcl/extractor.py) | Groups messages into windows (split on 3h gaps, max 80 messages, 5-message overlap), renders the prompt, and runs extraction through any `LLMClient`. |
| [`gemini.py`](src/gcl/gemini.py) | Gemini implementation of `LLMClient`, with retries on timeouts, rate limits and malformed JSON. |
| [`validation.py`](src/gcl/validation.py) | Rule-based checks on LLM output. Hard failures are rejected; soft problems get `flags`. |

### Supported export formats

```
Android:  14/10/2026, 9:15 pm - Name: text
iOS:      [14/10/26, 9:15:30 PM] Name: text
```

---

## Setup

Requires **Python 3.11+**.

```bash
git clone <repo-url>
cd gc

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Then create your config file:

```bash
cp .env.example .env
```

Edit `.env` and set at least your Gemini API key (get one at https://aistudio.google.com/apikey):

```
GCL_GEMINI_API_KEY=your-key-here
```

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `GCL_GEMINI_API_KEY` | *(none)* | Required only for extraction. Parsing works without it. |
| `GCL_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name. |
| `GCL_LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout (1–600). |
| `GCL_DB_PATH` | `data/gcl.db` | SQLite file location (not used yet). |
| `GCL_DEFAULT_TIMEZONE` | `Asia/Kolkata` | Used to interpret chat timestamps and dates like "tomorrow". |
| `GCL_MAX_IMPORT_MB` | `10` | Chat exports larger than this are rejected (1–100). |
| `GCL_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |

---

## Usage

### Try extraction on a sample chat

```bash
python scripts/smoke_extract.py tests/fixtures/android_dmy_12h.txt
```

This prints every accepted, flagged and rejected record, each with the messages it came from.

By default the script only accepts files inside `tests/fixtures/`, so a real chat is never sent to Gemini by accident. To run it on a real export:

```bash
python scripts/smoke_extract.py path/to/chat.zip --allow-real-chat
```

### Use it from Python

```python
from gcl.config import get_settings
from gcl.extractor import extract
from gcl.gemini import client_from_settings
from gcl.parser import parse_chat_file

settings = get_settings()
parsed = parse_chat_file("chat.txt", settings=settings)
result = extract(parsed, client_from_settings(settings), tz=settings.tz)

for e in result.expenses:
    print(e.payer, e.amount_paise / 100, e.description)
```

Amounts are stored as integer **paise** (₹1 = 100 paise) to avoid floating-point errors.

---

## Running tests

```bash
pytest
```

Tests use a fake LLM client and sample chats in `tests/fixtures/`. They make **no network calls** and need no API key.

Tests that call the real Gemini API are marked `live` and skipped by default. To run them (needs `GCL_GEMINI_API_KEY`):

```bash
pytest -m live
```

---

## Privacy

- **What leaves your machine:** only text messages and media captions, with sender names and timestamps, are sent to Google Gemini during extraction. System messages and deleted messages are never sent.
- **Parsing is fully local.** Nothing is sent when you parse a chat, import a module or create a client.
- **Logs contain counts only**, never message text. Errors report status codes, not prompt or response content.
- **Zip exports are read in memory** and never extracted to disk. Size limits guard against zip bombs.
- `.env`, databases, `chats/` and `*.zip` are git-ignored. **Never commit real chats or your API key.**

---

## Project layout

```
.
├── pyproject.toml
├── .env.example
├── scripts/
│   └── smoke_extract.py     # manual end-to-end run against Gemini
├── src/gcl/
│   ├── config.py
│   ├── models.py
│   ├── parser.py
│   ├── extractor.py
│   ├── gemini.py
│   └── validation.py
└── tests/
    ├── fixtures/            # sample Android, iOS and malformed chats
    ├── test_config.py
    ├── test_parser.py
    ├── test_extractor.py
    ├── test_validation.py
    └── test_gemini.py
```

## Roadmap

- [x] Parse Android / iOS exports
- [x] LLM extraction with windowing
- [x] Deterministic validation of LLM output
- [x] Gemini client
- [ ] SQLite storage (`db.py`)
- [ ] Balances: who owes whom (`ledger.py`)
- [ ] `gcl` command-line tool (`cli.py`)
