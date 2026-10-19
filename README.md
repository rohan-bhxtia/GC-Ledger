# Group Chat Ledger

**Ask questions about a WhatsApp chat, and get balances you can trust.**

Group Chat Ledger reads a WhatsApp export and turns it into a ledger of **expenses, settlements, commitments and events**. You can then ask questions in plain English or Hinglish, and every answer cites the exact messages it came from.

```text
Ask> who still owes money for the Goa trip?
Riya owes Karan ₹4,000 for the villa.
  - 14 Oct 21:07 Karan: Booked the villa, paid ₹12,000 / split 3 ways pls
```

An LLM (Google Gemini, or a local model through Ollama) reads the chat. **Deterministic Python code then checks every record before it's accepted**, and **does all the money maths**. The LLM never adds, splits or settles an amount.

---

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Choosing a model: Gemini or Ollama](#choosing-a-model-gemini-or-ollama)
- [Using it](#using-it)
- [How it works](#how-it-works)
- [Why you can trust the numbers](#why-you-can-trust-the-numbers)
- [Privacy and security](#privacy-and-security)
- [Configuration](#configuration)
- [Development](#development)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Features

- **One-command setup.** `python3 run.py` creates the virtualenv, installs dependencies, imports your chat and opens a question prompt.
- **Questions with citations.** Every answer lists the messages it's based on. Citations the model makes up are dropped, and an answer with no real source is marked as not found.
- **Who owes whom.** Balances are computed in integer paise, with the fewest transfers needed to settle up, and `explain NAME` shows the calculation step by step.
- **Plans and promises.** "I'll book the cab by tomorrow" becomes a commitment with an owner, a status and a due date. "Goa 20–23 Dec" becomes a multi-day event.
- **Hinglish-aware.** It understands mixed Hindi and English ("kal tak", "teen log", "2 lakh"), and resolves relative dates against each message's timestamp.
- **Cloud or local.** Use Google Gemini for the best quality, or Ollama for a fully offline setup where nothing leaves your computer.
- **Safe to re-run.** Re-importing the same or a longer export never double-counts. Only new messages go to the model, and failed or interrupted work picks up where it left off.
- **Android and iOS exports**, as `.txt` or `.zip`, in either date order (detected automatically).

---

## Quick start

Requires **Python 3.11+**. On macOS that's `python3`.

**1. Export the chat.** In WhatsApp, open the chat → ⋮ (or the contact name on iOS) → **More → Export chat → Without media**.

**2. Put the export in `chats/`**, in the project root:

```text
gc/
├── chats/
│   └── WhatsApp Chat with Goa Trip.txt   ← here (.txt or .zip)
└── run.py
```

A folder that macOS unzipped automatically (`chats/WhatsApp Chat with Goa Trip/…`) works too. `chats/` is git-ignored, so your chat is never committed.

**3. Pick a model** (see the next section). For the quickest start with Gemini, `run.py` asks for your API key on first run. For Ollama, add this to `.env`:

```dotenv
GCL_LLM_PROVIDER=ollama
GCL_OLLAMA_MODEL=qwen3:8b
```

**4. Run it:**

```bash
python3 run.py
```

The first run sets up `.venv` (once). Then it:

1. stores the chat's messages in a local SQLite database;
2. offers to extract expenses and plans (you can skip this and go straight to questions);
3. shows balances, commitments and events;
4. opens the `Ask>` prompt.

The chat name comes from the file name: `WhatsApp Chat with Goa Trip.txt` becomes **Goa Trip**. Keep the same file name when you re-export later, so new messages are added to the same chat.

---

## Choosing a model: Gemini or Ollama

|  | **Google Gemini** (default) | **Ollama** (local) |
|---|---|---|
| Where the chat goes | Google's API | Nowhere: runs on your machine |
| Answer quality | Best | Good on direct questions, weaker on vague ones |
| Speed per question | A few seconds | About 1–2 minutes (8B model, Apple M4) |
| Extracting a large chat | Fast, but limited by your quota | No limits, but slow (about 15 s per window) |
| Cost | Free tier is about 20 requests per model per day | Free |
| Setup | API key | [Install Ollama](https://ollama.com) + `ollama pull qwen3:8b` (about 5 GB) |

### Gemini

1. Get a key at <https://aistudio.google.com/apikey>.
2. Run `python3 run.py` and paste it when asked. It's saved to `.env`, which is git-ignored and readable only by you.

On the free tier, Google limits both requests per minute and requests per day. The client waits as long as Google asks after a per-minute limit. When the daily quota is used up, it stops straight away and tells you when it resets. Work that didn't finish stays pending and is retried on the next run.

### Ollama

```bash
ollama pull qwen3:8b
```

```dotenv
# .env
GCL_LLM_PROVIDER=ollama
GCL_OLLAMA_MODEL=qwen3:8b
GCL_OLLAMA_NUM_CTX=16384
```

**Recommended model:** `qwen3:8b` handles Hinglish and strict JSON well and fits in 16 GB of RAM. Older models such as Llama 3 8B tend to miss expenses and drift off-topic.

**Speed vs coverage:** `GCL_OLLAMA_NUM_CTX` sets how much of the chat each question can see. `8192` is roughly twice as fast; `16384` covers about twice as many messages.

**Large chats:** with a small local context, Q&A sends the most recent messages, plus older messages that share words with your question (with a little context around each). Very long pasted messages are cut short, so a single paste can't crowd out the rest.

---

## Using it

### The interactive prompt (`run.py`)

| Type | What it does |
|---|---|
| *any question* | Answers from the chat, with cited messages |
| `balances` | Who pays whom, plus the expenses that need review |
| `explain NAME` | Every line behind NAME's balance, with source messages |
| `plans` | Commitments and events |
| `help` | Lists these commands |
| `quit` | Exits (Ctrl+D works too) |

**Extraction** finds the expenses and plans behind `balances` and `plans`. It runs in batches of 10 windows and saves after each batch, so **Ctrl+C is always safe**: run `python3 run.py` again and it continues where it stopped. It shows a live progress counter with an estimate of the time left.

Questions work without extraction. They read the stored messages directly.

### The `gcl` command

The same features as one-off commands, for scripting:

```bash
source .venv/bin/activate

gcl import "chats/WhatsApp Chat with Goa Trip.txt" --chat "Goa Trip"
gcl chats                                   # list imported chats
gcl balances --chat "Goa Trip"              # who pays whom
gcl balances --chat "Goa Trip" --explain Riya
gcl plans --chat "Goa Trip"                 # commitments and events
gcl ask --chat "Goa Trip" "when are we leaving?"
gcl delete --chat "Goa Trip"                # permanently remove a chat and everything from it
```

`import` and `delete` ask for confirmation. Pass `--yes` to skip it. With Gemini, `import` also asks before sending anything; with Ollama there's nothing to confirm, because nothing leaves the machine.

### From Python

```python
from gcl.config import get_settings
from gcl.db import Database
from gcl.ledger import ledger_for_chat, rupees
from gcl.llm import client_from_settings
from gcl.pipeline import import_chat
from gcl.qa import answer_question

settings = get_settings()
client = client_from_settings(settings)          # Gemini or Ollama, per GCL_LLM_PROVIDER

with Database(settings.db_path) as db:
    report = import_chat("chat.txt", chat_name="Goa Trip", db=db, client=client, settings=settings)

    for t in ledger_for_chat(db, report.chat_id).transfers:
        print(f"{t.debtor} → {t.creditor}: {rupees(t.amount_paise)}")

    result = answer_question(db, report.chat_id, "who booked the villa?", client, settings.tz,
                             max_chars=settings.qa_max_chars)
    print(result.answer, [m.text for m in result.citations])
```

---

## How it works

```text
 WhatsApp export (.txt / .zip)
          │
          ▼
   parser.py ─────────── messages (sender, time, text, kind)        local, no LLM
          │
          ▼
   db.py ─────────────── SQLite: messages stored once, marked pending
          │
          ▼
   extractor.py ──────── conversation windows → prompt per window
          │
          ▼
   llm.py ─► gemini.py / ollama.py ── structured JSON (schema-constrained)
          │
          ▼
   validation.py ─────── every record checked against the cited messages
          │                 accepted · flagged · rejected (with reason)
          ▼
   db.py ─────────────── records stored, duplicates merged, sources linked
          │
          ├──► ledger.py ── balances, settle-up, step-by-step explanation   pure Python
          └──► qa.py ────── question + relevant messages → cited answer
```

| Module | Responsibility |
|---|---|
| [`run.py`](run.py) | One-command entry point: environment setup, chat discovery, import, interactive prompt |
| [`cli.py`](src/gcl/cli.py) | The `gcl` command: `import`, `chats`, `balances`, `plans`, `ask`, `delete` |
| [`config.py`](src/gcl/config.py) | Settings from environment variables and `.env` (prefix `GCL_`), validated with Pydantic |
| [`parser.py`](src/gcl/parser.py) | Android and iOS exports, `.txt` and `.zip`, DD/MM vs MM/DD detection, multi-line, media, deleted, system and edited messages |
| [`extractor.py`](src/gcl/extractor.py) | Splits the chat into windows (on 3-hour gaps, at most 80 messages, 5 messages of overlap), builds prompts, runs extraction through any `LLMClient` |
| [`llm.py`](src/gcl/llm.py) | Picks the backend from `GCL_LLM_PROVIDER` |
| [`gemini.py`](src/gcl/gemini.py) | Gemini client: structured output, retries, honours Google's retry delay, stops on daily quota |
| [`ollama.py`](src/gcl/ollama.py) | Ollama client: local, strict JSON schema, large context window, clear errors when Ollama isn't running or the model isn't pulled |
| [`validation.py`](src/gcl/validation.py) | Deterministic checks on everything the LLM returns |
| [`db.py`](src/gcl/db.py) | SQLite storage, deduplication, cascading secure delete |
| [`pipeline.py`](src/gcl/pipeline.py) | `import_chat`: parse → store → extract pending windows in saved batches → store records |
| [`ledger.py`](src/gcl/ledger.py) | Balances, minimal settle-up transfers, explanations |
| [`qa.py`](src/gcl/qa.py) | Question answering with verified citations |
| [`models.py`](src/gcl/models.py) | Pydantic schemas: messages, raw LLM output, validated records |

---

## Why you can trust the numbers

LLMs are good at reading messy chat and bad at being reliably right about money. So the model only **reads**, and code **decides**.

### 1. The model copies; code calculates

The model copies amounts **exactly as written** ("₹12,000", "1.5k", "2 lakh"). Code parses them into integer paise (₹1 = 100 paise), so there's no floating-point error. The model is told never to add, multiply or divide.

### 2. Every record is checked against its source messages

| Check | If it fails |
|---|---|
| The amount appears in the cited messages | Rejected |
| Every person is a real chat participant | Rejected |
| The cited message ids exist in that window | Rejected |
| Exact shares add up to the total | Rejected |
| A head count ("split 3 ways") is written in the message | Flagged |
| Dates are real and plausible; an end date isn't before the start | Dropped and flagged |
| A message contains instruction-like text ("ignore previous instructions") | Flagged |

**Rejected** records are listed with a reason and never stored. **Flagged** records are stored, but anything uncertain is kept out of the balances until a person looks at it.

### 3. Conservative ledger rules

- Only **confirmed** expenses count. Proposed and cancelled ones are ignored.
- **Disputed** or flagged expenses (possible duplicate, split unclear, participants unknown) are listed under **"Needs review"** and never silently added.
- "Split 3 ways" fills in the participants **only** if the chat has exactly 3 people. Otherwise nobody is guessed. Inferred participants are labelled as inferred.
- Equal splits divide in paise; leftover paise are assigned deterministically, so totals always match exactly.
- **Invariant:** all balances sum to zero. This is checked on every calculation, and if it ever fails, the ledger refuses to show any numbers.

### 4. Re-imports never double-count

| Situation | Result |
|---|---|
| The same chat imported again | 0 new messages, **0 LLM calls** |
| The same payer and amount from overlapping windows | Merged into one record, sources combined |
| A duplicate backed by a later message (proposed, then "paid") | Status updated to the newer one |
| The same payer and amount within 3 days, no shared message | Kept, flagged `possible_duplicate`, excluded from balances |
| A window failed (timeout, quota) or the run was interrupted | Its messages stay pending and are retried on the next run |

---

## Privacy and security

- **Fully local option.** With `GCL_LLM_PROVIDER=ollama`, no chat content leaves your computer.
- **What Gemini receives:** only text messages and media captions, with sender names and timestamps. System messages and deleted messages are never sent. `run.py` and `gcl import` ask before the first send.
- **Prompt-injection resistant.** Messages are sent as JSON data marked untrusted. The model is told never to follow instructions in them, and instruction-like messages are flagged. Long pasted texts are cut short in Q&A prompts.
- **Your data stays in one file.** `data/gcl.db` is created readable only by you (`0600`).
- **Deletion is real.** `gcl delete` removes a chat, its messages, records, sources and import history. SQLite `secure_delete` plus `VACUUM` means the text isn't left behind in the file, and a test checks this.
- **Parameterised SQL everywhere.** A message like `'); DROP TABLE records; --` is stored as plain text, and a test confirms it.
- **Logs contain counts only**, never message text. API errors report status codes, not prompt or response content.
- **Zip exports are read in memory**, never extracted to disk, with size limits against zip bombs.
- **Git-ignored by default:** `.env`, `data/`, `*.db`, `chats/`, `*.zip`. Never commit a real chat or your API key.

---

## Configuration

All settings are environment variables with the prefix `GCL_`, usually set in `.env`. Copy [`.env.example`](.env.example) to get started.

| Variable | Default | Description |
|---|---|---|
| `GCL_LLM_PROVIDER` | `gemini` | `gemini` (Google API) or `ollama` (local) |
| **Gemini** | | |
| `GCL_GEMINI_API_KEY` | *(none)* | Required only when the provider is `gemini` |
| `GCL_GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model your key can use |
| `GCL_GEMINI_THINKING_LEVEL` | `LOW` | `MINIMAL`, `LOW`, `MEDIUM`, `HIGH`, or empty for the model's default |
| `GCL_LLM_TIMEOUT_SECONDS` | `120` | Request deadline (1–600); it has to cover the model's thinking time |
| `GCL_LLM_MAX_ATTEMPTS` | `3` | Attempts per request (1–10) |
| **Ollama** | | |
| `GCL_OLLAMA_MODEL` | `qwen3:8b` | Any pulled Ollama model |
| `GCL_OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `GCL_OLLAMA_NUM_CTX` | `16384` | Context window in tokens (2048–131072): bigger sees more, but runs slower |
| `GCL_OLLAMA_TIMEOUT_SECONDS` | `600` | Per-request timeout (1–3600) |
| **General** | | |
| `GCL_DB_PATH` | `data/gcl.db` | SQLite database file |
| `GCL_DEFAULT_TIMEZONE` | `Asia/Kolkata` | Used for chat timestamps and dates like "kal" or "Sunday" |
| `GCL_MAX_IMPORT_MB` | `10` | Larger exports are refused (1–100) |
| `GCL_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

---

## Development

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

### Tests

```bash
python -m pytest -q
```

**180 tests**, all offline. They use fake LLM clients and fictional sample chats in `tests/fixtures/`, so they need no API key or Ollama. They cover parsing, windowing, validation rules, deduplication, secure delete, SQL injection, the ledger's zero-sum invariant, retries and quota handling, batched and resumable imports, Q&A citation checks and the CLI end to end.

Tests that call the real Gemini API are marked `live` and skipped by default:

```bash
python -m pytest -m live          # needs GCL_GEMINI_API_KEY
```

### Smoke test against a real model

```bash
python scripts/smoke_extract.py tests/fixtures/android_dmy_12h.txt
```

This prints every accepted, flagged and rejected record, with the messages it came from, using whichever provider `.env` selects. On the fixture, the expected output is:
- Karan paying ₹12,000 for the villa;
- Riya's and Karan's commitments;
- the Goa trip on 20–23 Dec.

The script refuses files outside `tests/fixtures/` unless you pass `--allow-real-chat`, so a real chat is never sent by accident.

### Adding an LLM backend

Implement a single method and return it from [`llm.py`](src/gcl/llm.py):

```python
class LLMClient(Protocol):
    def generate_json(self, *, system: str, user: str, schema: type[BaseModel]) -> str: ...
```

Raise `gcl.extractor.LLMError` on failure, or `QuotaExhausted` when further calls are pointless. Validation, storage and the ledger work unchanged.

---

## Project layout

```text
.
├── run.py                  # one-command entry point
├── pyproject.toml
├── .env.example
├── chats/                  # put WhatsApp exports here (git-ignored)
├── data/                   # SQLite database (git-ignored)
├── scripts/
│   └── smoke_extract.py    # manual end-to-end run against the configured model
├── src/gcl/
│   ├── cli.py              # `gcl` command
│   ├── config.py
│   ├── parser.py
│   ├── extractor.py
│   ├── llm.py              # backend selection
│   ├── gemini.py
│   ├── ollama.py
│   ├── validation.py
│   ├── db.py
│   ├── pipeline.py
│   ├── ledger.py
│   ├── qa.py
│   └── models.py
└── tests/
    ├── fixtures/           # fictional Android, iOS and malformed chats
    └── test_*.py
```

---

## Limitations

- **The free tier is small.** About 20 Gemini requests per model per day is enough for questions, but not for extracting a multi-year chat. Use Ollama or a billed key for that.
- **Local models are slower and less accurate.** Expect 1–2 minutes per question with an 8B model. Answers to vague or inference-heavy questions ("when did we last meet?") are less reliable than Gemini's.
- **Long one-to-one chats make many small windows.** Windows split on 3-hour gaps, so a years-long chat can need hundreds of extraction calls.
- **Q&A sees part of a long chat.** Gemini gets the most recent 1,500 messages; Ollama gets what fits its context window, chosen by recency and keyword match, not semantic search.
- **Text only.** Images, voice notes and documents aren't read; only media captions are.
- **Amounts are in rupees.** Parsing understands ₹, Rs, INR, `k` and `lakh`. Crore isn't supported yet, and amounts above ₹1 crore are treated as errors.

---

## Roadmap

- [x] Android and iOS export parsing
- [x] Windowed LLM extraction with deterministic validation
- [x] Gemini client with quota-aware retries
- [x] SQLite storage, deduplication, secure delete
- [x] Ledger: balances, settle-up and explanations
- [x] Question answering with verified citations
- [x] `gcl` CLI and one-command `run.py`
- [x] Local models through Ollama
- [x] Batched, resumable extraction with progress
- [ ] Semantic search (embeddings) for Q&A over very long chats
- [ ] Larger extraction windows for one-to-one chats, to cut LLM calls
- [ ] Review flow: approve or fix flagged expenses from the prompt
- [ ] `crore` amounts
- [ ] Export the ledger to CSV
