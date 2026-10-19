# Group Chat Ledger

**Ask questions about a WhatsApp chat, and get balances you can trust.**

Group Chat Ledger reads a WhatsApp export and turns it into a ledger of **expenses, settlements, commitments and events**. You can then ask questions in plain English or Hinglish, and every answer cites the exact messages it came from.

```text
Ask> Goa kab ja rahe hain?
The Goa trip is planned for 20–23 December.
  - 14 Oct 21:02 Aman Verma: Guys Goa plan final? 20-23 Dec

Ask> balances
Who pays whom:
  Aman Verma → Karan: ₹4,000.00
  Riya → Karan: ₹4,000.00
```

An LLM (Google Gemini, or a local model through Ollama) reads the chat. **Deterministic Python code then checks every record before it's accepted**, and **does all the money maths**. The LLM never adds, splits or settles an amount.

---

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [How to use](#how-to-use) — export, run, ask questions, balances, updating and deleting chats
- [Choosing a model: Gemini or Ollama](#choosing-a-model-gemini-or-ollama)
- [How it works](#how-it-works)
- [Why you can trust the numbers](#why-you-can-trust-the-numbers)
- [Privacy and security](#privacy-and-security)
- [Configuration](#configuration)
- [Development](#development)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
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

You need **Python 3.11+** (`python3 --version` to check). Nothing else is needed for Gemini; for a fully local setup, see [Choosing a model](#choosing-a-model-gemini-or-ollama).

```bash
git clone https://github.com/rohan-bhxtia/GC-Ledger.git
cd GC-Ledger
mkdir -p chats               # then copy your WhatsApp export into chats/
python3 run.py
```

That's it. The first run sets everything up and asks for anything it needs. The full step-by-step guide is below.

---

## How to use

### Step 1: Export the chat from WhatsApp

| Phone | Steps |
|---|---|
| **Android** | Open the chat → **⋮** (top right) → **More** → **Export chat** → **Without media** |
| **iPhone** | Open the chat → tap the **chat name** at the top → scroll down → **Export Chat** → **Without Media** |

Send the file to your computer: AirDrop, email it to yourself, or save it to Google Drive. You get a `.txt` file (Android) or a `.zip` (iPhone). Both work, and you don't need to unzip anything.

> **Without media** keeps the file small. Photos and voice notes aren't read anyway; only their captions are.

### Step 2: Put the file in `chats/`

Drop it straight into the `chats/` folder in the project. If the folder isn't there yet, create it (`mkdir chats`), or run `python3 run.py` once and it's created for you.

```text
GC-Ledger/
├── chats/
│   └── WhatsApp Chat with Goa Trip.txt   ← here
├── run.py
└── ...
```

- **A `.txt` or `.zip` works**, and so does the folder macOS creates when it unzips the file for you (`chats/WhatsApp Chat with Goa Trip/…`). Folders nested deeper than that aren't found.
- **The file name sets the chat name.** `WhatsApp Chat with Goa Trip.txt` becomes **Goa Trip**.
- **Your chat is never committed.** `chats/` is git-ignored.

### Step 3: Choose a model

Copy the example settings file once:

```bash
cp .env.example .env
```

Then pick **one** of the two:

**Option A: Google Gemini.** Best answers, needs an internet connection and an API key.

```dotenv
GCL_LLM_PROVIDER=gemini
```

Get a free key at <https://aistudio.google.com/apikey>. You don't need to paste it into `.env` yourself: `run.py` asks for it on the first run, hides it as you type, and saves it to `.env` for you.

**Option B: Ollama.** Runs completely on your computer, free, no limits, but slower.

```bash
ollama pull qwen3:8b          # once, about 5 GB
```

```dotenv
GCL_LLM_PROVIDER=ollama
GCL_OLLAMA_MODEL=qwen3:8b
GCL_OLLAMA_NUM_CTX=16384
```

You need [Ollama](https://ollama.com) installed and running (the Ollama app in your menu bar, or `ollama serve`). See [Choosing a model](#choosing-a-model-gemini-or-ollama) for the trade-offs.

### Step 4: Run it

```bash
python3 run.py
```

You don't need to activate a virtualenv. On the first run, `run.py` creates `.venv` and installs the dependencies (once, about a minute). Then you'll see something like:

```text
Model: Google Gemini (gemini-3.8-flash)
Chat: WhatsApp Chat with Goa Trip.txt  ->  saved as 'Goa Trip' (11 messages)
Message text is sent to Google Gemini (gemini-3.8-flash) to answer questions and extract expenses and plans.
Continue? [y/N] y
11 messages stored (11 new).

Expenses and plans not extracted yet for part of this chat: 3 model call(s) needed.
You can skip this and ask questions right away (1 call per question).
Extract now? [y/N] y
Extracting. Progress is saved every 10 windows, so Ctrl+C is safe; run again later to continue.
  3/3 windows done, about 0 min left
4 new records, 0 rejected.

=== Goa Trip ===
Who pays whom:
  Aman Verma → Karan: ₹4,000.00
  Riya → Karan: ₹4,000.00

See the calculation: gcl balances --chat NAME --explain PERSON

Commitments:
  [promised] Riya: bring the speaker
  [promised] Karan: book cab (due 2026-10-16)
Events:
  [confirmed] Goa plan: 2026-12-20 to 2026-12-23

Type a question, or one of:
  balances          who pays whom
  explain NAME      how NAME's balance was calculated
  plans             commitments and events
  help              show this again
  quit              exit

Ask>
```

The `Continue?` question only appears with Gemini, the first time a chat is sent to Google. With Ollama nothing leaves your computer, so it doesn't ask.

### Step 5: Extract expenses and plans, or skip it

**"Extract now?"** reads the whole chat and finds the expenses, payments, promises and plans behind `balances` and `plans`.

- **Small group chat:** say `y`. It takes seconds with Gemini, or a few minutes with Ollama.
- **Big chat** (years of messages, hundreds of "model calls"): press **Enter** to skip for now. **Questions work without extraction.** Extract later, for example overnight with Ollama.

While it runs, you'll see a live counter:

```text
  120/607 windows done, about 95 min left
```

**Ctrl+C is always safe.** Work is saved every 10 windows, and the next `python3 run.py` carries on where it stopped. With Gemini's free tier it stops by itself when the daily quota runs out, and continues on a later run.

### Step 6: Ask questions

Type a question at `Ask>` and press Enter. Every answer lists the messages it came from:

```text
Ask> when is the goa trip?
The Goa trip is planned for 20–23 December.
  - 14 Oct 21:02 Aman Verma: Guys Goa plan final? 20-23 Dec
```

English, Hindi and Hinglish all work. Some questions that work well:

| Kind | Examples |
|---|---|
| Money | *who paid for the villa?* · *how much did Karan spend?* · *kitne paise bache hain?* |
| Plans | *when are we leaving?* · *what time is dinner on Saturday?* · *Goa kab ja rahe hain?* |
| Promises | *who said they'd book the cab?* · *did anyone bring the speaker?* |
| Finding things | *what was the hotel's name?* · *did Riya send the address?* · *what did we decide about the budget?* |

**Tips for better answers:**
- **Be specific.** "When is the Goa trip?" works better than "what's happening?".
- **Use the words people used in the chat.** With Ollama, the older messages it sends are the ones that share words with your question.
- **"(no supporting message found)"** after an answer means the model gave an answer but couldn't point to a real message, so don't trust it.
- **For exact amounts, use `balances`.** It's calculated by code, not by the model, so it's always exact.

### Step 7: Check who owes whom

| Command | What you get |
|---|---|
| `balances` | The fewest payments that settle everyone up, plus anything that **needs review** |
| `explain NAME` | Every line behind one person's balance, with the messages it came from |
| `plans` | Promises (who will do what, by when) and events (what's happening, when) |

```text
Ask> explain Riya
Riya owes ₹4,000.00
  −      ₹4,000.00  share of "Villa booking" (₹12,000.00 ÷ 3)
      ↳ 14 Oct 21:07 Karan: "Booked the villa, paid ₹12,000 split 3 ways pls" [88268ef8]
  =      -₹4,000.00
```

**"Needs review (not counted)"** lists expenses the code wasn't sure about, for example when the chat doesn't say who shares the cost. They're shown, but never silently added to anyone's balance.

Type `quit` (or press Ctrl+D) to exit.

### Updating a chat later

Export the chat again, **with the same file name**, and replace the old file in `chats/`. Then run `python3 run.py`.

Only the new messages are read; nothing is counted twice. You can also delete the export from `chats/` after importing: everything needed is stored in `data/gcl.db`.

### Several chats

Put several exports in `chats/`. `run.py` lists them and asks which one to open:

```text
Chats found:
  1. WhatsApp Chat with Flatmates.txt
  2. WhatsApp Chat with Goa Trip.txt
Which one? [1-2]
```

Each chat is stored separately, under its own name.

### Deleting a chat

```bash
.venv/bin/gcl delete --chat "Goa Trip"
```

This permanently removes the chat, its messages and everything extracted from it. The text is wiped from the database file, not just hidden. To wipe everything, delete the `data/` folder.

### Switching models

Change `GCL_LLM_PROVIDER` in `.env` (and the model name if you like), then run `python3 run.py` again. Your stored chats stay as they are. Records already extracted are kept; only new work uses the new model.

### The `gcl` command

Everything `run.py` does is also available as separate commands, for scripting or quick checks:

```bash
source .venv/bin/activate

gcl import "chats/WhatsApp Chat with Goa Trip.txt" --chat "Goa Trip"
gcl chats                                       # list imported chats
gcl balances --chat "Goa Trip"                  # who pays whom
gcl balances --chat "Goa Trip" --explain Riya   # how Riya's balance was worked out
gcl plans --chat "Goa Trip"                     # commitments and events
gcl ask --chat "Goa Trip" "when are we leaving?"
gcl delete --chat "Goa Trip"                    # permanently remove a chat
```

`import` (with Gemini) and `delete` ask for confirmation. Pass `--yes` to skip it.

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

## Choosing a model: Gemini or Ollama

|  | **Google Gemini** (default) | **Ollama** (local) |
|---|---|---|
| Where the chat goes | Google's API | Nowhere: runs on your machine |
| Answer quality | Best | Good on direct questions, weaker on vague ones |
| Speed per question | A few seconds | About 1–2 minutes (8B model, Apple M4) |
| Extracting a large chat | Fast, but limited by your quota | No limits, but slow (about 15 s per window) |
| Cost | Free tier is about 20 requests per model per day | Free |
| Needs | An API key and internet | [Ollama](https://ollama.com), ~6 GB disk, 16 GB RAM recommended |

**My recommendation:** start with **Gemini** to try it, since it's fast and gives the best answers. Use **Ollama** for private chats you'd rather not send anywhere, or for extracting a very large chat without hitting a quota.

**Gemini's free tier** limits requests per minute and per day. The app waits as long as Google asks after a per-minute limit. When the daily quota runs out, it stops straight away, says when it resets, and picks up the unfinished work on a later run. Each Gemini model has its own daily quota, so switching `GCL_GEMINI_MODEL` (for example to `gemini-3.6-flash`) is a quick workaround.

**Ollama's recommended model** is `qwen3:8b`: good at Hinglish and strict JSON, and it fits in 16 GB of RAM. Older models such as Llama 3 8B tend to miss expenses and drift off-topic.

**Ollama: speed vs coverage.** `GCL_OLLAMA_NUM_CTX` sets how much of the chat each question can see. `8192` is roughly twice as fast; `16384` covers about twice as many messages. With a small context, Q&A sends the most recent messages, plus older messages that share words with your question (with a little context around each). Very long pasted messages are cut short, so one paste can't crowd out the rest.

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
| A window failed (timeout, quota) or the run was interrupted | Its messages stay pending and are retried on the next run |

---

## Privacy and security

- **Fully local option.** With `GCL_LLM_PROVIDER=ollama`, no chat content leaves your computer.
- **What Gemini receives:** only text messages and media captions, with sender names and timestamps. System messages and deleted messages are never sent. `run.py` and `gcl import` ask before the first send.
- **Prompt-injection resistant.** Messages are sent as JSON data marked untrusted. The model is told never to follow instructions in them, and instruction-like messages are flagged. Long pasted texts are cut short in Q&A prompts.
- **Your data stays in one file.** `data/gcl.db` is created readable only by you (`0600`).
- **Deletion is real.** `gcl delete` removes a chat, its messages, records and sources. SQLite `secure_delete` plus `VACUUM` means the text isn't left behind in the file, and a test checks this.
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

**179 tests**, all offline. They use fake LLM clients and fictional sample chats in `tests/fixtures/`, so they need no API key or Ollama. They cover parsing, windowing, validation rules, deduplication, secure delete, SQL injection, the ledger's zero-sum invariant, retries and quota handling, batched and resumable imports, Q&A citation checks and the CLI end to end.

Tests that call the real Gemini API are marked `live` and skipped by default:

```bash
python -m pytest -m live          # needs GCL_GEMINI_API_KEY
```

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
├── run.py                  # start here: one command does everything
├── pyproject.toml          # dependencies and the `gcl` command
├── .env.example            # copy to .env; every setting, with comments
├── chats/                  # put WhatsApp exports here (git-ignored)
├── data/                   # the SQLite database, gcl.db (git-ignored)
├── src/gcl/
│   ├── config.py           # settings from .env
│   ├── models.py           # the data shapes: messages, raw and checked records
│   ├── parser.py           # export file → messages
│   ├── extractor.py        # messages → windows → prompts → LLM → checked records
│   ├── validation.py       # the rules every LLM answer must pass
│   ├── llm.py              # picks Gemini or Ollama
│   ├── gemini.py           # Google Gemini client (retries, quota handling)
│   ├── ollama.py           # local Ollama client
│   ├── db.py               # SQLite storage and duplicate merging
│   ├── pipeline.py         # import_chat(): the whole import, in saved batches
│   ├── ledger.py           # who owes whom, settle-up, explanations
│   ├── qa.py               # questions and verified citations
│   └── cli.py              # the `gcl` command
└── tests/
    ├── fixtures/           # fictional Android, iOS and malformed chats
    └── test_*.py           # one test file per module
```

### Reading the code

Every file starts with a plain-English summary of what it does and the steps it follows. A good order to read them in, following one chat through the system:

1. [`models.py`](src/gcl/models.py): the shapes of the data everything else passes around.
2. [`parser.py`](src/gcl/parser.py): how an export file becomes a list of messages.
3. [`extractor.py`](src/gcl/extractor.py): how messages are grouped into windows and sent to the LLM.
4. [`validation.py`](src/gcl/validation.py): how every LLM answer is checked. This is the heart of the project.
5. [`pipeline.py`](src/gcl/pipeline.py) and [`db.py`](src/gcl/db.py): how it's all stored, and how re-imports avoid double-counting.
6. [`ledger.py`](src/gcl/ledger.py): how balances are worked out.
7. [`qa.py`](src/gcl/qa.py): how questions are answered.
8. [`run.py`](run.py): how it's all tied together for the user.

The matching file in `tests/` shows each module being used, with small, readable examples.

---

## Troubleshooting

| You see | What it means | What to do |
|---|---|---|
| `No chat found. Put your WhatsApp export (.txt or .zip) in: …/chats` | Nothing is in `chats/`, or the file is nested more than one folder deep | Move the `.txt` or `.zip` directly into `chats/` (or one folder below it) |
| `Gemini daily quota used up (429 RESOURCE_EXHAUSTED); resets in about 9.3 h` | The free tier's daily limit for this model is used up | Wait until it resets, switch `GCL_GEMINI_MODEL` to another model (each has its own quota), or use Ollama |
| `Gemini rate limit; waiting 31 s before retrying` | The per-minute limit was hit | Nothing: it waits and carries on. Don't press Ctrl+C |
| `Gemini rejected the request (400 INVALID_ARGUMENT)` | The model doesn't accept the thinking setting | Set `GCL_GEMINI_THINKING_LEVEL=` (empty) in `.env` |
| `Gemini rejected the request (403 …)` or `(401 …)` | The API key is wrong or disabled | Fix `GCL_GEMINI_API_KEY` in `.env`, or delete that line and run again to be asked for it |
| `Gemini rejected the request (404 NOT_FOUND)` | The model name doesn't exist | Check `GCL_GEMINI_MODEL` against the models in [Google AI Studio](https://aistudio.google.com) |
| `Ollama is not running at http://localhost:11434` | The Ollama app isn't open | Open Ollama (or run `ollama serve`) and try again |
| `Ollama model 'qwen3:8b' is not installed. Run: ollama pull qwen3:8b` | The model isn't downloaded | Run the command it shows |
| Extraction looks stuck | The counter updates after every 10 windows. With Ollama, that can take a couple of minutes | Wait for the first update. Ctrl+C is safe at any point |
| `Everyone is settled.` but people do owe money | Extraction hasn't finished, or the expenses are under "Needs review" | Run extraction (`y` at "Extract now?"), then check `balances` for "Needs review" |
| `(no supporting message found)` after an answer | The model answered, but its cited messages don't exist | Don't rely on that answer. Rephrase the question, or use Gemini |
| Answers take minutes (Ollama) | The local model is reading about 15k tokens per question | Set `GCL_OLLAMA_NUM_CTX=8192` for roughly twice the speed |
| `File is 12.3 MB; limit is 10 MB.` | The export is bigger than the safety limit | Raise `GCL_MAX_IMPORT_MB` in `.env` (up to 100) |
| `Inconsistent date format: both DD/MM and MM/DD dates found.` | The file mixes two date formats, e.g. two exports pasted together | Export the chat again as a single file |
| `Zip contains several chat files; export one chat at a time.` | The zip holds more than one chat | Export each chat on its own |
| `No WhatsApp messages found. Is this an exported chat?` | The file isn't a WhatsApp export | Export it again from WhatsApp (Step 1) |

To see more detail about what's happening, set `GCL_LOG_LEVEL=DEBUG` in `.env`. Logs never contain message text.

---

## FAQ

**Is my chat sent anywhere?**
With Ollama, no: everything stays on your computer. With Gemini, the text of the messages (with names and times) is sent to Google to answer questions and extract records, and the app asks before the first send. The database stays on your computer either way.

**Can it get the money wrong?**
The model only *reads*; Python code does every calculation. An amount is accepted only if it's written in the message the model cites, and anything uncertain is listed under "Needs review" instead of being counted. Mistakes are still possible, for example when a message is ambiguous, so `explain NAME` shows exactly which messages each number came from.

**Does it work for one-to-one chats, not just groups?**
Yes. Balances are most useful for groups, but questions work on any chat.

**Can I delete the export file after importing?**
Yes. Everything needed is stored in `data/gcl.db`. Keep the export only if you want to re-import it later.

**Which languages work?**
English, Hindi written in English letters (Hinglish), and a mix of the two. Amounts must be in rupees.

**Where is my data, and how do I remove it all?**
In one file, `data/gcl.db`, readable only by you. Use `gcl delete --chat NAME` for one chat, or delete the `data/` folder to remove everything.

**Do I need to keep `run.py` running?**
No. Close it any time with `quit` or Ctrl+C. Everything is saved, and the next run continues where you left off.

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
