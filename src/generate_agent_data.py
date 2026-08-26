
import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen3:14b"

DEFAULT_CAL_JSON   = BASE_DIR / "data/raw/formatted_data_v2.json"
DEFAULT_STYLE_JSON = BASE_DIR / "data/raw/calendar_context.json"
DEFAULT_OUT_DAYS      = BASE_DIR / "data/processed/calendar_agent_days.csv"
DEFAULT_OUT_REMINDERS = BASE_DIR / "data/processed/calendar_agent_reminders.csv"

# ============================================================
# SCHEMAS & PROMPTS
# ============================================================
GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "style_notes": {"type": "string"},
        "query":       {"type": "string"},
    },
    "required": ["style_notes", "query"],
}

# ------ DAY pipeline modes ----------------------------------
DAY_MODES = [
    (
        "schedule_overview",
        "Ask about the whole day in one go - e.g. 'how's my day looking', "
        "'what do I have on today', 'walk me through today's schedule'. "
        "Do NOT list every event by name - a real person asks generally "
        "and lets the assistant look it up.",
    ),
    (
        "feasibility_check",
        "Ask whether there is room for something NEW given what's already "
        "on the day - e.g. 'do I have time to grab lunch around noon', "
        "'can I fit a call in this afternoon', 'is there a gap before my "
        "evening thing'. Reference the day vaguely, not event-by-event.",
    ),
    (
        "move_event",
        "Ask to move/reschedule exactly ONE specific event from the list "
        "to a different time. Name that event plainly and confidently - "
        "e.g. 'push my 3pm sync to 5', 'move the gym to tomorrow morning'. "
        "No hedging - the user knows which event they mean.",
    ),
    (
        "delete_event",
        "Ask to cancel/remove exactly ONE specific event from the list. "
        "Name it plainly and confidently - e.g. 'cancel my 2pm', "
        "'drop the meditation block from today'. No hedging.",
    ),
    (
        "multi_intent",
        "Bundle two calendar asks into one message: move OR delete one "
        "real event from the list AND ask a feasibility question about "
        "adding something new - e.g. 'move my 10am class to 11 and do I "
        "have time for lunch after'. Both asks should be actionable.",
    ),
]

DAY_SYSTEM_PROMPT = """\
You are building a training dataset for a personal AI assistant's routing
classifier.  You will be given:

1. A REAL DAY from the user's Google Calendar with every event listed
   (time + event name).
2. Several REAL EXAMPLES of how this user actually writes - tone, typos,
   filler words, punctuation habits.  These are your primary style reference.
3. A MESSAGE MODE: the structural type of message to produce.
4. BANNED OPENERS you must not reuse.

Your job in two steps:

STEP 1 – style_notes: Name 2-3 CONCRETE stylistic traits visible in the
examples - actual observable patterns (all-lowercase, trailing '...', uses
'lol', run-on sentences with commas, self-corrections mid-sentence, typical
length, specific filler words).  Be specific; cite the pattern, not a vague
label like 'casual'.

STEP 2 – query: Write ONE message this user might send their AI assistant
about that day, following the MESSAGE MODE exactly, and ACTIVELY APPLYING
every trait you named in style_notes.

RULES:
- move_event / delete_event: reference exactly one event VERBATIM from the
  day list.  Never invent an event that isn't listed.
- schedule_overview / feasibility_check: do NOT name every event - a real
  person asks broadly and lets the assistant check.
- NEVER write machine timestamps (T, Z, 8-digit dates) - use the plain
  time already given ("2:00 PM") or natural phrases ("this afternoon").
- Do not repeat the same word or event name more than once.
- move_event and delete_event should be confident, not hedging.
- Match the LENGTH register of the real examples (short if they're short,
  rambling if they ramble).
- Keep it a single realistic message, not a list.
- No explanation outside the two JSON fields.

Return ONLY JSON: {"style_notes": "<traits>", "query": "<message>"}
"""

# ------ REMINDER pipeline modes -----------------------------
REMINDER_MODES = [
    (
        "set_reminder",
        "A direct ask to SET a reminder for this birthday/movie - e.g. "
        "'set a reminder for Zara's birthday next week', 'remind me when "
        "the Dune sequel releases'.  State it confidently - no hedging.",
    ),
    (
        "check_reminder",
        "A follow-up to CHECK whether a reminder was already set - e.g. "
        "'did I set a reminder for that movie release', 'have I got a "
        "reminder for her birthday already'.  Can be slightly uncertain.",
    ),
    (
        "nudge_reminder",
        "The birthday/movie is only a SMALL ASIDE in a longer message "
        "about something else entirely - remind me about that "
        "Spider-Man release too. The reminder ask should feel like an "
        "afterthought, not the main point.",
    ),
]

REMINDER_SYSTEM_PROMPT = """\
You are building a training dataset for a personal AI assistant's routing
classifier.  You will be given:

1. A BIRTHDAY or MOVIE RELEASE from the user's Google Calendar.
2. Several REAL EXAMPLES of how this user actually writes.
3. A MESSAGE MODE for the reminder ask.
4. BANNED OPENERS you must not reuse.

STEP 1 – style_notes: Name 2-3 CONCRETE stylistic traits from the examples.

STEP 2 – query: Write ONE message this user might send to set/check/mention
a reminder for this event, following the MESSAGE MODE, in their real voice.

RULES:
- Do not mention machine-format dates - use natural phrasing
  ('next month', 'when it drops', 'her birthday') or humanised dates
  ('April 5th', 'sometime in November').
- Keep it a single realistic message, not a list.
- No explanation outside the two JSON fields.

Return ONLY JSON: {"style_notes": "<traits>", "query": "<message>"}
"""

# ============================================================
# STYLE POOL
# ============================================================
CODE_HEAVY = re.compile(
    r"(def |import |Traceback|<html|SELECT \*|function\(|=>|\{\s*$)",
    re.MULTILINE,
)


def load_style_pool(json_path: Path, min_len: int = 20, max_len: int = 800) -> list[str]:
    """Loads style examples from calendar_context.json.
    Each item is expected to have a 'prompt' field.
    Skips code-heavy or too-short/long entries.
    """
    with open(json_path, encoding="utf-8") as f:
        items = json.load(f)
    pool = []
    for item in items:
        p = str(item.get("prompt") or "").strip()
        if not p:
            continue
        if not (min_len <= len(p) <= max_len):
            continue
        if CODE_HEAVY.search(p):
            continue
        pool.append(p)
    return pool


class StyleSampler:
    """Exhausting-shuffle sampler: cycles through all pool rows before
    repeating any, thread-safe for concurrent workers."""

    def __init__(self, pool: list[str]):
        self.pool = pool
        self._avail = list(range(len(pool)))
        random.shuffle(self._avail)
        self._used: list[int] = []
        self._lock = threading.Lock()

    def sample(self, n: int) -> list[str]:
        n = min(n, len(self.pool))
        with self._lock:
            out = []
            while len(out) < n:
                if not self._avail:
                    self._avail, self._used = self._used, []
                    random.shuffle(self._avail)
                idx = self._avail.pop()
                self._used.append(idx)   # store the index, not the string
                out.append(self.pool[idx])
            return out


# ============================================================
# LLM CALL
# ============================================================
def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _opener(text: str, n: int = 2) -> str:
    return " ".join(text.strip().split()[:n]).lower().strip(",.!?")


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _too_similar(a: str, b: str, thresh: float = 0.55) -> bool:
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= thresh


def _call_llm(system: str, user: str, retries: int = 2) -> str | None:
    payload = {
        "model":  MODEL,
        "system": system,
        "prompt": user,
        "stream": False,
        "format": GEN_SCHEMA,
        "think":  False,
        "options": {"temperature": 1.0, "top_p": 0.95, "num_predict": 300},
    }
    for _ in range(retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            raw = _strip_think(r.json().get("response", ""))
            if not raw:
                continue
            data = json.loads(raw)
            q = str(data.get("query", "")).strip()
            if q:
                return q
        except Exception:
            time.sleep(1)
    return None


def _generate_queries(
    anchor_text: str,
    modes: list[tuple[str, str]],
    style_sampler: StyleSampler,
    n: int,
    style_n: int,
    system_prompt: str,
) -> list[tuple[str, str]]:
    """Generate n queries for one anchor (day or reminder item).
    Returns list of (mode_name, query) tuples.
    """
    generated: list[str] = []
    banned: list[str]    = []
    results: list[tuple[str, str]] = []

    for i in range(n):
        mode_name, mode_desc = modes[i % len(modes)]
        style_examples = style_sampler.sample(style_n)
        style_block = "\n".join(f"{j+1}. {ex}" for j, ex in enumerate(style_examples))
        already_block = (
            "\nAlready generated for this anchor (write something structurally "
            "DIFFERENT, not a reskin):\n"
            + "\n".join(f"- {q}" for q in generated)
        ) if generated else ""
        banned_block = (
            "\nDo NOT start with any of: "
            + ", ".join(f"'{b}'" for b in banned)
        ) if banned else ""

        user_prompt = (
            f"CALENDAR DATA:\n{anchor_text}\n\n"
            f"REAL STYLE EXAMPLES:\n{style_block}\n"
            f"\nMESSAGE MODE ({mode_name}): {mode_desc}"
            f"{already_block}{banned_block}\n\n"
            "Write one message in this user's real voice following the mode above."
        )

        sep = "─" * 72
        tqdm.write(f"\n{sep}")
        tqdm.write(f"📤 PROMPT  [query {i+1}/{n}]  mode={mode_name}")
        tqdm.write(sep)
        tqdm.write(f"[REDACTED ANCHOR TEXT - Length: {len(anchor_text)}]")
        tqdm.write("\n--- STYLE EXAMPLES ---")
        tqdm.write(f"[REDACTED STYLE EXAMPLES - Length: {len(style_block)}]")
        tqdm.write(sep)

        query = None
        for _ in range(3):   # up to 3 dedup retries
            candidate = _call_llm(system_prompt, user_prompt)
            if candidate:
                tqdm.write(f"✅ RESPONSE: [Length: {len(candidate)}]")
            if candidate and not any(_too_similar(candidate, g) for g in generated):
                query = candidate
                break

        if query:
            generated.append(query)
            banned.append(_opener(query))
            results.append((mode_name, query))

    return results


# ============================================================
# DAY PIPELINE
# ============================================================
def _day_anchor_text(date_info: dict) -> str:
    pretty = date_info["pretty_date"]
    lines  = [f"- {ev['time']}: {ev['summary']}" for ev in date_info["events"]]
    return f"Date: {pretty}\nEvents that day:\n" + "\n".join(lines)


def generate_for_day(
    date_key: str,
    date_info: dict,
    style_sampler: StyleSampler,
    n: int,
    style_n: int,
) -> list[dict]:
    anchor_text = _day_anchor_text(date_info)
    results = _generate_queries(
        anchor_text, DAY_MODES, style_sampler, n, style_n, DAY_SYSTEM_PROMPT
    )
    rows = []
    for mode_name, query in results:
        rows.append({
            "prompt":          query,
            "category":        "AGENT",
            "subcategory":     "CALENDAR_AGENT_DAY",
            "action_type":     mode_name,
            "message_mode":    mode_name,
            "source_uid":      date_key,
            "source_summary":  date_info["pretty_date"],
        })
    return rows


# ============================================================
# REMINDER PIPELINE
# ============================================================
def _reminder_anchor_text(item: dict, kind: str) -> str:
    if kind == "birthday":
        text = f"Birthday: {item['summary']}\nAnnual date: {item['date']}"
    else:
        # movie sub-kinds: release_date / watch_schedule / news
        movie_kind = item.get("kind", "release_date")
        if movie_kind == "release_date":
            text = f"Movie release: {item['summary']}\nRelease date: {item.get('release_date', '')}"
        elif movie_kind == "watch_schedule":
            text = f"Movie watch block: {item['summary']}\nScheduled watch date: {item.get('watch_date', '')}"
        else:  # news
            text = f"MCU/movie news event: {item['summary']}\nDate noted: {item.get('event_date', '')}"
    if item.get("description"):
        text += f"\nDetails: {item['description'][:200]}"
    return text


def generate_for_reminder(
    item: dict,
    kind: str,
    style_sampler: StyleSampler,
    n: int,
    style_n: int,
) -> list[dict]:
    anchor_text = _reminder_anchor_text(item, kind)
    results = _generate_queries(
        anchor_text, REMINDER_MODES, style_sampler, n, style_n,
        REMINDER_SYSTEM_PROMPT
    )
    rows = []
    for mode_name, query in results:
        rows.append({
            "prompt":         query,
            "category":       "AGENT",
            "subcategory":    "CALENDAR_AGENT_REMINDER",
            "action_type":    mode_name,
            "message_mode":   mode_name,
            "source_uid":     item.get("summary", ""),
            "source_summary": item.get("summary", ""),
        })
    return rows


# ============================================================
# PIPELINE RUNNER
# ============================================================
def _write_markdown(rows: list[dict], path: Path, title: str):
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source_summary"], []).append(r)
    lines = [f"# {title} ({len(rows)} total)\n"]
    for src, src_rows in by_src.items():
        lines.append(f"## {src}\n")
        for r in src_rows:
            lines.append(f"- **{r['action_type']}**: {r['prompt']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_existing_rows(out_path: Path) -> tuple[list[dict], set[str]]:
    """Load already-saved rows and return (rows, set_of_source_uids) for
    resume logic.  Returns empty structures if the file does not exist."""
    if not out_path.exists():
        return [], set()
    try:
        df = pd.read_csv(out_path)
        rows = df.to_dict("records")
        done = set(str(r.get("source_uid", "")) for r in rows)
        print(f"   ♻️  Resuming: {len(rows)} rows already saved, "
              f"{len(done)} unique source_uids done.")
        return rows, done
    except Exception as e:
        print(f"   ⚠️  Could not load existing output ({e}); starting fresh.")
        return [], set()


def run_pipeline(
    items: list,
    generate_fn,
    style_pool: list[str],
    subcategory: str,
    style_n: int,
    out_path: Path,
    workers: int = 1,
    save_every: int = 10,
    item_uid_fn=None,
    resume: bool = True,
):
    """Run one generation pipeline.

    item_uid_fn: optional callable(item) -> str that returns the source_uid
    for a given item, used to skip already-processed items on resume.
    If None, no resume skipping is performed.
    resume: if False, ignore any existing output file and start from scratch.
    """
    if len(style_pool) < style_n:
        raise ValueError(f"Style pool ({len(style_pool)}) < style_sample_size ({style_n})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = out_path.with_suffix(".md")

    # ---- resume: skip items already in the output file ----
    rows, done_uids = _load_existing_rows(out_path) if resume else ([], set())
    if resume and item_uid_fn and done_uids:
        items_before = len(items)
        items = [it for it in items if str(item_uid_fn(it)) not in done_uids]
        print(f"   ⏭️  Skipping {items_before - len(items)} already-done items; "
              f"{len(items)} remaining.")

    if not items:
        print(f"   ✅ {subcategory}: nothing new to process.")
        return

    sampler = StyleSampler(style_pool)
    lock = threading.Lock()
    last_save = len(rows)   # track absolute count, not delta

    pbar = tqdm(total=len(items), desc=f"Generating {subcategory}", unit="item")
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(generate_fn, item, sampler): item
                for item in items
            }
            for future in as_completed(futures):
                # ── per-future exception guard: one bad LLM call must NOT
                # kill the whole pipeline — just log and continue.
                try:
                    item_rows = future.result()
                except Exception as exc:
                    item = futures[future]
                    tqdm.write(f"  ⚠️  Worker error for {item!r}: {exc}")
                    pbar.update(1)
                    continue

                with lock:
                    rows.extend(item_rows)
                    for r in item_rows:
                        tqdm.write(
                            f"  💬 [{r['action_type']}] "
                            f"[Length: {len(r['prompt'])}]"
                        )
                    if len(rows) - last_save >= save_every:
                        pd.DataFrame(rows).to_csv(out_path, index=False)
                        _write_markdown(rows, md_path, subcategory)
                        last_save = len(rows)
                        tqdm.write(f"  💾 Auto-saved {len(rows)} rows → {out_path.name}")
                pbar.update(1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted - saving progress...")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        _write_markdown(rows, md_path, subcategory)
        print(f"💾 Saved {len(rows)} rows → {out_path}")
        sys.exit(1)
    finally:
        pbar.close()

    pd.DataFrame(rows).to_csv(out_path, index=False)
    _write_markdown(rows, md_path, subcategory)
    print(
        f"✅ {subcategory}: {len(rows)} rows from {len(items)} items"
        f" → {out_path}"
    )


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic calendar-agent training data"
    )
    ap.add_argument("--calendar-json",    default=str(DEFAULT_CAL_JSON))
    ap.add_argument("--style-json",        default=str(DEFAULT_STYLE_JSON),
                    help="Path to calendar_context.json style examples")
    ap.add_argument("--style-sample-size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=10,
                    help="Auto-save every N generated rows (default 10)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing output and start from scratch")

    # Day pipeline
    ap.add_argument("--out-days",          default=str(DEFAULT_OUT_DAYS))
    ap.add_argument("--examples-per-day",  type=int, default=5,
                    help="Queries per calendar day (default 5, max meaningful = len(DAY_MODES)=5)")
    ap.add_argument("--max-days",          type=int, default=None)
    ap.add_argument("--min-events-per-day",type=int, default=2)
    ap.add_argument("--skip-days",         action="store_true")

    # Reminder pipeline
    ap.add_argument("--out-reminders",         default=str(DEFAULT_OUT_REMINDERS))
    ap.add_argument("--examples-per-reminder", type=int, default=3,
                    help="Queries per birthday/movie (default 3, max meaningful = 3)")
    ap.add_argument("--max-reminders",         type=int, default=None)
    ap.add_argument("--skip-reminders",        action="store_true")

    args = ap.parse_args()

    # ---- load shared data ----
    print(f"⏳ Loading style pool from {args.style_json}...")
    style_pool = load_style_pool(Path(args.style_json))
    print(f"   {len(style_pool)} usable style examples from calendar_context.json")

    print(f"⏳ Loading calendar data from {args.calendar_json}...")
    with open(args.calendar_json, encoding="utf-8") as f:
        cal = json.load(f)

    # ---- DAY PIPELINE ----
    if not args.skip_days:
        days = cal.get("days", {})
        day_items = [
            (date_key, info)
            for date_key, info in days.items()
            if len(info["events"]) >= args.min_events_per_day
        ]
        print(f"\n📅 Day pipeline: {len(day_items)} days with ≥ {args.min_events_per_day} events")

        if args.max_days:
            day_items = random.sample(day_items, min(args.max_days, len(day_items)))
            print(f"   Capped to {len(day_items)} days")

        n_per_day = args.examples_per_day
        style_n   = args.style_sample_size

        # Use default-arg binding to freeze n_per_day / style_n at definition
        # time, preventing the late-binding closure bug if the reminder block
        # re-assigns style_n before all day-pipeline workers finish.
        def _day_fn(item, sampler, _n=n_per_day, _s=style_n):
            date_key, date_info = item
            return generate_for_day(date_key, date_info, sampler, _n, _s)

        run_pipeline(
            items       = day_items,
            generate_fn = _day_fn,
            style_pool  = style_pool,
            subcategory = "CALENDAR_AGENT_DAY",
            style_n     = style_n,
            out_path    = Path(args.out_days),
            workers     = args.workers,
            save_every  = args.save_every,
            # Resume key: the date string (first element of the tuple)
            item_uid_fn = lambda item: item[0],
            resume      = not args.no_resume,
        )

    # ---- REMINDER PIPELINE ----
    if not args.skip_reminders:
        birthdays = cal.get("birthdays", [])
        movies    = cal.get("movies",    [])
        reminder_items = (
            [(b, "birthday") for b in birthdays]
            + [(m, "movie")   for m in movies]
        )
        print(f"\n🎂 Reminder pipeline: {len(birthdays)} birthdays + {len(movies)} movies")

        if args.max_reminders:
            reminder_items = random.sample(
                reminder_items, min(args.max_reminders, len(reminder_items))
            )
            print(f"   Capped to {len(reminder_items)} reminders")

        n_per_rem = args.examples_per_reminder
        style_n   = args.style_sample_size

        # Same late-binding fix: freeze n_per_rem / style_n as defaults.
        def _rem_fn(item, sampler, _n=n_per_rem, _s=style_n):
            reminder_item, kind = item
            return generate_for_reminder(reminder_item, kind, sampler, _n, _s)

        run_pipeline(
            items       = reminder_items,
            generate_fn = _rem_fn,
            style_pool  = style_pool,
            subcategory = "CALENDAR_AGENT_REMINDER",
            style_n     = style_n,
            out_path    = Path(args.out_reminders),
            workers     = args.workers,
            save_every  = args.save_every,
            # Resume key: the summary field of the reminder item
            item_uid_fn = lambda item: item[0].get("summary", ""),
            resume      = not args.no_resume,
        )


if __name__ == "__main__":
    main()