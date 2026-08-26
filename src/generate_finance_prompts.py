"""
generate_finance_prompts.py
----------------------------
Production-quality Finance Prompt Builder for Elyestra.
Generates 3,000–5,000 synthetic finance prompts in Ezaan's natural writing style.

Datasets used:
  - personal_context_100.csv  → writing style ONLY
  - transactions_user1_decrypted.csv → real transactions
  - my_online_orders.csv → real online shopping
  - upcoming_marvel_dc_movies.csv → upcoming movies (finance angle only)

Output: data/processed/finance_prompts.csv
  Columns: prompt, category, subcategory, action_type, message_mode, source_uid, source_summary
  (matches calendar_agent_days.csv schema)
"""

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

STYLE_CSV        = BASE_DIR / "data/raw/personal_context_100.csv"
TRANSACTIONS_CSV = BASE_DIR / "data/raw/transactions_user1_decrypted.csv"
ORDERS_CSV       = BASE_DIR / "data/raw/my_online_orders.csv"
MOVIES_CSV       = BASE_DIR / "data/raw/upcoming_marvel_dc_movies.csv"
OUT_CSV          = BASE_DIR / "data/processed/finance_prompts.csv"

WORKERS    = 1      # Ollama processes 1 request at a time
SAVE_EVERY = 10     # save every 10 new prompts
TARGET_N   = 4000   # approximate target prompts


# ============================================================
# FINANCE PROMPT MODES
# Each mode = (name, short instruction for the LLM)
# ============================================================
FINANCE_MODES = [
    ("spending_check",
     "Ask how much was spent on a specific category or item from the data. "
     "Keep it short and natural. E.g. 'how much did i spend on food last month'"),

    ("income_credit",
     "Ask about income, bonuses, credits, or reimbursements received. "
     "E.g. 'what bonuses did i get in june', 'did my reimbursement come in'"),

    ("monthly_report",
     "Ask for a full monthly summary or breakdown. "
     "Can be casual like 'how was april for me financially'"),

    ("weekly_report",
     "Ask for a weekly spending summary or quick recap for a specific week."),

    ("yearly_stats",
     "Ask for a yearly overview, annual totals, or year-to-date stats. "
     "E.g. 'how much have i spent so far this year'"),

    ("budget_check",
     "Ask whether they can afford something or if budget allows a purchase. "
     "Ground it in real items/movies from the data."),

    ("trend_analysis",
     "Ask about a spending trend over time. E.g. 'is my food spending going up', "
     "'am i spending more on uber lately'"),

    ("comparison",
     "Compare two months, two categories, or two time periods against each other."),

    ("category_deep_dive",
     "Ask to break down a single category in detail. "
     "E.g. 'show me all my therapy payments', 'list every uber ride'"),

    ("add_transaction",
     "Ask to add / log a new transaction. "
     "E.g. 'add 500 for biryani today', 'log gym payment for july'"),

    ("edit_transaction",
     "Ask to edit or correct an existing transaction. "
     "E.g. 'that food entry on june 17 was actually 2000 not 1700'"),

    ("delete_transaction",
     "Ask to remove or delete a specific transaction. "
     "E.g. 'delete that duplicate uber from march 6'"),

    ("savings_planning",
     "Ask about savings potential or how much can be saved. "
     "E.g. 'how much could i save if i cut uber', 'what if i skip eating out'"),

    ("entertainment_budget",
     "Ask about movie or entertainment spending specifically, or plan for upcoming movies. "
     "Ground in real movies from the dataset."),

    ("subscription_check",
     "Ask about recurring subscriptions like GPT, Audible, Cursor, gym. "
     "E.g. 'when does my next audible payment hit', 'how much do subscriptions cost me monthly'"),

    ("purchase_decision",
     "Ask whether to buy something specific — can reference online orders or movies. "
     "E.g. 'should i get another mechanical keyboard', 'can i afford the batman part 2 in imax'"),

    ("shopping_history",
     "Ask about past Daraz or online purchases. "
     "E.g. 'how much have i spent on daraz', 'did i overspend on electronics'"),

    ("food_expenses",
     "Ask specifically about food and snacks spending patterns. "
     "E.g. 'i feel like i spend too much on food tbh', 'what's my avg food spend per week'"),

    ("uber_transport",
     "Ask about Uber / transport spending. "
     "E.g. 'how much did uber cost me in march', 'is transport killing my budget'"),

    ("research_tools",
     "Ask about research and learning tool expenses: Udemy, GPT, Cursor, research materials. "
     "E.g. 'how much have i spent on courses', 'is my gpt sub worth it'"),

    ("net_flow",
     "Ask about net income vs expenses, overall cash flow, or net balance. "
     "E.g. 'am i in surplus or deficit this month', 'how much did i actually save in june'"),

    ("follow_up",
     "Write a very short follow-up or clarifying question that assumes a previous response. "
     "E.g. 'and what about last month?', 'wait what was the total again'"),

    ("vague_incomplete",
     "Write an intentionally incomplete or vague prompt, like the user started typing "
     "but didn't finish. E.g. 'how much did i', 'what was my spending on'"),

    ("typo_casual",
     "Write a casual prompt with natural typos or abbreviations, very short. "
     "E.g. 'hw much was food in feb', 'did i overspnd on movies lol'"),
]


# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """\
You are building a synthetic finance dataset for a personal AI assistant.

Your task: write ONE realistic finance-related prompt that Ezaan would send to his AI.

You will be given:
1. REAL STYLE EXAMPLES — how Ezaan actually types (tone, typos, filler words, punctuation, length).
   Learn ONLY style. Do NOT copy or paraphrase these examples.
2. FINANCIAL ANCHOR — real data (transaction, order, or movie) to ground the prompt.
3. MODE — the type of prompt to write.
4. BANNED OPENERS — do not start with any of these words/phrases.

RULES:
- Write exactly ONE prompt. No explanation. No label. Just the prompt text.
- Match Ezaan's actual writing style from the examples.
- Ground the prompt in real data from the anchor — never hallucinate amounts or purchases.
- The prompt must be FINANCE-RELATED — even if the anchor is a movie, ask about budget/cost.
- Do not write markdown, bullet points, or JSON — just the raw prompt text.
- Vary length: some prompts should be 3 words, some 3 sentences, most somewhere between.
- Do NOT start with "I want" or "Can you tell me" — use Ezaan's more direct/casual style.

Return ONLY the raw prompt text. Nothing else.
"""


# ============================================================
# LOAD DATASETS
# ============================================================
CODE_HEAVY = re.compile(
    r"(def |import |Traceback|<html|SELECT \*|function\(|=>|\{\s*$)",
    re.MULTILINE,
)


def load_style_pool(csv_path: Path, min_len: int = 15, max_len: int = 500) -> list[str]:
    df = pd.read_csv(csv_path)
    pool = []
    for p in df["prompt"].dropna().astype(str):
        p = p.strip()
        if not (min_len <= len(p) <= max_len):
            continue
        if CODE_HEAVY.search(p):
            continue
        pool.append(p)
    return pool


def load_transactions(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "date":     str(row.get("date", ""))[:10],
            "purpose":  str(row.get("purpose", "")).strip(),
            "amount":   row.get("amount", 0),
            "type":     str(row.get("type", "")).strip(),
            "category": str(row.get("category", "")).strip(),
        })
    return records


def load_orders(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        item = str(row.get("item", "")).strip()
        if not item or item == "nan":
            continue
        # Truncate very long item names
        if len(item) > 80:
            item = item[:80] + "..."
        records.append({
            "vendor":    str(row.get("vendor", "")).strip(),
            "item":      item,
            "price":     row.get("price_pkr", 0),
            "status":    str(row.get("status", "")).strip(),
            "order_type": str(row.get("order_type", "")).strip(),
            "date":      str(row.get("date", "")).strip(),
        })
    return records


def load_movies(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "title":        str(row.get("title", "")).strip(),
            "studio":       str(row.get("studio", "")).strip(),
            "release_date": str(row.get("release_date", "")).strip(),
            "status":       str(row.get("status", "")).strip(),
            "notes":        str(row.get("notes", "")).strip()[:120],
        })
    return records


# ============================================================
# ANCHOR BUILDERS
# Each anchor = (source_type, short text shown to the LLM)
# ============================================================
def build_anchors(transactions, orders, movies) -> list[tuple[str, str]]:
    anchors = []

    # --- Transaction anchors ---
    for t in transactions:
        anchor = (
            f"Transaction: {t['purpose']} | {t['type']} | Rs{t['amount']} | {t['date']}"
        )
        anchors.append(("transaction", anchor))

    # --- Order anchors ---
    for o in orders:
        anchor = (
            f"Online order: {o['item']} from {o['vendor']} | "
            f"Rs{o['price']} | Status: {o['status']}"
        )
        anchors.append(("order", anchor))

    # --- Movie anchors ---
    for m in movies:
        anchor = (
            f"Upcoming movie: {m['title']} ({m['studio']}) | "
            f"Release: {m['release_date']} | Status: {m['status']}"
        )
        anchors.append(("movie", anchor))

    # --- Cross-dataset combos ---
    # Transactions + Orders
    for _ in range(40):
        t = random.choice(transactions)
        o = random.choice(orders)
        anchor = (
            f"Transaction: {t['purpose']} Rs{t['amount']} ({t['date']}) | "
            f"Order: {o['item']} Rs{o['price']}"
        )
        anchors.append(("tx_order_combo", anchor))

    # Transactions + Movies
    for _ in range(40):
        t = random.choice(transactions)
        m = random.choice(movies)
        anchor = (
            f"Transaction: {t['purpose']} Rs{t['amount']} ({t['date']}) | "
            f"Movie: {m['title']} (releasing {m['release_date']})"
        )
        anchors.append(("tx_movie_combo", anchor))

    # Orders + Movies
    for _ in range(30):
        o = random.choice(orders)
        m = random.choice(movies)
        anchor = (
            f"Order: {o['item']} Rs{o['price']} | "
            f"Movie: {m['title']} (releasing {m['release_date']})"
        )
        anchors.append(("order_movie_combo", anchor))

    # Category-level anchors (aggregate summaries for budget/trend prompts)
    categories = list({t["purpose"] for t in transactions if t["purpose"] != "nan"})
    for cat in categories:
        cat_txns = [t for t in transactions if t["purpose"] == cat]
        total = sum(t["amount"] for t in cat_txns)
        anchor = (
            f"Category: {cat} | Total: Rs{total:.0f} | "
            f"Entries: {len(cat_txns)} transactions"
        )
        anchors.append(("category_summary", anchor))

    random.shuffle(anchors)
    return anchors


# ============================================================
# STYLE SAMPLER (thread-safe, exhausting-shuffle)
# ============================================================
class StyleSampler:
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
                self._used.append(idx)
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


def _too_similar(a: str, b: str, thresh: float = 0.60) -> bool:
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= thresh


def _call_llm(user_prompt: str, retries: int = 2) -> str | None:
    payload = {
        "model":  MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "think":  False,
        "options": {"temperature": 1.05, "top_p": 0.95, "num_predict": 120},
    }
    for _ in range(retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            raw = _strip_think(r.json().get("response", "")).strip()
            # Strip surrounding quotes if any
            raw = raw.strip('"\'')
            if raw and len(raw) > 3:
                return raw
        except Exception:
            time.sleep(1)
    return None


# ============================================================
# GENERATE PROMPTS FOR ONE ANCHOR
# ============================================================
def _extract_source_uid(source_type: str, anchor_text: str) -> str:
    """Extract a short identifier from the anchor text for source_uid."""
    try:
        if source_type == "transaction":
            parts = anchor_text.split("|")
            date_part    = parts[-1].strip() if len(parts) >= 4 else ""
            purpose_part = parts[0].replace("Transaction:", "").strip() if parts else ""
            return f"{date_part}::{purpose_part}" if date_part else purpose_part
        elif source_type == "order":
            inner = anchor_text.replace("Online order:", "").split(" from ")[0].strip()
            return inner[:60]
        elif source_type == "movie":
            inner = anchor_text.replace("Upcoming movie:", "").split("(")[0].strip()
            return inner[:60]
        elif source_type == "category_summary":
            inner = anchor_text.replace("Category:", "").split("|")[0].strip()
            return f"category::{inner[:50]}"
        elif source_type in ("tx_order_combo", "tx_movie_combo", "order_movie_combo"):
            return source_type
    except Exception:
        pass
    return anchor_text[:60]


def _build_source_summary(source_type: str, anchor_text: str) -> str:
    """Build a human-readable source_summary from the anchor text."""
    try:
        if source_type == "transaction":
            parts   = [p.strip() for p in anchor_text.split("|")]
            purpose = parts[0].replace("Transaction:", "").strip()
            amount  = parts[2].strip() if len(parts) >= 3 else ""
            date    = parts[3].strip() if len(parts) >= 4 else ""
            return f"{purpose} {amount} on {date}"
        elif source_type in ("order", "movie", "category_summary",
                             "tx_order_combo", "tx_movie_combo", "order_movie_combo"):
            return anchor_text[:100]
    except Exception:
        pass
    return anchor_text[:100]


def generate_for_anchor(
    anchor: tuple[str, str],
    sampler: StyleSampler,
    modes: list[tuple[str, str]],
    n_per_anchor: int,
    style_n: int,
    global_prompts: list[str] | None = None,
    global_lock: "threading.Lock | None" = None,
) -> list[dict]:
    """Returns list of dicts with prompt + metadata (7 columns).

    global_prompts (optional): a shared, growing list of every prompt text
    generated so far across ALL anchors. When provided, new candidates are
    deduped against this global pool (not just the prompts generated for
    this one anchor), and lock-protected access is used since multiple
    anchors may run concurrently.
    """
    source_type, anchor_text = anchor
    source_uid     = _extract_source_uid(source_type, anchor_text)
    source_summary = _build_source_summary(source_type, anchor_text)

    generated: list[dict] = []
    banned: list[str] = []

    # Pick modes — bias certain source types to certain modes
    if source_type == "movie":
        eligible_modes = [m for m in modes if m[0] in (
            "entertainment_budget", "budget_check", "purchase_decision",
            "follow_up", "vague_incomplete", "typo_casual", "savings_planning"
        )]
    elif source_type == "order":
        eligible_modes = [m for m in modes if m[0] in (
            "shopping_history", "purchase_decision", "budget_check",
            "spending_check", "follow_up", "typo_casual", "vague_incomplete"
        )]
    elif source_type == "category_summary":
        eligible_modes = [m for m in modes if m[0] in (
            "trend_analysis", "monthly_report", "category_deep_dive",
            "net_flow", "comparison", "savings_planning", "budget_check"
        )]
    else:
        eligible_modes = modes  # transactions and combos use all modes

    if not eligible_modes:
        eligible_modes = modes

    for i in range(n_per_anchor):
        mode_name, mode_desc = eligible_modes[i % len(eligible_modes)]
        style_examples = sampler.sample(style_n)
        style_block = "\n".join(f"{j+1}. {ex}" for j, ex in enumerate(style_examples))

        already_block = ""
        if generated:
            already_block = (
                "\nALREADY GENERATED (write something STRUCTURALLY DIFFERENT):\n"
                + "\n".join(f"- {q['prompt']}" for q in generated[-3:])
            )

        banned_block = ""
        if banned:
            banned_block = (
                "\nDo NOT start with: "
                + ", ".join(f"'{b}'" for b in banned[-6:])
            )

        user_prompt = (
            f"FINANCIAL ANCHOR:\n{anchor_text}\n\n"
            f"STYLE EXAMPLES (learn voice only, do not copy):\n{style_block}\n\n"
            f"MODE ({mode_name}): {mode_desc}"
            f"{already_block}{banned_block}\n\n"
            "Write ONE finance prompt in Ezaan's real voice. Return ONLY the prompt text."
        )

        for _ in range(3):  # dedup retries
            candidate = _call_llm(user_prompt)
            if not candidate:
                continue

            # Dedup against this anchor's own outputs...
            if any(_too_similar(candidate, g["prompt"]) for g in generated):
                continue

            # ...and against every prompt generated so far, globally.
            if global_prompts is not None:
                if global_lock is not None:
                    with global_lock:
                        is_dupe = any(_too_similar(candidate, p) for p in global_prompts)
                        if not is_dupe:
                            global_prompts.append(candidate)
                else:
                    is_dupe = any(_too_similar(candidate, p) for p in global_prompts)
                    if not is_dupe:
                        global_prompts.append(candidate)
                if is_dupe:
                    continue

            generated.append({
                "prompt":         candidate,
                "category":       "AGENT",
                "subcategory":    "FINANCE_AGENT",
                "action_type":    mode_name,
                "message_mode":   mode_name,
                "source_uid":     source_uid,
                "source_summary": source_summary,
            })
            banned.append(_opener(candidate))
            break

    return generated


# ============================================================
# MAIN PIPELINE
# ============================================================
REQUIRED_COLS = ["prompt", "category", "subcategory", "action_type", "message_mode", "source_uid", "source_summary"]


def _load_existing(out_path: Path) -> list[dict]:
    """Load existing rows. Handles both old (prompt-only) and new (full-schema) format."""
    if not out_path.exists():
        return []
    try:
        df = pd.read_csv(out_path)
        if "prompt" not in df.columns:
            print("   ⚠️  Output file has no 'prompt' column; starting fresh.")
            return []
        # If old format (no metadata cols), migrate with unknown metadata
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = "unknown" if col != "category" else "AGENT"
        if "subcategory" in df.columns:
            df["subcategory"] = df["subcategory"].replace("unknown", "FINANCE_AGENT")
        rows = df[REQUIRED_COLS].dropna(subset=["prompt"]).to_dict("records")
        has_meta = all(c in pd.read_csv(out_path).columns for c in REQUIRED_COLS)
        fmt = "full schema" if has_meta else "migrated from prompt-only"
        print(f"   ♻️  Resuming: {len(rows)} prompts already saved ({fmt}).")
        return rows
    except Exception as e:
        print(f"   ⚠️  Could not load existing output ({e}); starting fresh.")
        return []


def main():
    print("⏳ Loading datasets...")
    style_pool   = load_style_pool(STYLE_CSV)
    transactions = load_transactions(TRANSACTIONS_CSV)
    orders       = load_orders(ORDERS_CSV)
    movies       = load_movies(MOVIES_CSV)

    print(f"   Style pool    : {len(style_pool)} examples")
    print(f"   Transactions  : {len(transactions)} rows")
    print(f"   Orders        : {len(orders)} rows")
    print(f"   Movies        : {len(movies)} titles")

    anchors = build_anchors(transactions, orders, movies)
    print(f"   Total anchors : {len(anchors)}")

    # How many prompts per anchor to hit TARGET_N
    n_per_anchor = max(1, min(6, TARGET_N // max(len(anchors), 1)))
    style_n = 5

    print(f"\n🎯 Target: ~{TARGET_N} prompts | {n_per_anchor} per anchor | {WORKERS} workers")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sampler = StyleSampler(style_pool)

    # Resume support
    all_rows: list[dict] = _load_existing(OUT_CSV)
    done_count = len(all_rows)

    # If we already have enough, exit
    if done_count >= TARGET_N:
        print(f"✅ Already have {done_count} prompts. Nothing to do.")
        return

    lock = threading.Lock()
    last_save = done_count
    all_prompts = [r["prompt"] for r in all_rows]  # global dedup pool, actually used now

    def generate_fn(anchor):
        return generate_for_anchor(
            anchor, sampler, FINANCE_MODES, n_per_anchor, style_n,
            global_prompts=all_prompts, global_lock=lock,
        )

    # Shuffle anchors for variety
    random.shuffle(anchors)
    # Only process as many anchors as needed, plus a small buffer for
    # anchors whose generation attempts fail outright.
    needed = TARGET_N - done_count
    max_anchors = (needed // n_per_anchor) + 10
    anchors = anchors[:max_anchors]

    def _save(rows: list[dict]) -> None:
        pd.DataFrame(rows, columns=REQUIRED_COLS).to_csv(OUT_CSV, index=False)

    # Process in small batches so we can stop submitting new work — and the
    # tqdm loop can break out — as soon as the target is actually reached,
    # instead of always running through every selected anchor regardless.
    #
    # The bar tracks PROMPTS toward TARGET_N (not anchors processed), seeded
    # with whatever was already saved on resume. This is what "progress"
    # actually means here: anchors yield a variable number of prompts (0 to
    # n_per_anchor, depending on dedup/failures), so an anchor-based bar
    # doesn't correlate with real progress and would also stall short of
    # 100% whenever we stop early after hitting the target.
    BATCH_SIZE = max(WORKERS * 4, 1)
    pbar = tqdm(
        total=TARGET_N,
        initial=done_count,
        desc="Generating finance prompts",
        unit="prompt",
        dynamic_ncols=True,
    )
    target_reached = False
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for batch_start in range(0, len(anchors), BATCH_SIZE):
                if target_reached:
                    break

                batch = anchors[batch_start:batch_start + BATCH_SIZE]
                futures = {ex.submit(generate_fn, a): a for a in batch}

                for future in as_completed(futures):
                    try:
                        new_rows = future.result()
                    except Exception as exc:
                        tqdm.write(f"  ⚠️  Worker error: {exc}")
                        continue

                    with lock:
                        for row in new_rows:
                            tqdm.write(f"  💬 {row['prompt'][:80]}")
                        all_rows.extend(new_rows)
                        # Advance by however many prompts this anchor actually
                        # produced, clamped so the bar never overshoots 100%.
                        remaining = max(pbar.total - pbar.n, 0)
                        pbar.update(min(len(new_rows), remaining))

                        if len(all_rows) - last_save >= SAVE_EVERY:
                            _save(all_rows)
                            last_save = len(all_rows)
                            tqdm.write(f"  💾 Auto-saved {len(all_rows)} prompts → {OUT_CSV.name}")

                        if len(all_rows) >= TARGET_N and not target_reached:
                            target_reached = True
                            tqdm.write(f"  🎯 Reached target of {TARGET_N} prompts — stopping.")

                    if target_reached:
                        # Let already-running futures in this batch finish,
                        # but don't start any further batches.
                        break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — saving progress...")
    finally:
        pbar.close()

    # Final save
    _save(all_rows)
    print(f"\n✅ Done! {len(all_rows)} finance prompts → {OUT_CSV}")


if __name__ == "__main__":
    main()