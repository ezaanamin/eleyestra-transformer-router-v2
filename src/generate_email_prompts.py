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
MOVIES_CSV       = BASE_DIR / "data/raw/upcoming_marvel_dc_movies.csv"
OUT_CSV          = BASE_DIR / "data/processed/email_prompts.csv"

WORKERS    = 1      # Ollama processes 1 request at a time
SAVE_EVERY = 10     # save every 10 new prompts
TARGET_N   = 2000   # approximate target prompts


# ============================================================
# SYNTHETIC ANCHOR DATA (ML/Software/Job & AI/Tech)
# ============================================================
ML_SOFTWARE_PROJECTS = [
    "Elyestra Router Agent", "LLM Fine-tuning Pipeline", 
    "PyTorch Dataloader Optimization", "RAG Document Parser", 
    "Semantic Search API", "User Auth Microservice", 
    "React Dashboard HUD", "PostgreSQL Migration",
    "Model Inference Server", "Web Scraping Bot",
    "Dataset Builder Scripts", "FastAPI Backend Routes",
    "Rust Tauri Desktop Client", "Transformers Integration",
    "HuggingFace Model Deployment"
]

ML_SOFTWARE_DOCS = [
    "Architecture Design Doc", "API Schema Requirements", 
    "Sprint Planning Notes", "Q3 OKRs", 
    "Model Evaluation Report", "Security Audit Findings",
    "Performance Benchmarks", "Docker Compose Config",
    "Database Schema Migration Plan", "User Onboarding Flow"
]

AI_TECH_TOPICS = [
    "Llama 3 open source release", "Qwen 2.5 benchmark scores",
    "Agentic AI frameworks", "RAG vs Long-Context window",
    "Cursor AI IDE features", "OpenAI Sora capabilities",
    "Nvidia GPU shortage", "Fine-tuning vs Prompt Engineering",
    "Local LLM inference optimization", "Mixture of Experts architecture",
    "Gemini 1.5 Pro Context Window", "Claude 3.5 Sonnet Coding Abilities"
]

JOB_RELATED = [
    "Machine Learning Engineer application", "Senior Software Engineer interview",
    "Offer negotiation", "Reference request", "Portfolio feedback",
    "Resume update for ML roles", "System Design Interview Prep",
    "Technical Assessment follow-up", "Recruiter outreach reply"
]

# ============================================================
# EMAIL PROMPT MODES
# ============================================================
EMAIL_MODES = [
    ("send_email_work_req",
     "Ask to send an email to a colleague asking for requirements or updates on a specific software/ML project. Keep it natural."),

    ("send_email_work_doc",
     "Ask to send an email with an attached document, PR, or report related to Software Engineering or ML."),

    ("send_email_job",
     "Ask to send an email regarding a job application, interview follow-up, or career opportunity in the ML/AI space."),

    ("send_email_personal_marvel",
     "Ask to send an email to a friend discussing an upcoming Marvel movie, trailers, or planning to watch it."),

    ("send_email_personal_ai",
     "Ask to send an email to a friend or colleague about a recent AI/Tech trend, news, or paper."),

    ("check_emails_general",
     "Ask if there are any new or unread emails in the inbox today. Keep it short and casual."),

    ("check_emails_work",
     "Ask if a specific person replied or if there's an update regarding a software/ML project in the inbox."),

    ("check_emails_personal",
     "Ask if a friend replied to an email about a movie, AI trend, or general hangout."),

    ("reply_email_work",
     "Ask to draft a reply to an email about a project, giving a quick status update or acknowledging receipt."),

    ("summarize_emails",
     "Ask the assistant to summarize the unread emails or the emails received from a specific person/project/newsletter."),

    ("search_email",
     "Ask the assistant to find an old email about a specific topic, document, or from a specific person."),

    ("delete_spam_email",
     "Ask the assistant to clear out spam, delete promotional emails, or unsubscribe from a newsletter."),

    ("vague_email",
     "Write an intentionally incomplete or vague prompt about an email. E.g. 'did i get an email from', 'send an email to', 'check my mail'"),

    ("typo_casual_email",
     "Write a casual prompt with natural typos or abbreviations about checking or sending an email. E.g. 'chk my emails', 'snd that doc to ali', 'any mails from hr'"),
]


# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """\
You are building a synthetic email intent dataset for a personal AI assistant.

Your task: write ONE realistic email-related prompt that Ezaan would send to his AI.

You will be given:
1. REAL STYLE EXAMPLES — how Ezaan actually types (tone, typos, filler words, punctuation, length).
   Learn ONLY style. Do NOT copy or paraphrase these examples.
2. EMAIL ANCHOR — real data (a movie, a software project, or tech topic) to ground the prompt.
3. MODE — the type of email action to write.
4. BANNED OPENERS — do not start with any of these words/phrases.

RULES:
- Write exactly ONE prompt. No explanation. No label. Just the prompt text.
- Match Ezaan's actual writing style from the examples.
- Ground the prompt in the provided anchor — never hallucinate the core topic.
- The prompt must be EMAIL-RELATED (e.g. sending, checking, summarizing, replying to an email).
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


def load_movies(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        studio = str(row.get("studio", "")).strip()
        if "marvel" not in studio.lower() and "mcu" not in studio.lower() and "disney" not in studio.lower():
            continue
        records.append({
            "title":        str(row.get("title", "")).strip(),
            "studio":       studio,
            "release_date": str(row.get("release_date", "")).strip(),
            "status":       str(row.get("status", "")).strip(),
            "notes":        str(row.get("notes", "")).strip()[:120],
        })
    return records


# ============================================================
# ANCHOR BUILDERS
# ============================================================
def build_anchors(movies) -> list[tuple[str, str]]:
    anchors = []

    # --- Movie anchors (Marvel focus) ---
    for m in movies:
        anchor = (
            f"Upcoming Marvel Movie: {m['title']} | "
            f"Release: {m['release_date']} | Status: {m['status']}"
        )
        anchors.append(("movie", anchor))
        
    # --- Software/ML Projects ---
    for _ in range(100):
        proj = random.choice(ML_SOFTWARE_PROJECTS)
        anchor = f"Software/ML Project: {proj}"
        anchors.append(("work_project", anchor))

    # --- Work Documents ---
    for _ in range(100):
        doc = random.choice(ML_SOFTWARE_DOCS)
        proj = random.choice(ML_SOFTWARE_PROJECTS)
        anchor = f"Work Document: {doc} for project {proj}"
        anchors.append(("work_doc", anchor))

    # --- AI/Tech Topics ---
    for _ in range(100):
        topic = random.choice(AI_TECH_TOPICS)
        anchor = f"AI/Tech Trend: {topic}"
        anchors.append(("tech_topic", anchor))

    # --- Job Related ---
    for _ in range(80):
        job = random.choice(JOB_RELATED)
        anchor = f"Job Application/Career: {job}"
        anchors.append(("job_related", anchor))

    # --- Cross-dataset combos ---
    for _ in range(50):
        proj = random.choice(ML_SOFTWARE_PROJECTS)
        topic = random.choice(AI_TECH_TOPICS)
        anchor = f"Project: {proj} | Applying tech: {topic}"
        anchors.append(("work_tech_combo", anchor))

    random.shuffle(anchors)
    return anchors


# ============================================================
# STYLE SAMPLER
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
    try:
        if source_type == "movie":
            inner = anchor_text.replace("Upcoming Marvel Movie:", "").split("|")[0].strip()
            return f"movie::{inner[:60]}"
        elif source_type in ("work_project", "work_doc"):
            parts = anchor_text.split(":")
            return f"work::{parts[-1].strip()[:60]}"
        elif source_type == "tech_topic":
            parts = anchor_text.split(":")
            return f"tech::{parts[-1].strip()[:60]}"
        elif source_type == "job_related":
            parts = anchor_text.split(":")
            return f"job::{parts[-1].strip()[:60]}"
        elif source_type == "work_tech_combo":
            return "work_tech_combo"
    except Exception:
        pass
    return anchor_text[:60].replace(" ", "_")


def _build_source_summary(source_type: str, anchor_text: str) -> str:
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
    source_type, anchor_text = anchor
    source_uid     = _extract_source_uid(source_type, anchor_text)
    source_summary = _build_source_summary(source_type, anchor_text)

    generated: list[dict] = []
    banned: list[str] = []

    if source_type == "movie":
        eligible_modes = [m for m in modes if m[0] in (
            "send_email_personal_marvel", "check_emails_personal", "search_email",
            "vague_email", "typo_casual_email", "summarize_emails"
        )]
    elif source_type in ("work_project", "work_doc"):
        eligible_modes = [m for m in modes if m[0] in (
            "send_email_work_req", "send_email_work_doc", "check_emails_work",
            "reply_email_work", "summarize_emails", "search_email", 
            "vague_email", "typo_casual_email"
        )]
    elif source_type == "tech_topic":
        eligible_modes = [m for m in modes if m[0] in (
            "send_email_personal_ai", "check_emails_general", "search_email",
            "vague_email", "typo_casual_email", "summarize_emails"
        )]
    elif source_type == "job_related":
        eligible_modes = [m for m in modes if m[0] in (
            "send_email_job", "check_emails_work", "reply_email_work", 
            "search_email", "vague_email", "typo_casual_email"
        )]
    else:
        eligible_modes = modes

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
            f"EMAIL ANCHOR:\n{anchor_text}\n\n"
            f"STYLE EXAMPLES (learn voice only, do not copy):\n{style_block}\n\n"
            f"MODE ({mode_name}): {mode_desc}"
            f"{already_block}{banned_block}\n\n"
            "Write ONE email prompt in Ezaan's real voice. Return ONLY the prompt text."
        )

        for _ in range(3):
            candidate = _call_llm(user_prompt)
            if not candidate:
                continue

            if any(_too_similar(candidate, g["prompt"]) for g in generated):
                continue

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
                "subcategory":    "EMAIL_AGENT",
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
    if not out_path.exists():
        return []
    try:
        df = pd.read_csv(out_path)
        if "prompt" not in df.columns:
            print("   ⚠️  Output file has no 'prompt' column; starting fresh.")
            return []
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = "unknown" if col != "category" else "AGENT"
        if "subcategory" in df.columns:
            df["subcategory"] = df["subcategory"].replace("unknown", "EMAIL_AGENT")
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
    movies       = load_movies(MOVIES_CSV)

    print(f"   Style pool    : {len(style_pool)} examples")
    print(f"   Movies (Mvl)  : {len(movies)} titles")

    anchors = build_anchors(movies)
    print(f"   Total anchors : {len(anchors)}")

    n_per_anchor = max(1, min(10, TARGET_N // max(len(anchors), 1))) + 1
    style_n = 5

    print(f"\n🎯 Target: ~{TARGET_N} prompts | {n_per_anchor} per anchor | {WORKERS} workers")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sampler = StyleSampler(style_pool)

    all_rows: list[dict] = _load_existing(OUT_CSV)
    done_count = len(all_rows)

    if done_count >= TARGET_N:
        print(f"✅ Already have {done_count} prompts. Nothing to do.")
        return

    lock = threading.Lock()
    last_save = done_count
    all_prompts = [r["prompt"] for r in all_rows]

    def generate_fn(anchor):
        return generate_for_anchor(
            anchor, sampler, EMAIL_MODES, n_per_anchor, style_n,
            global_prompts=all_prompts, global_lock=lock,
        )

    random.shuffle(anchors)
    needed = TARGET_N - done_count
    max_anchors = (needed // max(1, n_per_anchor - 2)) + 20
    anchors = anchors[:max_anchors]

    def _save(rows: list[dict]) -> None:
        pd.DataFrame(rows, columns=REQUIRED_COLS).to_csv(OUT_CSV, index=False)

    BATCH_SIZE = max(WORKERS * 4, 1)
    pbar = tqdm(
        total=TARGET_N,
        initial=done_count,
        desc="Generating email prompts",
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
                        break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — saving progress...")
    finally:
        pbar.close()

    _save(all_rows)
    print(f"\n✅ Done! {len(all_rows)} email prompts → {OUT_CSV}")


if __name__ == "__main__":
    main()
