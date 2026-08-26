import os
import sys
import pandas as pd
import requests
import json
import re
import logging
from tqdm import tqdm
from pathlib import Path

# ======================================================
# LOGGING
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs/labeling.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger("ElyestraRouterLabeler")

# ======================================================
# CONFIG
# ======================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:14b"

LABELS = ["GENERAL", "PERSONAL_CONTEXT", "CODING_CONTEXT", "AGENT"]
LABEL_SET = set(LABELS)

# Sentinel used to mark rows that failed classification, so they are NOT
# mistaken for a real "GENERAL" label and are NOT skipped on the next run.
ERROR_SENTINEL = "_ERROR_"

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": LABELS
        },
        "confidence": {"type": "number"},
        "reason": {"type": "string"}
    },
    "required": ["category", "confidence", "reason"]
}

# ======================================================
# SYSTEM PROMPT
# ======================================================
SYSTEM_PROMPT = """You are a dataset labeling system for a routing classifier.

Classify the user's query into exactly ONE of these labels:
GENERAL, PERSONAL_CONTEXT, CODING_CONTEXT, AGENT.

You must return the label string EXACTLY as written above. Never invent a new
label, never pluralize, never add/remove words, never guess a label that is
not in that list. There is no "unknown" or "none" option - every message
must be forced into the single best-fitting category below, even if the
message is short, ambiguous, or a mid-conversation fragment.

Ask yourself: "What context, memory, or system would be required to answer
this correctly?" Do not classify purely because a keyword (like a name,
brand, or topic word) appears in the text - a single word appearing inside a
long, mostly-unrelated message should NOT flip the label.

Note: you are labeling one message at a time with no conversation history.
Many messages are mid-conversation fragments (e.g. "correct??", "now what?",
"am i right or wrong"). Use whatever content is present (code syntax,
variable names, technical terms, error messages, data dumps, emotional tone,
personal references) to infer the most likely category - never leave a
message unclassified.

RULES OF THUMB
- Self-contained factual/how-it-works question, no user history needed,
  no action required -> GENERAL
- Depends on the user's own goals, plans, projects, feelings, opinions,
  or past decisions to answer well -> PERSONAL_CONTEXT
- Contains code, a traceback, technical/data fragments (variable names,
  loops, error messages, raw data dumps), or asks to write/fix/implement
  software -> CODING_CONTEXT
- Requires the assistant to DO or PRODUCE something task-specific, or covers
  a specialized domain that's really "help me accomplish X" rather than
  "explain X in the abstract" -> AGENT
    Includes: writing/organizing an email, scheduling/reminders/meetings/dates,
    saving/organizing notes or documents, resume/CV/LinkedIn/interview prep,
    personal spending/income/budget/transactions, comic/superhero-specific
    content requests, AI/ML research paper or architecture explanations tied
    to a concrete task or output, GENERATING media (images, presentations,
    slides, documents) even with no code involved, and requests to SEARCH or
    LOOK UP information as an action ("search the web and tell me...").
- A data dump (list of items, URLs, domains, etc.) followed by a transform
  instruction ("convert this", "fix this", "reformat this") is CODING_CONTEXT
  even if the instruction doesn't explicitly say "write code" - converting,
  reformatting, or restructuring raw data is a coding/technical task.

FALLBACK RULES FOR AMBIGUOUS OR FRAGMENTARY TEXT
- Bare acknowledgments with zero other content ("yes", "ok do it", "sure",
  "write") -> GENERAL
- A short reaction/confirmation fragment ("am i right or wrong", "correct??",
  "now what?") that carries no topic signal on its own -> look at tone and
  phrasing only, not invented context. If it reads like a technical check-in
  (short, clipped, no emotional language) -> CODING_CONTEXT. If it reads like
  a personal/reflective check-in (first-person feelings, self-doubt,
  life-decision framing) -> PERSONAL_CONTEXT. If genuinely coin-flip -> GENERAL.
- Raw data dumps, logs, or lists with NO instruction attached at all (just
  pasted text, no question or ask) -> CODING_CONTEXT, since pasting
  logs/data/lists is characteristic of a technical work session.
- Garbled, misspelled, or nonsensical short text with no technical or
  personal signal (e.g. "asdkjfh 2939", "I am batman you can't get je
  illuminati") -> GENERAL, since it cannot be routed to a specialized
  system and GENERAL is the safest default catch-all.

PRIORITY WHEN MULTIPLE COULD APPLY
1. CODING_CONTEXT wins if code/traceback/technical data is present, even if
   the coding task itself serves a personal project or research goal.
2. Otherwise, AGENT wins if a concrete action, generation task, search
   request, or specialized task is being requested (write this email,
   schedule this, track this expense, prep this resume, generate this image,
   build this presentation, explain this paper so I can use it, etc.)
3. Otherwise, if the answer depends on the user's own history/goals/feelings
   with no concrete task attached -> PERSONAL_CONTEXT
4. GENERAL is the fallback for everything else - self-contained abstract
   questions, bare acknowledgments, and genuinely unclassifiable text.

EXAMPLES
"Explain Java" -> GENERAL
"Should I learn Java based on my goals?" -> PERSONAL_CONTEXT
"Fix my Java error" -> CODING_CONTEXT
"How do transformers work?" -> GENERAL
"Explain this paper's architecture so I can reimplement it" -> AGENT
"Implement a transformer model in PyTorch" -> CODING_CONTEXT
"I want to build my own AI assistant someday" -> PERSONAL_CONTEXT
"Create my AI assistant backend" -> CODING_CONTEXT
"Can Doctor Doom defeat Thanos?" -> AGENT
"I mentioned Marvel once while talking about my family" -> PERSONAL_CONTEXT
"Draft an email to my boss" -> AGENT
"Remind me about my dentist appointment Friday" -> AGENT
"Create an image of a sunset over mountains" -> AGENT
"Build a presentation from this abstract" -> AGENT
"Search the web and tell me if this API is broken" -> AGENT
"domain1.com domain2.com domain3.com [long list] convert into url" -> CODING_CONTEXT
"num_trials num_channels where do i get theser two??" -> CODING_CONTEXT
"[pasted raw log/traceback with no question attached]" -> CODING_CONTEXT
"am i right or wrong" (no other context) -> GENERAL
"asdkjfh 2939 ???" -> GENERAL

Respond strictly in JSON format.
"""

# ======================================================
# DETERMINISTIC PRE-FILTER
# ======================================================
CODE_SIGNAL_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"^\s*(import|from)\s+\w+",
    r"\bdef\s+\w+\s*\(",
    r"\basync\s+def\s+\w+",
    r"File \"<ipython-input",
    r"^\s*(async\s+for|async\s+with)\b",
    r"^\w+@[\w-]+:.*\$\s",              # shell prompt e.g. user@host:~/path$
    r"\b(npm|npx|pip|python3?|gradlew)\b.*\b(install|start|run|clean)\b",
    r"\bError\b.*\bat\b.*:\d+:\d+",     # stack-trace-style "Error ... at file:line:col"
    r"^\s*(SyntaxError|TypeError|ValueError|RuntimeError|AttributeError|KeyError|IndexError):",
]
_code_regexes = [re.compile(p, re.MULTILINE) for p in CODE_SIGNAL_PATTERNS]

# A single hit on any of these is unambiguous on its own - no need for 2.
_strong_solo_patterns = [
    r"Traceback \(most recent call last\)",
    r"^\s*(SyntaxError|TypeError|ValueError|RuntimeError|AttributeError|KeyError|IndexError):",
]
_strong_regexes = [re.compile(p, re.MULTILINE) for p in _strong_solo_patterns]

def prefilter(prompt: str):
    if not isinstance(prompt, str):
        return None
    if any(rx.search(prompt) for rx in _strong_regexes):
        return "CODING_CONTEXT", 0.95, "Deterministic match: unambiguous traceback/error signature"
    hits = sum(1 for rx in _code_regexes if rx.search(prompt))
    if hits >= 2:
        return "CODING_CONTEXT", 0.95, "Deterministic match: prompt contains code/traceback structure"
    return None

# ======================================================
# OLLAMA CLASSIFIER
# ======================================================
def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def classify(prompt: str, retries: int = 2):
    pre = prefilter(prompt)
    if pre:
        return pre

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": f"PROMPT:\n{prompt}",
        "stream": False,
        "format": JSON_SCHEMA,
        "think": False,  # CRITICAL: disable Qwen3 reasoning pass, or num_predict
                          # gets consumed by <think> tokens before the JSON answer
        "options": {"temperature": 0.0, "num_predict": 300},
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            raw = response.json().get("response", "")
            cleaned = _strip_think_tags(raw)

            if not cleaned:
                last_error = "Empty response from model"
                continue  # retry - do NOT silently fall back yet

            data = json.loads(cleaned)
            category = str(data.get("category", "")).strip().upper()

            if category not in LABEL_SET:
                last_error = f"Model returned invalid category: {category!r}"
                continue  # retry

            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", ""))
            return category, max(0.0, min(1.0, confidence)), reason

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            continue

    # All retries exhausted - mark as a real failure, NOT a fake GENERAL label,
    # so this row is retried (not skipped) the next time the script runs.
    logger.error(f"Classification permanently failed after {retries + 1} attempts: {last_error}")
    return ERROR_SENTINEL, 0.0, f"FAILED after {retries + 1} attempts: {last_error}"

# ======================================================
# MAIN
# ======================================================
def main():
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1]).resolve()
    else:
        default_path = BASE_DIR / "data/raw/batch/unknown.csv"
        if not default_path.exists():
            default_path = BASE_DIR / "data/raw/unknown.csv"
        input_file = default_path

    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2]).resolve()
    else:
        output_file = BASE_DIR / "data/processed/unknown_labeled.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting Elyestra router labeling")
    print(f"   Model : {MODEL}")
    print(f"   Input : {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Allowed Labels: {LABELS}")
    logger.info(f"Starting | Model={MODEL} | Input={input_file} | Labels={LABELS}")

    print("\n⏳ Loading input file...")
    try:
        df = pd.read_csv(input_file, dtype={"prompt": str})
    except Exception as e:
        print(f"⚠️ Error reading CSV ({e}). Attempting recovery...")
        df = pd.read_csv(input_file, dtype={"prompt": str}, engine="python", on_bad_lines="skip")

    for col in ["category", "reason"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype("string").fillna("")

    if "confidence" not in df.columns:
        df["confidence"] = float("nan")
    else:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    print(f"   Loaded {len(df)} rows from input")

    # Re-queue any rows previously marked as failures so they get retried.
    n_prev_errors = (df["category"].astype(str).str.upper() == ERROR_SENTINEL).sum()
    if n_prev_errors:
        print(f"   Found {n_prev_errors} rows previously marked as failed - will retry these")

    pbar = tqdm(df.iterrows(), total=len(df), desc="Labeling", unit="row", dynamic_ncols=True)

    try:
        for idx, row in pbar:
            prompt = str(row["prompt"])

            # Skip only if category is already one of the 4 valid allowed labels.
            # Rows marked ERROR_SENTINEL are NOT skipped - they get retried.
            cat_str = str(row.get("category", "")).strip().upper()
            if cat_str in LABEL_SET:
                continue

            category, confidence, reason = classify(prompt)

            df.at[idx, "category"] = category
            df.at[idx, "confidence"] = confidence
            df.at[idx, "reason"] = reason

            msg = f"ROW {idx} | {category} ({confidence:.2f}) | Length: {len(prompt)} chars"
            tqdm.write(msg)

            logger.info(
                f"ROW: {idx} | CATEGORY: {category} | CONFIDENCE: {confidence} | "
                f"REASON: {reason} | PROMPT_LENGTH: {len(prompt)}"
            )

            # Save every 25 rows instead of 100 - smaller loss window if the
            # process dies or the machine is interrupted mid-run.
            if (idx + 1) % 25 == 0:
                df.to_csv(output_file, index=False)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted - saving progress before exiting...")
        df.to_csv(output_file, index=False)
        print(f"💾 Progress saved to {output_file}")
        sys.exit(1)

    print(f"\n💾 Saving labeled data to {output_file}...")
    df.to_csv(output_file, index=False)

    n_failed = (df["category"].astype(str).str.upper() == ERROR_SENTINEL).sum()
    print("✅ Done!")
    if n_failed:
        print(f"⚠️ {n_failed} rows still marked as failed ({ERROR_SENTINEL}) - rerun the script "
              f"on {output_file} to retry just these rows.")

    print("\n📊 Dataset Statistics:")
    print("Dataset shape:", df.shape)
    print("Classes:", df["category"].nunique())
    if "category" in df.columns:
        print("\nClass Distribution:")
        print(df["category"].value_counts())
    print("\nMissing Values:")
    print(df.isna().sum())
    logger.info("=========================")
    logger.info("LABELING COMPLETE")
    logger.info(f"Saved: {output_file}")
    logger.info(f"Rows still failed: {n_failed}")

if __name__ == "__main__":
    main()