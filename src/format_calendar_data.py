"""
format_calendar_data.py
=======================
Reads the raw exported Google Calendar JSON (formatted_data.json) and
produces a clean, day-oriented version (formatted_data_v2.json) that
generate_agent_data.py will consume.

Output schema
-------------
{
  "days": {
    "2024-03-15": {
      "pretty_date": "March 15, 2024",
      "events": [
        {"time": "9:00 AM", "summary": "Team sync", "location": "", "description": ""},
        {"time": "All day", "summary": "Project deadline", ...}
      ]
    },
    ...
  },
  "birthdays": [
    {"summary": "Ezaan's Birthday", "date": "January 5", "year": 2024},
    ...
  ],
  "movies": [
    {"summary": "Avengers: Endgame (2019)", "release_date": "April 26, 2019",
     "location": "", "description": ""},
    ...
  ]
}

Rules applied
-------------
- Each calendar source (*.ics) is merged into one event pool.
- Events with no parseable DTSTART are skipped.
- Birthdays and movies go into their own flat lists (deduplicated by
  normalised summary text, keeping at most 3 recurrences).
- All other events are grouped by date (YYYY-MM-DD).  Days that end up
  with only 1 event are still kept (generate_agent_data can filter those
  if it wants fewer-than-N days).
- Times are human-readable ("9:00 AM", "All day") - never raw ICS strings.
- Descriptions are trimmed to 300 chars to avoid giant prompt blobs.

Usage
-----
    python3 src/format_calendar_data.py
    # produces data/raw/formatted_data_v2.json
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH  = BASE_DIR / "data/raw/formatted_data.json"
OUTPUT_PATH = BASE_DIR / "data/raw/formatted_data_v2.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(raw: str):
    """Return (datetime, has_time) or (None, False)."""
    if not raw:
        return None, False
    raw = raw.strip()
    if ":" in raw:
        raw = raw.split(":")[-1]   # strip 'VALUE=DATE:' prefix
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(raw, fmt), True
        except ValueError:
            continue
    try:
        return datetime.strptime(raw, "%Y%m%d"), False
    except ValueError:
        return None, False


def _human_time(dt: datetime, has_time: bool) -> str:
    if not has_time:
        return "All day"
    return dt.strftime("%I:%M %p").lstrip("0")   # "9:00 AM"


def _human_date(dt: datetime) -> str:
    return dt.strftime("%B %d, %Y").replace(" 0", " ")  # "March 5, 2024"


def _birthday_date(dt: datetime, has_time: bool) -> str:
    """Returns 'March 5' (month + day only, no year) for birthday labels."""
    return dt.strftime("%B %d").replace(" 0", " ")


BIRTHDAY_MARKERS = ("birthday", "brithday", "briday", "bday", "b-day", "b day")
MOVIE_YEAR_RE = re.compile(r"\(\d{4}\)")

# ----- Release-date keyword variants (typos included) ---------------------
RELEASE_DATE_MARKERS = (
    "release date", "relase date", "releae date", "relaese date",
    "relesse date", "rellade date",   # 'rellade' seen in raw data
    "releasing",
)

# ----- Known movie / TV titles that appear as stand-alone watch events ----
# These are scheduled "watch this film tonight" calendar blocks.
MOVIE_TITLE_KEYWORDS = (
    # MCU films
    "iron man", "incredible hulk", "thor", "avengers", "captain america",
    "captain marvel", "black widow", "black panther", "ant man", "ant-man",
    "guardians of the galaxy", "age of ultron", "civil war", "homecoming",
    "doctor strange", "infinity war", "endgame", "dark world", "ragnarok",
    "winter soldier", "winter soilder",                # 'soilder' = typo
    "far from home", "no way home", "multiverse of madness",
    "love and thunder", "wakanda forever", "quantumania", "eternals",
    "shang-chi", "venom", "aquaman", "aquman",
    # DC / other
    "dark knight", "batman", "superman",
    # Generic markers that only appear in movie-watch events
    "blue ray", "blu ray", "blu-ray",
)

# ----- MCU news / tracker events -----------------------------------------
# These are calendar notes about MCU universe events, not viewings.
MCU_NEWS_KEYWORDS = (
    "mcu", "disney bought", "disney buys",
    "x men in mcu", "ff and x men", "fantastic four",
    "shield returning", "agents of shield", "agent of shield",
    "infinity war tril", "infity war tril",  # trailer/teaser tracking notes
    "rellade",             # 'new rellade date' typo
    "introducing back",
)


def _classify(summary: str, location: str = "", description: str = "") -> str:
    """Returns 'birthday', 'movie', or 'other'.

    'movie' covers:
      - events with a year in parentheses  e.g. "Doctor Strange (2016)"
      - events with 'release date' / typo variants
      - known movie/TV title keywords  (watch-schedule blocks)
      - MCU news / tracker events
      - IMDB-linked events
      - events at location 'in theaters'
    """
    s    = (summary     or "").lower()
    loc  = (location    or "").lower()
    desc = (description or "").lower()

    # --- Birthday ---
    if any(m in s for m in BIRTHDAY_MARKERS):
        return "birthday"

    # --- Movie: explicit markers ---
    if MOVIE_YEAR_RE.search(summary or ""):       # "Doctor Strange (2016)"
        return "movie"
    if loc == "in theaters":
        return "movie"
    if "imdb" in desc:
        return "movie"
    if any(m in s for m in RELEASE_DATE_MARKERS): # 'release date' + typos
        return "movie"

    # --- Movie: known titles (watch-schedule blocks) ---
    if any(kw in s for kw in MOVIE_TITLE_KEYWORDS):
        return "movie"

    # --- Movie: MCU news / tracker events ---
    if any(kw in s for kw in MCU_NEWS_KEYWORDS):
        return "movie"

    return "other"


def _dtstart(event: dict) -> str:
    """Retrieve DTSTART regardless of whether it appears as
    'DTSTART' or 'DTSTART;VALUE=DATE' in the original ICS export."""
    return (
        event.get("DTSTART")
        or event.get("DTSTART;VALUE=DATE")
        or ""
    )


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def build_structured(raw_path: Path) -> dict:
    with open(raw_path, encoding="utf-8") as f:
        data = json.load(f)

    # Collect all events from all *.ics calendars
    all_events: list[dict] = []
    for cal_name, payload in data.items():
        if not isinstance(payload, dict):
            continue
        events = payload.get("events") or []
        for e in events:
            e["_cal_source"] = cal_name
            all_events.append(e)

    days: dict[str, list[dict]] = defaultdict(list)
    birthdays_raw: list[dict] = []
    movies_raw: list[dict] = []

    for e in all_events:
        summary = (e.get("SUMMARY") or "").strip()
        if not summary:
            continue

        location    = (e.get("LOCATION")    or "").strip()
        description = (e.get("DESCRIPTION") or "").strip()
        if len(description) > 300:
            description = description[:300] + "…"

        start_raw = _dtstart(e)
        dt, has_time = _parse_dt(start_raw)
        if dt is None:
            continue  # can't place it on a day, skip

        category = _classify(summary, location, description)

        if category == "birthday":
            birthdays_raw.append({
                "summary":  summary,
                "date":     _birthday_date(dt, has_time),
                "year":     dt.year,
                "location": location,
                "description": description,
                "_sort_key": dt,
            })
        elif category == "movie":
            # Determine sub-kind so generate_agent_data can phrase prompts correctly:
            #   release_date  – has 'release date' or IMDB / (YYYY) marker
            #   watch_schedule – a personal "watch this film tonight" block
            #   news          – MCU news / tracker (Disney bought fox etc.)
            s_low = summary.lower()
            if (MOVIE_YEAR_RE.search(summary) or "imdb" in (description or "").lower()
                    or (location or "").lower() == "in theaters"
                    or any(m in s_low for m in RELEASE_DATE_MARKERS)):
                kind = "release_date"
                date_field = "release_date"
            elif any(kw in s_low for kw in MCU_NEWS_KEYWORDS):
                kind = "news"
                date_field = "event_date"
            else:
                kind = "watch_schedule"
                date_field = "watch_date"

            movies_raw.append({
                "summary":   summary,
                "kind":      kind,
                date_field:  _human_date(dt),
                "location":  location,
                "description": description,
                "_sort_key": dt,
            })
        else:
            date_key = dt.strftime("%Y-%m-%d")
            days[date_key].append({
                "time":        _human_time(dt, has_time),
                "summary":     summary,
                "location":    location,
                "description": description,
                "_sort_key":   dt,
            })

    # ---- Sort events within each day by time ----
    structured_days: dict[str, dict] = {}
    for date_key in sorted(days.keys()):
        day_events = sorted(days[date_key], key=lambda ev: ev["_sort_key"])
        pretty = day_events[0]["_sort_key"].strftime("%B %d, %Y").replace(" 0", " ")
        clean_events = [
            {k: v for k, v in ev.items() if k != "_sort_key"}
            for ev in day_events
        ]
        structured_days[date_key] = {
            "pretty_date": pretty,
            "events":      clean_events,
        }

    # ---- Deduplicate birthdays (keep ≤3 recurrences per normalised name) ----
    seen_birthdays: dict[str, int] = {}
    clean_birthdays: list[dict] = []
    for b in sorted(birthdays_raw, key=lambda x: x["_sort_key"]):
        key = b["summary"].lower().strip()
        seen_birthdays[key] = seen_birthdays.get(key, 0) + 1
        if seen_birthdays[key] <= 3:
            clean_birthdays.append({k: v for k, v in b.items() if k != "_sort_key"})

    # ---- Deduplicate movies similarly ----
    seen_movies: dict[str, int] = {}
    clean_movies: list[dict] = []
    for m in sorted(movies_raw, key=lambda x: x["_sort_key"]):
        key = m["summary"].lower().strip()
        seen_movies[key] = seen_movies.get(key, 0) + 1
        if seen_movies[key] <= 3:
            clean_movies.append({k: v for k, v in m.items() if k != "_sort_key"})

    return {
        "days":      structured_days,
        "birthdays": clean_birthdays,
        "movies":    clean_movies,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"📂 Reading  : {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print(f"❌ Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    structured = build_structured(INPUT_PATH)

    n_days      = len(structured["days"])
    n_events    = sum(len(d["events"]) for d in structured["days"].values())
    n_birthdays = len(structured["birthdays"])
    n_movies    = len(structured["movies"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"✅ Written  : {OUTPUT_PATH}")
    print(f"   Days     : {n_days}  ({n_events} events across all days)")
    print(f"   Birthdays: {n_birthdays}")
    print(f"   Movies   : {n_movies}")


if __name__ == "__main__":
    main()
