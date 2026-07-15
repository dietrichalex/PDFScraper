#!/usr/bin/env python3
"""
build_articles_index.py

One linear pipeline, no fallback branching:
  1. Convert PDF with Docling (GPU-accelerated, OCR disabled since the text
     layer already exists).
  2. Collect every table Docling finds in the first N pages (a TOC can span
     several separate table objects across multiple pages).
  3. Clean each row: drop duplicate values within a row (handles Docling
     sometimes repeating a cell across what should be separate columns),
     then treat the last remaining cell as the page-number candidate
     (plain digit, Roman numeral, or common OCR-garbled digit pattern).
     Rows with no parseable page number are dropped (section headers,
     stray footer text, etc).
  4. Deduplicate rows across tables (handles the same TOC content
     occasionally appearing in more than one detected table).
  5. Send all the raw "everything but the page number" text blobs to the
     local LLM in ONE batched call, asking it to split each into
     title / subtitle / author(s) - there's no punctuation to split on
     mechanically since Docling's cell text runs them together.
  6. Ground each entry's printed page number against the real document
     text (search for the title) to find its actual PDF page.
  7. Write everything to a CSV: source_file, title, authors, start_page, end_page.

Requires:
    pip install docling pandas requests

Assumes LM Studio is running its local server at http://localhost:1234
with a model loaded.
"""

import csv
import json
import re
import sys
from pathlib import Path

import requests
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.datamodel.base_models import InputFormat

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "qwen2.5-7b"
REQUEST_TIMEOUT = 180

SEARCH_PAGES = 20        # only look at tables within the first N pages
MIN_ENTRIES = 3          # if fewer usable entries than this are found, bail rather than trust garbage

INPUT_DIR = Path(__file__).parent / "mini_data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_CSV = OUTPUT_DIR / "articles_index.csv"

DEBUG = True


# --------------------------------------------------------------------------
# Step 1: Docling conversion
# --------------------------------------------------------------------------

def build_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # text layer already exists on every input PDF
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})


# --------------------------------------------------------------------------
# Step 2-4: table extraction, row cleaning, dedup
# --------------------------------------------------------------------------

_ROMAN_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}


def try_parse_page_number(token):
    """Parse a table cell as a page number: plain digits, a (small) Roman
    numeral, or a common OCR-garbled digit pattern (e.g. stray leading
    apostrophe, letter/digit lookalike confusion)."""
    token = token.strip().strip(".,'\"")
    if not token:
        return None
    if re.fullmatch(r"\d{1,4}", token):
        return int(token)
    if token.upper() in _ROMAN_MAP:
        return _ROMAN_MAP[token.upper()]
    fixed = token.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    if re.fullmatch(r"\d{1,4}", fixed):
        return int(fixed)
    return None


def clean_raw_text(text):
    """Strip dot-leader noise (runs of periods used to visually connect a
    title to its page number) and collapse whitespace."""
    text = re.sub(r"\.{2,}", " ", text)
    text = text.replace(".", " ")  # old scans often glue a single stray dot onto a name/title too
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_candidate_rows(docling_document, search_pages):
    """Pull (raw_text, printed_page) candidates from every table in the
    first `search_pages` pages, with row-level and cross-table dedup."""
    candidates = []
    seen_keys = set()

    for table in docling_document.tables:
        page_no = table.prov[0].page_no if table.prov else None
        if page_no is not None and page_no > search_pages:
            continue
        try:
            df = table.export_to_dataframe(docling_document)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        for _, row in df.iterrows():
            # Drop duplicate values within the row (handles Docling
            # sometimes repeating one cell across what should be separate
            # columns), preserving first-seen order.
            unique_vals = []
            for val in row:
                val = str(val).strip()
                if not val or val.lower() == "nan":
                    continue
                if val not in unique_vals:
                    unique_vals.append(val)

            if len(unique_vals) < 2:
                continue  # need at least a text part and a page-number part

            printed_page = try_parse_page_number(unique_vals[-1])
            if printed_page is None:
                continue  # header row, footer blurb, or other non-entry row

            raw_text = clean_raw_text(" ".join(unique_vals[:-1]))
            if len(raw_text) < 5 or len(raw_text) > 250:
                continue  # too short to be real, or too long (a stray paragraph/quote block)

            # Cross-table/row dedup: normalize the first few words as a key
            key_words = re.sub(r"[^a-z0-9\s]", "", raw_text.lower()).split()[:6]
            key = " ".join(key_words)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            candidates.append({"raw_text": raw_text, "printed_page": printed_page})

    return candidates


# --------------------------------------------------------------------------
# Step 5: LLM-based title/subtitle/author splitting (one batched call)
# --------------------------------------------------------------------------

SPLIT_PROMPT = """Below is a JSON array of raw text snippets from a magazine's table of \
contents. Each snippet combines a title, an optional subtitle, and author name(s) all run \
together with no punctuation separating them (this is a side effect of how the source table's \
cells were parsed - there was no delimiter to split on mechanically). Text may contain OCR errors.

For each snippet:
  - "title": the main headline - almost always the FIRST few words, a short phrase
  - "subtitle": a longer descriptive subtitle, if present, else null - this is whatever falls
    between the title and the author name(s)
  - "authors": list of author name(s) - almost always at the very END of the snippet, else []

Respond with ONLY a valid JSON array, no other text, matching each input by its "index":
[{{"index": <integer, exactly as given>, "title": "string", "subtitle": "string or null", "authors": ["string", ...]}}, ...]

Snippets:
{snippets_json}
"""


def _repair_json_array_text(text):
    return re.sub(r",(\s*[\]}])", r"\1", text)


def parse_llm_json_array(content):
    content = content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if "<think>" in content and "</think>" not in content:
        return None
    content = re.sub(r"^```(json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return None
    array_text = match.group(0)
    try:
        result = json.loads(array_text)
    except json.JSONDecodeError:
        try:
            result = json.loads(_repair_json_array_text(array_text))
        except json.JSONDecodeError:
            return None
    return result if isinstance(result, list) else None


_JUNK_AUTHOR_VALUES = {"null", "none", "n/a", "na", "unknown", ""}


def sanitize_authors(raw_authors):
    if not raw_authors:
        return []
    if isinstance(raw_authors, str):
        raw_authors = [raw_authors]
    return [a.strip() for a in raw_authors if isinstance(a, str) and a.strip().lower() not in _JUNK_AUTHOR_VALUES]


def split_entries_via_llm(candidates):
    if not candidates:
        return {}
    snippets = [{"index": i, "text": c["raw_text"]} for i, c in enumerate(candidates)]
    prompt = SPLIT_PROMPT.format(snippets_json=json.dumps(snippets, ensure_ascii=False))
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 3000,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content", "")
    except Exception as e:
        print(f"  [!] LLM request failed: {e}", file=sys.stderr)
        return {}

    parsed = parse_llm_json_array(content)
    if parsed is None:
        if DEBUG:
            print(f"  [!] Could not parse split JSON. Raw reply:\n      {content[:3000]!r}", file=sys.stderr)
        return {}

    result = {}
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            result[item["index"]] = {
                "title": (item.get("title") or "").strip(),
                "subtitle": item.get("subtitle"),
                "authors": sanitize_authors(item.get("authors")),
            }
    return result


# --------------------------------------------------------------------------
# Step 6: ground each entry's page number against real document text
# --------------------------------------------------------------------------

def _normalize_for_match(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def locate_title_page(title, page_texts, skip_pages, search_hint=None):
    norm_title = _normalize_for_match(title)
    words = norm_title.split()
    if not words:
        return None
    key = " ".join(words[:6])
    if len(key) < 6:
        return None

    candidates = []
    for i, text in enumerate(page_texts):
        if i in skip_pages:
            continue
        if key in _normalize_for_match(text):
            candidates.append(i)

    if not candidates:
        return None
    if search_hint is not None:
        candidates.sort(key=lambda i: abs(i - search_hint))
    return candidates[0]


# --------------------------------------------------------------------------
# Main per-PDF pipeline
# --------------------------------------------------------------------------

def process_pdf(pdf_path, converter):
    print(f"Converting {pdf_path.name} with Docling...")
    result = converter.convert(str(pdf_path))
    doc = result.document
    num_pages = doc.num_pages()

    candidates = extract_candidate_rows(doc, SEARCH_PAGES)
    print(f"  {len(candidates)} candidate row(s) after cleaning/dedup")
    if len(candidates) < MIN_ENTRIES:
        print(f"  Too few candidates found ({len(candidates)}) - skipping this file.")
        return []

    split_by_index = split_entries_via_llm(candidates)

    page_texts = [doc.export_to_text(page_no=n) for n in range(1, num_pages + 1)]

    # Best-effort: assume any table page might be referenced, so nothing to
    # explicitly skip here (unlike the earlier pipeline, tables are usually
    # short enough that self-matching isn't a real risk).
    located = []
    offset_estimate = None

    for i, cand in enumerate(candidates):
        split = split_by_index.get(i, {})
        title = split.get("title") or cand["raw_text"]
        subtitle = split.get("subtitle")
        authors = split.get("authors", [])
        display_title = f"{title}: {subtitle}" if subtitle else title
        printed_page = cand["printed_page"]

        search_hint = printed_page - 1 + offset_estimate if offset_estimate is not None else None
        page_idx = locate_title_page(title, page_texts, set(), search_hint)

        if page_idx is None and offset_estimate is not None:
            guess = printed_page - 1 + offset_estimate
            if 0 <= guess < num_pages:
                page_idx = guess
                print(f"  [~] \"{display_title}\": using estimated offset (no direct text match)")

        if page_idx is None:
            print(f"  [x] \"{display_title}\": could not locate in document, skipping")
            continue

        if offset_estimate is None:
            offset_estimate = page_idx - (printed_page - 1)

        located.append((page_idx, display_title, authors))
        print(f"  [+] page {page_idx + 1}: \"{display_title}\"")

    if len(located) < MIN_ENTRIES:
        print(f"  Only {len(located)} entries could be located - not enough to trust this file.")
        return []

    located.sort(key=lambda t: t[0])
    deduped = []
    for page_idx, title, authors in located:
        if deduped and deduped[-1][0] == page_idx:
            prev_page, prev_title, prev_authors = deduped[-1]
            combined_title = f"{prev_title}: {title}" if title.lower() not in prev_title.lower() else prev_title
            combined_authors = prev_authors if prev_authors else authors
            deduped[-1] = (prev_page, combined_title, combined_authors)
        else:
            deduped.append((page_idx, title, authors))

    entries = []
    for i, (start, title, authors) in enumerate(deduped):
        end = (deduped[i + 1][0] - 1) if i + 1 < len(deduped) else num_pages - 1
        entries.append({
            "source_file": pdf_path.name,
            "title": title,
            "authors": "; ".join(authors) if authors else "",
            "start_page": start + 1,
            "end_page": end + 1,
        })

    return entries


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    pdf_paths = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {INPUT_DIR}")
        return

    converter = build_converter()
    all_entries = []
    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n[{i}/{len(pdf_paths)}] {pdf_path.name}")
        try:
            all_entries.extend(process_pdf(pdf_path, converter))
        except Exception as e:
            print(f"  [!] Failed: {e}", file=sys.stderr)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "title", "authors", "start_page", "end_page"])
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"\nDone. Wrote {len(all_entries)} entries to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()