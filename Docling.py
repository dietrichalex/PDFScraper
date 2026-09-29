#!/usr/bin/env python3
"""
build_articles_index.py

Two-tier pipeline (table-based primary, heading-based fallback):

STRATEGY 1 - TOC table extraction (tried first):
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

STRATEGY 2 - Docling heading detection (only if strategy 1 finds too few
entries, e.g. no usable TOC table exists):
  1. Pull every text item Docling's layout model labeled SECTION_HEADER or
     TITLE - a trained model's judgment of "this is a heading", not a
     hand-rolled font-size heuristic. Each heading already carries its real
     PDF page number directly (no printed-page-vs-PDF-page offset problem
     here, unlike the TOC table path).
  2. Batch these candidates (with a little surrounding page text for
     context) to the local LLM, asking it to distinguish genuine new-article
     starts from mere subsection headings within an ongoing article, and to
     extract title/author(s) for confirmed starts.

Either way, results are written to a CSV: source_file, title, authors,
start_page, end_page.

Requires:
    pip install docling pandas requests

Assumes LM Studio is running its local server at http://localhost:1234
with a model loaded.
"""

import csv
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.labels import DocItemLabel

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "qwen2.5-7b"
REQUEST_TIMEOUT = 180

SEARCH_PAGES = 20        # only look at tables within the first N pages
MIN_ENTRIES = 3          # if fewer usable entries than this are found, bail rather than trust garbage

# How many PDFs to process concurrently. Each file's Docling conversion is
# GPU-bound - running several at once on a single GPU may cause contention
# and NOT actually speed things up (could even slow things down or error
# out depending on available VRAM). The LLM calls to LM Studio only benefit
# from this too if its server has multiple parallel slots enabled (see
# Developer tab -> server settings). Start at 1 (serial, safest) and only
# raise it if you've confirmed your GPU/LM Studio setup handles concurrent
# load well.
MAX_WORKERS = 1

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


def gather_row_texts(docling_document, search_pages):
    """Collect the raw combined text for every table row within the first
    `search_pages` pages, with cross-table/row dedup.

    Deliberately does NOT try to mechanically split off a trailing page
    number here - that assumption ("last cell = the page number, everything
    else = exactly one title") breaks in enough real-world ways that it's
    not worth hand-coding around: page numbers get glued onto the previous
    word with no space (e.g. "Gordon12"), sometimes come out corrupted
    (a single unrecognizable character), and a single detected table row
    can genuinely contain more than one distinct article. All of that is
    handled by the LLM in split_rows_via_llm() instead, which can recognize
    these patterns semantically rather than needing yet another regex for
    every new edge case.

    Also returns the set of 0-based page indices the tables came from -
    every title here was extracted FROM that page's own text, so it will
    always match there first if we don't explicitly exclude it during the
    later real-page search."""
    seen_keys = set()
    rows = []
    source_pages = set()

    for table in docling_document.tables:
        page_no = table.prov[0].page_no if table.prov else None
        if page_no is not None and page_no > search_pages:
            continue
        if page_no is not None:
            source_pages.add(page_no - 1)  # convert to 0-based
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

            if not unique_vals:
                continue

            combined = clean_raw_text(" ".join(unique_vals))
            if len(combined) < 5 or len(combined) > 3000:
                continue  # too short to be real, or wildly too long (corrupted/unrelated block)

            # Cross-table/row dedup: normalize the first few words as a key
            key_words = re.sub(r"[^a-z0-9\s]", "", combined.lower()).split()[:6]
            key = " ".join(key_words)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            rows.append(combined)

    return rows, source_pages

    return candidates, source_pages


# --------------------------------------------------------------------------
# Step 5: LLM-based title/subtitle/author splitting (one batched call)
# --------------------------------------------------------------------------

ROW_SPLIT_PROMPT = """Below is a JSON array of raw text blocks extracted from a magazine's \
table of contents. Each block may combine ONE OR MORE distinct articles - each with a title, \
author name(s), and often a subtitle and/or a paragraph-length abstract/summary - all run \
together with no clear punctuation separating these parts (a side effect of how the source \
table's cells were parsed). Text may contain OCR errors.

A page number for each article is usually present somewhere near the end of that article's \
portion of the text. Sometimes it got accidentally glued onto the last word with no space at \
all (e.g. "Gordon12" means the author is "Gordon" and the page number is 12) - watch for this \
and separate it out correctly.

For each raw text block, identify EVERY distinct article it contains (usually 1, but sometimes \
2 or more genuinely different articles ended up combined into one block) and for each one extract:
  - "title": the main headline - a short phrase
  - "subtitle": a short descriptive subtitle if clearly present, else null
  - "authors": author name(s) - a short proper-name phrase, NOT a sentence or clause
  - "printed_page": the page number for THIS specific article, as an integer, if you can
    confidently determine it - else null. Do NOT guess a page number you aren't confident about;
    null is the correct answer when it's missing, corrupted, or ambiguous.

CRITICAL: ignore any long paragraph of flowing prose (multiple full sentences describing what \
the article is about) - that is an abstract/summary, not a subtitle. Never include any part of \
it in "title", "subtitle", or "authors".

Respond with ONLY a valid JSON array, no other text, matching each input by its "index" (more \
than one object may share the same index, if that block contained multiple distinct articles):
[{{"index": <integer, exactly as given>, "title": "string", "subtitle": "string or null", "authors": ["string", ...], "printed_page": <integer or null>}}, ...]

Text blocks:
{blocks_json}
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


def split_rows_via_llm(rows):
    """Send every raw table-row text block to the LLM in one batched call.
    Returns a flat list of entries: {"title", "subtitle", "authors",
    "printed_page"} - printed_page may be None, and a single input row can
    produce more than one output entry (when it secretly contained more
    than one article)."""
    if not rows:
        return []
    blocks = [{"index": i, "text": r} for i, r in enumerate(rows)]
    prompt = ROW_SPLIT_PROMPT.format(blocks_json=json.dumps(blocks, ensure_ascii=False))
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4000,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content", "")
    except Exception as e:
        print(f"  [!] LLM request failed: {e}", file=sys.stderr)
        return []

    parsed = parse_llm_json_array(content)
    if parsed is None:
        if DEBUG:
            print(f"  [!] Could not parse split JSON. Raw reply:\n      {content[:3000]!r}", file=sys.stderr)
        return []

    entries = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        printed_page = item.get("printed_page")
        if not isinstance(printed_page, int):
            printed_page = None
        entries.append({
            "title": title,
            "subtitle": item.get("subtitle"),
            "authors": sanitize_authors(item.get("authors")),
            "printed_page": printed_page,
        })
    return entries


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

# --------------------------------------------------------------------------
# Fallback strategy: Docling heading detection (for files with no usable
# TOC table). Uses the layout model's own SECTION_HEADER/TITLE labels
# instead of a hand-rolled font-size heuristic - critically, this should
# avoid mistaking a repeated running header for a new article, since
# Docling labels those separately as PAGE_HEADER/PAGE_FOOTER. Each heading
# already carries its real page number directly, so unlike the TOC-table
# path there's no printed-page-vs-PDF-page offset to solve here.
# --------------------------------------------------------------------------

HEADING_BATCH_SIZE = 30
HEADING_BATCH_OVERLAP = 3

HEADING_BATCH_PROMPT = """You are analyzing a list of headings that a document layout model \
detected in a compilation of scientific articles, essays, and responses (spanning the 1950s to \
today). Each item below includes the heading text, the page it appears on, and a short snippet \
of that page's text for context.

Some of these headings genuinely start a new article/essay/response. Others are just a \
SUBSECTION heading within an ongoing article (e.g. "Part I", "The Early Years", "Conclusion") \
and must NOT be treated as a new article.

For each heading that IS a genuine new-article start, extract its title and author(s) from the \
page snippet, correcting obvious OCR typos where confident. Some pieces (e.g. short symposium \
responses) may only have an author byline and no distinct headline - use a title like \
"Response by AUTHOR NAME" in that case.

Respond with ONLY a valid JSON array, no other text, in exactly this format:
[{{"page": <page number, exactly as given>, "title": "string", "authors": ["string", ...]}}, ...]
Only include entries for headings that ARE genuine new-article starts - omit subsection headings entirely.

Headings:
{headings_json}
"""


def extract_heading_candidates(docling_document):
    """Pull every SECTION_HEADER/TITLE item Docling's layout model detected,
    in document order, with its page number. This is a trained model's
    judgment of "this is a heading", not a font-size heuristic."""
    candidates = []
    seen_keys = set()
    for item in docling_document.texts:
        label = getattr(item, "label", None)
        if label not in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            continue
        text = (item.text or "").strip()
        if len(text) < 3 or not item.prov:
            continue
        page_no = item.prov[0].page_no

        # Cheap insurance against any repeated heading text that slipped
        # through with a heading label anyway (e.g. a running header) -
        # keep only the first occurrence.
        key = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
        if key in seen_keys:
            continue
        seen_keys.add(key)

        candidates.append({"page": page_no, "text": text})
    return candidates


def make_batches(items, batch_size, overlap):
    batches = []
    step = max(batch_size - overlap, 1)
    i = 0
    while i < len(items):
        batch = items[i:i + batch_size]
        if batch:
            batches.append(batch)
        if i + batch_size >= len(items):
            break
        i += step
    return batches


def confirm_heading_batch(batch, page_texts):
    payload_items = []
    for h in batch:
        page_idx = h["page"] - 1
        snippet = page_texts[page_idx][:400] if 0 <= page_idx < len(page_texts) else ""
        payload_items.append({"page": h["page"], "heading": h["text"], "snippet": snippet})

    prompt = HEADING_BATCH_PROMPT.format(headings_json=json.dumps(payload_items, ensure_ascii=False))
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1500,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content", "")
    except Exception as e:
        print(f"  [!] LLM request failed: {e}", file=sys.stderr)
        return None

    parsed = parse_llm_json_array(content)
    if parsed is None and DEBUG:
        print(f"  [!] Could not parse heading-batch JSON. Raw reply:\n      {content[:2000]!r}", file=sys.stderr)
    return parsed


def segment_via_headings(docling_document, page_texts, num_pages):
    candidates = extract_heading_candidates(docling_document)
    print(f"  {len(candidates)} heading candidate(s) detected by Docling's layout model")
    if not candidates:
        return []

    batches = make_batches(candidates, HEADING_BATCH_SIZE, HEADING_BATCH_OVERLAP)
    all_starts = []
    for i, batch in enumerate(batches):
        print(f"  Confirming heading batch {i + 1}/{len(batches)}...")
        result = confirm_heading_batch(batch, page_texts)
        if not result:
            continue
        for item in result:
            if not isinstance(item, dict):
                continue
            page_no = item.get("page")
            if not isinstance(page_no, int):
                continue
            title = item.get("title") or "UNKNOWN TITLE"
            authors = sanitize_authors(item.get("authors"))
            all_starts.append((page_no - 1, title, authors))

    # Dedup by page (batches overlap on purpose, so the same confirmed start
    # can legitimately appear twice)
    by_page = {}
    for page_idx, title, authors in all_starts:
        if page_idx not in by_page:
            by_page[page_idx] = (title, authors)
    located = [(p, t, a) for p, (t, a) in sorted(by_page.items())]

    for page_idx, title, authors in located:
        print(f"  [+] page {page_idx + 1}: \"{title}\"")

    return located


def build_final_entries(pdf_path, located, num_pages):
    """Shared final stage for both the table-based and heading-based
    strategies: merge same-page duplicates and compute start/end ranges."""
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


def process_pdf(pdf_path, converter):
    print(f"Converting {pdf_path.name} with Docling...")
    result = converter.convert(str(pdf_path))
    doc = result.document
    num_pages = doc.num_pages()
    page_texts = [doc.export_to_text(page_no=n) for n in range(1, num_pages + 1)]

    # --- Strategy 1: TOC table extraction ---
    rows, source_pages = gather_row_texts(doc, SEARCH_PAGES)
    print(f"  {len(rows)} candidate row(s) from tables after cleaning/dedup")
    if source_pages:
        print(f"  Table page(s) excluded from title search: {sorted(p + 1 for p in source_pages)}")

    located = []
    if len(rows) >= MIN_ENTRIES:
        entries = split_rows_via_llm(rows)
        print(f"  LLM identified {len(entries)} article(s) across those rows")
        offset_estimate = None

        for entry in entries:
            title = entry["title"]
            subtitle = entry.get("subtitle")
            authors = entry.get("authors", [])
            display_title = f"{title}: {subtitle}" if subtitle else title
            printed_page = entry.get("printed_page")  # may be None - handled below

            search_hint = None
            if printed_page is not None and offset_estimate is not None:
                search_hint = printed_page - 1 + offset_estimate

            # CRITICAL: exclude the TOC/table source pages here. Every title
            # was extracted FROM that page's own text, so without this
            # exclusion it always matches there first (itself) instead of
            # the real article page.
            page_idx = locate_title_page(title, page_texts, source_pages, search_hint)

            if page_idx is None and printed_page is not None and offset_estimate is not None:
                guess = printed_page - 1 + offset_estimate
                if 0 <= guess < num_pages:
                    page_idx = guess
                    print(f"  [~] \"{display_title}\": using estimated offset (no direct text match)")

            if page_idx is None:
                print(f"  [x] \"{display_title}\": could not locate in document, skipping")
                continue

            if offset_estimate is None and printed_page is not None:
                offset_estimate = page_idx - (printed_page - 1)

            located.append((page_idx, display_title, authors))
            print(f"  [+] page {page_idx + 1}: \"{display_title}\"")

    if len(located) >= MIN_ENTRIES:
        print(f"  Table-based extraction succeeded ({len(located)} entries).")
        return build_final_entries(pdf_path, located, num_pages)

    # --- Strategy 2: Docling heading detection (fallback) ---
    print("  Table-based extraction insufficient - falling back to Docling heading detection...")
    located = segment_via_headings(doc, page_texts, num_pages)

    if len(located) < MIN_ENTRIES:
        print(f"  Only {len(located)} entries found via headings either - skipping this file.")
        return []

    return build_final_entries(pdf_path, located, num_pages)


_thread_local = threading.local()


def _get_thread_converter():
    """One Docling converter per worker thread (not per file) - reused
    across whichever files that thread processes, so GPU model weights
    only get loaded once per thread rather than once per PDF."""
    if not hasattr(_thread_local, "converter"):
        _thread_local.converter = build_converter()
    return _thread_local.converter


def _process_one(pdf_path):
    converter = _get_thread_converter()
    return pdf_path, process_pdf(pdf_path, converter)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    pdf_paths = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {INPUT_DIR}")
        return

    all_entries = []

    if MAX_WORKERS <= 1:
        converter = build_converter()
        for i, pdf_path in enumerate(pdf_paths, start=1):
            print(f"\n[{i}/{len(pdf_paths)}] {pdf_path.name}")
            try:
                all_entries.extend(process_pdf(pdf_path, converter))
            except Exception as e:
                print(f"  [!] Failed: {e}", file=sys.stderr)
    else:
        print(f"Processing with up to {MAX_WORKERS} concurrent workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_one, p): p for p in pdf_paths}
            done = 0
            for future in as_completed(futures):
                done += 1
                pdf_path = futures[future]
                print(f"\n[{done}/{len(pdf_paths)}] {pdf_path.name}")
                try:
                    _, entries = future.result()
                    all_entries.extend(entries)
                except Exception as e:
                    print(f"  [!] Failed: {e}", file=sys.stderr)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "title", "authors", "start_page", "end_page"])
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"\nDone. Wrote {len(all_entries)} entries to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()