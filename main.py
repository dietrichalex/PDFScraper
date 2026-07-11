#!/usr/bin/env python3
"""
segment_pdfs.py

Scans a folder of PDFs (scientific articles/essays/responses, spanning
1950s-today, some OCR'd, some digital-native) and produces a CSV listing:
    source_file, title, authors, start_page, end_page

STRATEGY (v2 - batch/document-context approach):

The earlier version of this script sent ONE isolated page at a time to the
LLM, gated by a brittle font-size heuristic that decided which pages were
even worth checking. That has two structural problems: (1) the LLM has no
memory between calls, so it can't tell a genuine new article from a running
header/footer repeating an earlier title, and (2) any article the heuristic
failed to flag was silently never checked at all - a real cause of missed
articles.

This version instead:
  1. Builds a lightweight "digest" for EVERY page (page number, a text
     snippet, max font size, image coverage) - cheap, no LLM involved.
  2. Sends the LLM a BATCH of ~25 consecutive page digests at once (not
     single isolated pages), asking it to identify article starts using
     the whole batch as context. This lets it directly see things like
     "this exact title/byline reappears on pages 111, 112, and 118" and
     correctly conclude that's a running header, not new articles.
  3. Every page gets a chance to be considered - no upfront heuristic
     gate deciding what's even worth checking.
  4. Far fewer LLM calls overall than the old per-candidate-page approach.

Requires:
    pip install pymupdf requests

Assumes LM Studio is running its local server (default: Developer tab ->
"Start Server") at http://localhost:1234 with a model loaded.
"""

import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF
import requests

# --------------------------------------------------------------------------
# Config (tweak as needed)
# --------------------------------------------------------------------------

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "qwen2.5-7b"  # LM Studio typically uses whatever's loaded regardless of this field
REQUEST_TIMEOUT = 180

SNIPPET_CHARS = 400           # how much text per page goes into its digest (title + byline usually fit easily)
BATCH_SIZE = 25               # how many consecutive page digests get sent to the LLM per call
BATCH_OVERLAP = 3             # pages of overlap between consecutive batches, so an article start
                               # right at a batch boundary still has full context on both sides
MAX_TOKENS_RESPONSE = 1200    # a batch can contain several article starts, so give room for a longer JSON array

# How many batches to send to the local LLM concurrently.
# LM Studio's server (llama.cpp) can process multiple requests concurrently
# ONLY if you've enabled multiple parallel slots in the server settings
# (Developer tab -> server settings -> "Number of expected parallel requests"
# or similar, depending on version). If your server only has 1 slot, raising
# this won't speed anything up - the requests will just queue on the server
# side instead of in this script.
MAX_WORKERS = 2

# A page with a large image and almost no text is very likely an ad, a
# full-page photo, or a divider - not an article start. These are dropped
# from the digest entirely to save tokens (they essentially never matter).
AD_IMAGE_COVERAGE_THRESHOLD = 0.5
AD_MAX_TEXT_CHARS = 150

DEBUG = True  # print raw LLM replies / failure reasons to help diagnose issues

# Hardcoded folder structure (relative to this script's location):
#   ./data/     -> input PDFs
#   ./output/   -> output CSV
INPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_CSV = OUTPUT_DIR / "articles_index.csv"


# --------------------------------------------------------------------------
# Step 1: per-page text + layout extraction
# --------------------------------------------------------------------------

def get_page_text_ordered(page):
    """Return the page's text in visual reading order, plus the max font
    size found in the top ~60% of the page (a weak signal, not a gate).

    Sorting happens at the BLOCK level (not per-line): each text block's
    internal line order is left untouched (so a column's sentences stay
    intact and don't get scrambled), but blocks themselves are ordered by
    vertical band then left-to-right. This puts a full-width title block
    first (above the columns) without interleaving sentences from a 2+
    column body layout, which naive line-by-line sorting would do."""
    data = page.get_text("dict")
    blocks_info = []  # (block_y0, block_x0, [(text, size, y0), ...])

    for block in data.get("blocks", []):
        block_lines = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            size = max(s["size"] for s in spans)
            y0 = line["bbox"][1]
            block_lines.append((text, size, y0))
        if not block_lines:
            continue
        bbox = block.get("bbox", (0, 0, 0, 0))
        blocks_info.append((bbox[1], bbox[0], block_lines))

    blocks_info.sort(key=lambda b: (round(b[0] / 50) * 50, b[1]))

    lines = []
    for _, _, block_lines in blocks_info:
        lines.extend(block_lines)

    full_text = "\n".join(text for text, _, _ in lines)

    page_height = page.rect.height
    top_sizes = [size for _, size, y0 in lines if y0 < page_height * 0.6]
    max_font = max(top_sizes) if top_sizes else 0

    return full_text, max_font


def image_coverage_ratio(page):
    """Fraction of the page area covered by embedded images (0.0 - 1.0)."""
    page_area = page.rect.width * page.rect.height
    if not page_area:
        return 0.0
    total_img_area = 0.0
    for img in page.get_images(full=True):
        xref = img[0]
        for rect in page.get_image_rects(xref):
            total_img_area += rect.width * rect.height
    return min(total_img_area / page_area, 1.0)


def build_page_digests(doc):
    """Build a compact digest for every page worth considering. Pages that
    are essentially a full-page ad/photo/divider (heavy image coverage,
    almost no text) are dropped entirely - they're never article starts
    and would just waste tokens."""
    digests = []
    for page_num, page in enumerate(doc):
        full_text, max_font = get_page_text_ordered(page)
        char_count = len(full_text.strip())

        if char_count < AD_MAX_TEXT_CHARS:
            if image_coverage_ratio(page) > AD_IMAGE_COVERAGE_THRESHOLD:
                continue
        if char_count == 0:
            continue

        digests.append({
            "page": page_num + 1,  # 1-based, matches what we'll show the user/model
            "snippet": full_text[:SNIPPET_CHARS],
            "max_font": round(max_font, 1),
        })
    return digests


# --------------------------------------------------------------------------
# Step 2: batch confirmation + extraction via LM Studio
# --------------------------------------------------------------------------

BATCH_PROMPT_TEMPLATE = """You are analyzing consecutive pages from a scanned/digital PDF that contains \
a compilation of scientific articles, essays, and responses, ranging from the 1950s to today. \
Some text may contain OCR errors.

Below is a JSON array of page digests, one per page, each with:
  - "page": the page number
  - "snippet": the first ~{snippet_chars} characters of text on that page, in reading order
  - "max_font": the largest font size found near the top of that page (a WEAK signal only -
    running headers, pull-quotes, captions, and ads can also have large font, so don't trust
    this alone)

Your task: identify which pages are the START of a NEW, DISTINCT article, essay, or response -
as opposed to: a continuation of a previous piece, a repeated running header/footer, a table of
contents, an index, an advertisement, or a reference list.

IMPORTANT - use the fact that you can see many consecutive pages at once: if the same title or
author phrase reappears on multiple pages in this batch, that is almost always a running header
on continuation pages, not a new article each time - only the FIRST page where a given piece
truly begins should be reported as a start.

For each genuine start page, extract the title and author(s) from its snippet, correcting
obvious OCR typos where confident. Some pieces (e.g. short symposium responses or letters) may
only have an author byline and no distinct headline - in that case use a title of the form
"Response by AUTHOR NAME" instead of leaving title null. Only omit authors if you truly cannot
determine any.

Respond with ONLY a valid JSON array, no other text, in exactly this format:
[{{"page": <page number, exactly as given in the input>, "title": "string", "authors": ["string", ...]}}, ...]
If NO pages in this batch are genuine article starts, respond with an empty array: []

Page digests:
{digest_json}
"""


def call_local_llm_batch(digest_batch):
    prompt = BATCH_PROMPT_TEMPLATE.format(
        snippet_chars=SNIPPET_CHARS,
        digest_json=json.dumps(digest_batch, ensure_ascii=False),
    )
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS_RESPONSE,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] LLM request failed: {e}", file=sys.stderr)
        return None

    try:
        response_json = resp.json()
        message = response_json["choices"][0]["message"]
        content = message.get("content", "")
        finish_reason = response_json["choices"][0].get("finish_reason", "?")
    except (KeyError, IndexError, json.JSONDecodeError):
        print(f"  [!] Unexpected LLM response shape: {resp.text[:300]}", file=sys.stderr)
        return None

    parsed = parse_llm_json_array(content)
    if parsed is None and DEBUG:
        print(
            f"  [!] Could not parse JSON array from LLM reply (finish_reason={finish_reason}). Raw reply was:\n"
            f"      {content[:500]!r}",
            file=sys.stderr,
        )
    return parsed


def parse_llm_json_array(content):
    """Extract a JSON array from the model's reply, tolerating stray text,
    markdown fences, and reasoning-model <think>...</think> blocks."""
    content = content.strip()

    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if "<think>" in content and "</think>" not in content:
        return None  # truncated mid-thought

    content = re.sub(r"^```(json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()

    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, list) else None


# --------------------------------------------------------------------------
# Step 3: author sanitization + cross-batch merge safety net
# --------------------------------------------------------------------------

_JUNK_AUTHOR_VALUES = {"null", "none", "n/a", "na", "unknown", ""}


def sanitize_authors(raw_authors):
    """Some local models emit the literal string 'null' inside the authors
    array instead of a real JSON null. Strip those and other placeholder junk."""
    if not raw_authors:
        return []
    if isinstance(raw_authors, str):
        raw_authors = [raw_authors]
    cleaned = []
    for a in raw_authors:
        if not isinstance(a, str):
            continue
        a = a.strip()
        if a.lower() in _JUNK_AUTHOR_VALUES:
            continue
        cleaned.append(a)
    return cleaned


def _normalize_authors(authors):
    return {a.strip().lower() for a in authors if a and a.strip()}


def _authors_overlap(current, previous):
    cur_set = _normalize_authors(current)
    prev_set = _normalize_authors(previous)
    for c in cur_set:
        for p in prev_set:
            if c == p or (len(c) > 3 and c in p) or (len(p) > 3 and p in c):
                return True
    return False


def merge_duplicates(confirmed_starts):
    """The model already handles most running-header dedup itself (since it
    sees whole batches at once), but batch OVERLAP means the same true start
    can legitimately appear in two consecutive batch results. This pass:
      - drops exact duplicate page numbers (keeps the first occurrence)
      - merges immediately-adjacent entries that share an author or an
        identical title (a light safety net for boundary artifacts)
    confirmed_starts: list of (page_num_1_based, title, authors), any order.
    """
    if not confirmed_starts:
        return confirmed_starts

    # Dedupe exact same page number first (can happen from batch overlap)
    by_page = {}
    for page_num, title, authors in confirmed_starts:
        if page_num not in by_page:
            by_page[page_num] = (title, authors)
    ordered = sorted(by_page.items())  # [(page_num, (title, authors)), ...]

    merged = []
    for page_num, (title, authors) in ordered:
        matched = False
        if merged:
            m_page, m_title, m_authors = merged[-1]
            same_title = (
                title != "UNKNOWN TITLE"
                and m_title != "UNKNOWN TITLE"
                and title.strip().lower() == m_title.strip().lower()
            )
            same_authors = _authors_overlap(authors, m_authors)
            if same_title or same_authors:
                better_title = m_title if m_title != "UNKNOWN TITLE" else title
                better_authors = m_authors if m_authors else authors
                merged[-1] = (m_page, better_title, better_authors)
                matched = True
        if not matched:
            merged.append((page_num, title, authors))

    return merged


# --------------------------------------------------------------------------
# Step 4: assemble segments per PDF
# --------------------------------------------------------------------------

def make_batches(digests, batch_size, overlap):
    """Split the digest list into overlapping batches."""
    batches = []
    step = max(batch_size - overlap, 1)
    i = 0
    while i < len(digests):
        batch = digests[i:i + batch_size]
        if batch:
            batches.append(batch)
        if i + batch_size >= len(digests):
            break
        i += step
    return batches


def segment_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    digests = build_page_digests(doc)
    doc.close()

    print(f"  {len(digests)} page(s) with content out of {num_pages} total")

    batches = make_batches(digests, BATCH_SIZE, BATCH_OVERLAP)
    total = len(batches)
    print(f"  Split into {total} batch(es) of ~{BATCH_SIZE} pages")

    all_starts = []  # list of (page_num_1_based, title, authors)
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call_local_llm_batch, batch): idx for idx, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx = futures[future]
            done += 1
            sys.stdout.write(f"\r  Processing batch {done}/{total}...   ")
            sys.stdout.flush()
            try:
                result = future.result()
            except Exception as e:
                print(f"\n  [!] batch {idx + 1} failed: {e}", file=sys.stderr)
                continue

            if result is None:
                print(f"\n  [x] batch {idx + 1}: no usable response from LLM (see [!] above)")
                continue

            for item in result:
                if not isinstance(item, dict):
                    continue
                page_num = item.get("page")
                if not isinstance(page_num, int):
                    continue
                title = item.get("title") or "UNKNOWN TITLE"
                authors = sanitize_authors(item.get("authors"))
                all_starts.append((page_num, title, authors))

    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

    if not all_starts:
        # Fallback: nothing confirmed anywhere - treat whole file as one entry
        all_starts = [(1, pdf_path.stem, [])]

    confirmed_starts = merge_duplicates(all_starts)

    for page_num, title, authors in confirmed_starts:
        print(f"  [+] page {page_num}: \"{title}\"")

    # Build (title, authors, start_page, end_page)
    entries = []
    for i, (start, title, authors) in enumerate(confirmed_starts):
        end = (confirmed_starts[i + 1][0] - 1) if i + 1 < len(confirmed_starts) else num_pages
        entries.append({
            "source_file": pdf_path.name,
            "title": title,
            "authors": "; ".join(authors) if authors else "",
            "start_page": start,
            "end_page": end,
        })

    return entries


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

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
    total_files = len(pdf_paths)
    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n[{i}/{total_files}] Processing: {pdf_path.name}")
        try:
            entries = segment_pdf(pdf_path)
            all_entries.extend(entries)
        except Exception as e:
            print(f"  [!] Failed to process {pdf_path.name}: {e}", file=sys.stderr)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "title", "authors", "start_page", "end_page"])
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"\nDone. Wrote {len(all_entries)} entries to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()