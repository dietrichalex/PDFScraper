#!/usr/bin/env python3
"""
segment_pdfs.py

Scans a folder of PDFs (scientific articles/essays/responses, spanning
1950s-today, some OCR'd, some digital-native) and produces a CSV listing:
    source_file, title, authors, start_page, end_page

STRATEGY (v3 - table-of-contents-first):

Magazines like Foreign Affairs already print a table of contents listing
every article's title, author, and starting page number. Trying to
reconstruct that same structure by scanning the entire body of the document
page-by-page is solving a much harder problem than necessary when the
answer is usually sitting on page 1 or 2. So:

  1. PRIMARY: find the table-of-contents page(s), extract their text, and
     ask the LLM to parse that listing directly into {title, authors,
     printed_page}. This is a small, easy, reliable task compared to
     scanning hundreds of body pages.
  2. The page number printed in a TOC (e.g. "72") is the magazine's own
     pagination, which usually does NOT match the PDF's actual page index
     (there's a cover, ads, and front matter before article 1 starts). So
     for each TOC entry we search the actual document text for that title
     to find its real PDF page, instead of trusting the printed number.
  3. FALLBACK: if no usable TOC is found (or parsing yields too few
     entries relative to the document's length), fall back to a
     batch-scanning approach: build a lightweight per-page digest (text
     snippet + max font size) for every page, and send the LLM batches of
     ~25 consecutive page digests at once so it has enough context to tell
     a genuine new article from a repeated running header.

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

# --- TOC detection/parsing ---
TOC_SEARCH_PAGES = 20          # only look for a TOC within the first N pages (ads can push it back)
TOC_MIN_NUMBERED_LINES = 3     # a page needs at least this many bare "N" (page-number-only) lines to count as a TOC
TOC_MAX_PAGES = 3              # a TOC can span a few consecutive pages
TOC_MIN_ENTRIES = 3            # if the parsed TOC yields fewer than this, don't trust it - fall back

# --- batch-scanning fallback ---
SNIPPET_CHARS = 400
BATCH_SIZE = 25
BATCH_OVERLAP = 3
MAX_TOKENS_RESPONSE = 1200

# How many requests to send to the local LLM concurrently.
# LM Studio's server (llama.cpp) can process multiple requests concurrently
# ONLY if you've enabled multiple parallel slots in the server settings
# (Developer tab -> server settings -> "Number of expected parallel requests"
# or similar, depending on version). If your server only has 1 slot, raising
# this won't speed anything up - the requests will just queue on the server
# side instead of in this script.
MAX_WORKERS = 2

# A page with a large image and almost no text is very likely an ad, a
# full-page photo, or a divider - not an article start. Dropped from
# consideration entirely (both TOC page-matching and the batch fallback).
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
# Shared: per-page text + layout extraction
# --------------------------------------------------------------------------

def get_page_text_ordered(page):
    """Return the page's text in visual reading order, plus the max font
    size found in the top ~60% of the page.

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


def is_ad_page(page, full_text):
    char_count = len(full_text.strip())
    if char_count == 0:
        return True
    if char_count < AD_MAX_TEXT_CHARS and image_coverage_ratio(page) > AD_IMAGE_COVERAGE_THRESHOLD:
        return True
    return False


def get_all_page_texts(doc):
    """Full extracted text for every page, 0-based indexed list."""
    return [get_page_text_ordered(page)[0] for page in doc]


# --------------------------------------------------------------------------
# LLM call helpers (shared)
# --------------------------------------------------------------------------

def _call_llm(prompt, max_tokens):
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] LLM request failed: {e}", file=sys.stderr)
        return None, None

    try:
        response_json = resp.json()
        message = response_json["choices"][0]["message"]
        content = message.get("content", "")
        finish_reason = response_json["choices"][0].get("finish_reason", "?")
    except (KeyError, IndexError, json.JSONDecodeError):
        print(f"  [!] Unexpected LLM response shape: {resp.text[:300]}", file=sys.stderr)
        return None, None

    return content, finish_reason


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
    array_text = match.group(0)
    try:
        result = json.loads(array_text)
    except json.JSONDecodeError:
        try:
            result = json.loads(_repair_json_array_text(array_text))
        except json.JSONDecodeError:
            return None
    return result if isinstance(result, list) else None


# --------------------------------------------------------------------------
# STRATEGY A: table-of-contents based extraction (primary)
# --------------------------------------------------------------------------

_STANDALONE_NUMBER_LINE = re.compile(r"^\d{1,4}$")


def _score_toc_page(text):
    """Heuristic score for how much a page looks like a table of contents.

    Real-world PDFs (both OCR'd and digital-native) very often extract each
    text run as its own line, which means a TOC entry like "Title .... 42"
    frequently comes out as separate lines - the title, the dot leader, and
    the page number each on their own line - rather than one combined line.
    So instead of matching "title...number" as a single line, we count
    lines that are JUST a bare number (1-4 digits, nothing else). A real
    TOC page tends to have one such line per listed article; an ordinary
    body page essentially never does."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    standalone_number_lines = sum(1 for l in lines if _STANDALONE_NUMBER_LINE.match(l))
    bonus = 5 if "contents" in text[:300].lower() else 0
    return standalone_number_lines + bonus, standalone_number_lines


def find_toc_pages(doc, page_texts):
    """Return the 0-based page indices that look like table-of-contents
    pages. These do NOT need to be contiguous: magazines commonly split
    the TOC into multiple sections (e.g. a main articles list and a
    separate "Reviews & Responses" list a page or two later, with an ad
    page in between) - requiring adjacency would miss the second section."""
    n = min(TOC_SEARCH_PAGES, len(doc))
    scores = []
    for i in range(n):
        score, numbered = _score_toc_page(page_texts[i])
        scores.append((i, score, numbered))

    toc_pages = [i for i, score, numbered in scores if numbered >= TOC_MIN_NUMBERED_LINES]

    if not toc_pages and DEBUG:
        _print_toc_diagnostics(page_texts, scores)

    return toc_pages


def _print_toc_diagnostics(page_texts, scores):
    """Show why TOC detection failed: the top-scoring candidates (even if
    below threshold) plus a text preview, so a human can see what's really
    on those pages and adjust the detection logic accordingly."""
    print("  [debug] No page met the TOC threshold. Top candidates by score:", file=sys.stderr)
    ranked = sorted(scores, key=lambda s: s[1], reverse=True)[:5]
    for page_idx, score, numbered in ranked:
        preview = page_texts[page_idx][:250].replace("\n", " | ")
        print(f"    page {page_idx + 1}: score={score} numbered_lines={numbered}", file=sys.stderr)
        print(f"      preview: {preview!r}", file=sys.stderr)


def _repair_json_array_text(text):
    """Fix the most common LLM JSON mistakes before parsing: trailing commas
    before a closing bracket/brace, which json.loads rejects outright."""
    return re.sub(r",(\s*[\]}])", r"\1", text)


def deterministic_parse_toc_entries(toc_pages, page_texts):
    """Split TOC page(s) into (title, printed_page, aux_text) entries using
    the reliable structural rule observed across real magazines: a title
    line is ALMOST ALWAYS immediately followed by its own page number on a
    separate line. That means entry BOUNDARIES can be found deterministically
    in code (every standalone-number line marks exactly one boundary) rather
    than asking an LLM to judge where one article ends and the next begins -
    which turned out to be unreliable (it would either split a title+subtitle
    into two fake entries, or merge two real neighboring articles into one).

    Only the small residual text between one entry's page number and the
    next entry's title (typically a subtitle line and/or an author line) is
    genuinely ambiguous, and that's left for a separate, narrower LLM call."""
    entries = []
    for page_idx in toc_pages:
        lines = [l.strip() for l in page_texts[page_idx].split("\n") if l.strip()]
        number_indices = [i for i, l in enumerate(lines) if _STANDALONE_NUMBER_LINE.match(l)]
        for k, num_idx in enumerate(number_indices):
            if num_idx == 0:
                continue  # no line before this number - can't be a real title+page pair
            title = lines[num_idx - 1]
            if len(title) < 3 or not re.search(r"[a-zA-Z]", title):
                continue  # too short/junk to plausibly be a real title
            try:
                printed_page = int(lines[num_idx])
            except ValueError:
                continue
            end_idx = number_indices[k + 1] - 1 if k + 1 < len(number_indices) else len(lines)
            aux_text = "\n".join(lines[num_idx + 1:end_idx])
            entries.append({"title": title, "printed_page": printed_page, "aux_text": aux_text})
    return entries


AUTHOR_EXTRACTION_PROMPT = """Below is a JSON array of short snippets. Each snippet is the \
one or two lines of text that appeared directly under an article's title in a magazine's table \
of contents (already separated from the title itself) - typically a subtitle/description line \
and/or an author byline, in some order, or only one of the two, or neither. Text may contain \
OCR errors.

For each snippet, identify:
  - "subtitle": the descriptive subtitle text if present, or null
  - "authors": a list of author name(s) if present, or an empty list

Respond with ONLY a valid JSON array, no other text, matching each input by its "index":
[{{"index": <integer, exactly as given>, "subtitle": "string or null", "authors": ["string", ...]}}, ...]

Snippets:
{snippets_json}
"""


def extract_subtitles_and_authors(entries):
    """One batched LLM call covering every entry's short residual snippet,
    rather than one call per entry - cheap and keeps the LLM's job narrow."""
    snippets = [{"index": i, "text": e["aux_text"]} for i, e in enumerate(entries) if e["aux_text"].strip()]
    if not snippets:
        return {}
    prompt = AUTHOR_EXTRACTION_PROMPT.format(snippets_json=json.dumps(snippets, ensure_ascii=False))
    content, finish_reason = _call_llm(prompt, max_tokens=2500)
    if content is None:
        return {}
    parsed = parse_llm_json_array(content)
    if parsed is None:
        if DEBUG:
            print(
                f"  [!] Could not parse author/subtitle JSON (finish_reason={finish_reason}). Raw reply:\n      {content[:3000]!r}",
                file=sys.stderr,
            )
        return {}
    result = {}
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            result[item["index"]] = {
                "subtitle": item.get("subtitle"),
                "authors": sanitize_authors(item.get("authors")),
            }
    return result


def _normalize_for_match(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def locate_title_page(title, page_texts, skip_pages, search_hint=None):
    """Find the 0-based page index where this title actually starts, by
    searching the document's real text rather than trusting the TOC's
    printed page number (which is usually offset from the PDF's page index
    by however many cover/ad/front-matter pages precede article 1).

    search_hint: an approximate 0-based page index to search near first
    (based on the printed page number and a running offset estimate), to
    disambiguate if the title text happens to match in more than one place."""
    norm_title = _normalize_for_match(title)
    words = norm_title.split()
    if not words:
        return None
    # Use a distinctive chunk of the title (first ~6 words) as the search key -
    # long enough to be specific, short enough to survive a subtitle being
    # abbreviated differently between the TOC and the actual article page.
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


def segment_pdf_via_toc(pdf_path, doc, page_texts, num_pages):
    toc_pages = find_toc_pages(doc, page_texts)
    if not toc_pages:
        print("  No table-of-contents page detected.")
        return None

    print(f"  Detected table of contents on page(s): {[p + 1 for p in toc_pages]}")
    toc_entries = deterministic_parse_toc_entries(toc_pages, page_texts)

    if not toc_entries:
        print("  Could not parse a usable TOC.")
        return None

    print(f"  TOC lists {len(toc_entries)} entr(y/ies)")

    info_by_index = extract_subtitles_and_authors(toc_entries)

    # Estimate the offset between "printed page number" and "PDF page index"
    # from the first entry we can confidently locate, then use it to help
    # disambiguate later matches (title text can occasionally appear more
    # than once, e.g. also referenced elsewhere).
    skip_pages = set(toc_pages)
    located = []  # list of (pdf_page_0based, title, authors)
    offset_estimate = None

    for i, entry in enumerate(toc_entries):
        title = entry["title"]
        printed_page = entry["printed_page"]
        info = info_by_index.get(i, {})
        subtitle = info.get("subtitle")
        authors = info.get("authors", [])

        display_title = f"{title}: {subtitle}" if subtitle else title

        search_hint = None
        if offset_estimate is not None:
            search_hint = printed_page - 1 + offset_estimate

        # Search using ONLY the title (not the combined title+subtitle) -
        # the actual page's headline is far more likely to literally match
        # just the title than a compound string that includes a subtitle
        # the model may have ordered or worded slightly differently.
        page_idx = locate_title_page(title, page_texts, skip_pages, search_hint)

        if page_idx is None and offset_estimate is not None:
            guess = printed_page - 1 + offset_estimate
            if 0 <= guess < num_pages:
                page_idx = guess
                print(f"  [~] \"{title}\": using estimated offset (no direct text match)")

        if page_idx is None:
            print(f"  [x] \"{title}\": could not locate in document, skipping")
            continue

        if offset_estimate is None:
            offset_estimate = page_idx - (printed_page - 1)

        located.append((page_idx, display_title, authors))
        print(f"  [+] page {page_idx + 1}: \"{display_title}\"")

    if len(located) < TOC_MIN_ENTRIES:
        print(f"  Only {len(located)} entries could be located - not enough to trust the TOC path.")
        return None

    located.sort(key=lambda t: t[0])
    # Merge entries that resolve to the exact same PDF page (can still happen
    # occasionally, e.g. two TOC lines pointing at the same page). Combine
    # rather than silently drop, so we don't lose information.
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


# --------------------------------------------------------------------------
# STRATEGY B: batch-scanning (fallback)
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


def build_page_digests(doc, page_texts):
    digests = []
    for page_num, page in enumerate(doc):
        full_text = page_texts[page_num]
        if is_ad_page(page, full_text):
            continue
        page_height = page.rect.height
        _, max_font = get_page_text_ordered(page)
        digests.append({
            "page": page_num + 1,
            "snippet": full_text[:SNIPPET_CHARS],
            "max_font": round(max_font, 1),
        })
    return digests


def make_batches(digests, batch_size, overlap):
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


def call_local_llm_batch(digest_batch):
    prompt = BATCH_PROMPT_TEMPLATE.format(
        snippet_chars=SNIPPET_CHARS,
        digest_json=json.dumps(digest_batch, ensure_ascii=False),
    )
    content, finish_reason = _call_llm(prompt, max_tokens=MAX_TOKENS_RESPONSE)
    if content is None:
        return None
    parsed = parse_llm_json_array(content)
    if parsed is None and DEBUG:
        print(
            f"  [!] Could not parse batch JSON (finish_reason={finish_reason}). Raw reply:\n      {content[:3000]!r}",
            file=sys.stderr,
        )
    return parsed


def merge_duplicates(confirmed_starts):
    if not confirmed_starts:
        return confirmed_starts
    by_page = {}
    for page_num, title, authors in confirmed_starts:
        if page_num not in by_page:
            by_page[page_num] = (title, authors)
    ordered = sorted(by_page.items())

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


def segment_pdf_via_batches(pdf_path, doc, page_texts, num_pages):
    digests = build_page_digests(doc, page_texts)
    print(f"  {len(digests)} page(s) with content out of {num_pages} total")

    batches = make_batches(digests, BATCH_SIZE, BATCH_OVERLAP)
    total = len(batches)
    print(f"  Split into {total} batch(es) of ~{BATCH_SIZE} pages")

    all_starts = []
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
                print(f"\n  [x] batch {idx + 1}: no usable response from LLM")
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
        all_starts = [(1, pdf_path.stem, [])]

    confirmed_starts = merge_duplicates(all_starts)
    for page_num, title, authors in confirmed_starts:
        print(f"  [+] page {page_num}: \"{title}\"")

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
# Shared helpers
# --------------------------------------------------------------------------

_JUNK_AUTHOR_VALUES = {"null", "none", "n/a", "na", "unknown", ""}


def sanitize_authors(raw_authors):
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


# --------------------------------------------------------------------------
# Top-level per-PDF dispatch
# --------------------------------------------------------------------------

def segment_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    page_texts = get_all_page_texts(doc)

    print("  Trying table-of-contents extraction first...")
    entries = segment_pdf_via_toc(pdf_path, doc, page_texts, num_pages)

    if entries is None:
        print("  Falling back to batch-scanning approach...")
        entries = segment_pdf_via_batches(pdf_path, doc, page_texts, num_pages)

    doc.close()
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