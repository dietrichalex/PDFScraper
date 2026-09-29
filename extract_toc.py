#!/usr/bin/env python3
"""
extract_toc.py

Simple, standalone tool: run Docling on a PDF and dump every table it finds
in the first N pages (where a table of contents usually lives). No fallback
logic, no LLM calls, no heuristics guessing at what a column "means" - just
Docling's own layout/table detection, printed as-is and saved to CSV so you
can see exactly what it extracted and decide what to do with it.

Requires:
    pip install docling pandas

Usage:
    python extract_toc.py path/to/file.pdf
"""

import sys
from pathlib import Path

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

# Only look at tables within the first N pages (where a TOC usually lives).
# Set to None to dump every table in the whole document.
SEARCH_PAGES = 20

INPUT_FILE = Path(__file__).parent / "mini_data/FAJulyAug2000.pdf"
OUTPUT_DIR = Path(__file__).parent / "output"


def build_converter():
    pipeline_options = PdfPipelineOptions()

    # All input PDFs already have an embedded text layer (digital-native, or
    # pre-OCR'd before we ever see them), so we skip Docling's own OCR pass -
    # it's redundant and much slower than just reading existing text.
    pipeline_options.do_ocr = False

    # Use the GPU if available. Falls back to CPU automatically if no CUDA
    # device is present - AcceleratorDevice.AUTO would also do this, but
    # CUDA is explicit about intent per your request.
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def main():
    pdf_path = INPUT_FILE
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    print(f"Converting {pdf_path.name} with Docling (GPU if available)...")
    converter = build_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    print(f"Document has {doc.num_pages()} page(s) and {len(doc.tables)} table(s) total.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    found_any = False

    for i, table in enumerate(doc.tables):
        page_no = table.prov[0].page_no if table.prov else None
        if SEARCH_PAGES is not None and page_no is not None and page_no > SEARCH_PAGES:
            continue

        found_any = True
        df = table.export_to_dataframe(doc)

        print(f"\n--- Table {i} (page {page_no}) - {df.shape[0]} rows x {df.shape[1]} cols ---")
        print(df.to_string(index=False))

        out_path = OUTPUT_DIR / f"{pdf_path.stem}_table_{i}_page{page_no}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

    if not found_any:
        print(f"\nNo tables found in the first {SEARCH_PAGES} pages.")
        print("Dumping plain extracted text for those pages instead, so you can inspect the raw structure:")
        for page_no in range(1, min(SEARCH_PAGES, doc.num_pages()) + 1):
            text = doc.export_to_text(page_no=page_no)
            print(f"\n--- Page {page_no} text ---")
            print(text[:1000])


if __name__ == "__main__":
    main()