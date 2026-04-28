#!/usr/bin/env python3
"""
Generate AMR graphs for the processed Easy Read dataset.

Reads the post-processed JSONL from the data processing pipeline and adds
AMR representations for both the normal text and the rewritten easy-read
sentences.

Input
-----
A JSONL file where each line is a JSON object with at least:
    - "id": str
    - "normal_text": str
    - "easy_text": list[str]
    - "easy_read_rewritten": list[str]

Output
------
A JSONL file with the same fields plus:
    - "normal_amr": str   (AMR graph for the normal text)
    - "easy_amr": list[str] (one AMR graph per easy_read_rewritten sentence)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield successive batches from *items*."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parse_amrs(sentences: List[str], stog_model, batch_size: int = 16) -> List[str]:
    """Run AMR parsing in batches while preserving order."""
    if not sentences:
        return []

    outputs: List[str] = []
    for batch in batched(sentences, batch_size):
        graphs = stog_model.parse_sents(batch)
        outputs.extend(graphs)
    return outputs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_records(input_path: Path) -> List[dict]:
    """Read every line of a JSONL file into a list of dicts."""
    records: List[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed JSON on line {line_no}: {exc}",
                      file=sys.stderr)
    return records


def write_jsonl(records: List[dict], output_path: Path) -> None:
    """Write a list of dicts as a JSONL file."""
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def enrich_records(
    records: List[dict],
    stog_model,
    batch_size: int = 16,
) -> List[dict]:
    """Add *normal_amr* and *easy_amr* fields to every record."""

    # --- 1. Collect all sentences that need parsing -------------------------
    normal_texts: List[str] = []
    easy_sentences: List[str] = []
    easy_counts: List[int] = []          # how many easy sents per record

    for rec in records:
        normal_texts.append(rec.get("normal_text", "") or "")

        rewritten = rec.get("easy_read_rewritten", []) or []
        easy_sentences.extend(rewritten)
        easy_counts.append(len(rewritten))

    # --- 2. Parse everything in bulk ----------------------------------------
    print(f"Parsing {len(normal_texts)} normal-text sentences …")
    normal_amrs = parse_amrs(normal_texts, stog_model, batch_size=batch_size)

    print(f"Parsing {len(easy_sentences)} easy-read sentences …")
    easy_amrs = parse_amrs(easy_sentences, stog_model, batch_size=batch_size)

    # --- 3. Distribute results back into records ----------------------------
    easy_idx = 0
    for i, rec in enumerate(records):
        rec["normal_amr"] = normal_amrs[i] if normal_amrs[i] else None

        n = easy_counts[i]
        rec["easy_amr"] = easy_amrs[easy_idx : easy_idx + n]
        easy_idx += n

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add AMR graphs to a post-processed Easy Read JSONL file.",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data_processing/post_processed_segmentation.jsonl"),
        help="Path to input JSONL (default: data_processing/post_processed_segmentation.jsonl)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("data_processing/final_data_with_amrs.jsonl"),
        help="Path to output JSONL (default: data_processing/final_data_with_amrs.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="AMR parsing batch size (default: 16).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/model_parse_xfm_bart_base-v0_1_0"),
        help="Path to the amrlib stog model directory (default: models/model_parse_xfm_bart_base-v0_1_0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N records (useful for testing).",
    )
    args = parser.parse_args()

    # Lazy import so --help works without the heavy dependency.
    import amrlib

    print(f"Loading records from {args.input} …")
    records = load_records(args.input)

    if args.limit:
        records = records[: args.limit]
        print(f"(limited to first {args.limit} records)")

    if not records:
        print("No records found – writing empty output.")
        write_jsonl([], args.output)
        return

    print(f"Loading AMR model from {args.model_dir} …")
    stog_model = amrlib.load_stog_model(model_dir=str(args.model_dir))

    enriched = enrich_records(records, stog_model, batch_size=args.batch_size)

    write_jsonl(enriched, args.output)
    print(f"Done – wrote {len(enriched)} records to {args.output}")


if __name__ == "__main__":
    main()
