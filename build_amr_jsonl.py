#!/usr/bin/env python3
"""
Convert segmented Easy Read data into JSONL records with normal/easy text and AMRs.

Input expectations
------------------
- A plain text file containing lines such as:
    It is written _seg_ in all European languages.
    Heather Gilchrist is a self-advocate _seg_ from ENABLE Scotland.

Processing rules
----------------
1. Split text into sentences using periods only.
2. For each sentence:
   - normal_text: remove `_seg_` markers and normalize whitespace.
   - normal_amr: generate AMR for the normal_text sentence.
   - easy_text: split on `_seg_` and keep each segment as its own entry.
   - easy_amr: generate one AMR per easy_text segment.
3. Write one JSON object per sentence to a JSONL file.

Notes
-----
- This script uses amrlib for sentence-to-graph AMR parsing.
- The normal_amr and easy_amr fields are lists of strings to match your schema.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

SEG_TOKEN = "_seg_"
WHITESPACE_RE = re.compile(r"\s+")


def normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace and strip ends."""
    return WHITESPACE_RE.sub(" ", text).strip()


def split_by_period_only(text: str) -> List[str]:
    """
    Split text into sentences using periods only.

    Examples:
        "A. B." -> ["A.", "B."]
        "A"     -> ["A"]
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = text.split(".")
    sentences: List[str] = []

    for i, chunk in enumerate(chunks):
        cleaned = normalize_spaces(chunk)
        if not cleaned:
            continue

        if i < len(chunks) - 1:
            cleaned = f"{cleaned}."
        sentences.append(cleaned)

    return sentences


def make_normal_text(sentence: str) -> str:
    """Remove segment markers and clean spacing."""
    text = sentence.replace(SEG_TOKEN, " ")
    text = normalize_spaces(text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def make_easy_segments(sentence: str) -> List[str]:
    """Split a sentence on _seg_ and return non-empty cleaned segments."""
    parts = sentence.split(SEG_TOKEN)
    segments = []
    for part in parts:
        segment = normalize_spaces(part)
        if segment:
            segment = re.sub(r"\s+([.,!?;:])", r"\1", segment)
            segments.append(segment)
    return segments


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
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


def build_records(sentences: List[str], stog_model, id_prefix: str, batch_size: int) -> List[dict]:
    """Build JSON-serializable records for each sentence."""
    normal_texts = [make_normal_text(s) for s in sentences]
    easy_text_lists = [make_easy_segments(s) for s in sentences]

    normal_amrs = parse_amrs(normal_texts, stog_model, batch_size=batch_size)

    flat_easy_texts = [seg for segs in easy_text_lists for seg in segs]
    flat_easy_amrs = parse_amrs(flat_easy_texts, stog_model, batch_size=batch_size)

    records: List[dict] = []
    easy_idx = 0
    for idx, (normal_text, easy_texts, normal_amr) in enumerate(
        zip(normal_texts, easy_text_lists, normal_amrs), start=1
    ):
        n_easy = len(easy_texts)
        easy_amrs = flat_easy_amrs[easy_idx : easy_idx + n_easy]
        easy_idx += n_easy

        record = {
            "id": f"{id_prefix}-{idx:04d}",
            "easy_text": easy_texts,
            "normal_text": normal_text if normal_text else None,
            "easy_amr": easy_amrs,
            "normal_amr": [normal_amr] if normal_amr else [],
        }
        records.append(record)

    return records


def load_input_sentences(input_path: Path) -> List[str]:
    raw_text = input_path.read_text(encoding="utf-8")
    return split_by_period_only(raw_text)


def write_jsonl(records: List[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL with Easy Read and normal AMRs.")
    parser.add_argument("input", type=Path, help="Path to input text file with _seg_ markers.")
    parser.add_argument("output", type=Path, help="Path to output JSONL file.")
    parser.add_argument(
        "--id-prefix",
        default="",
        help="Prefix used when generating IDs (default: nothing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="AMR parsing batch size (default: 16).",
    )
    args = parser.parse_args()

    # Imported lazily so users can still read --help without the dependency installed.
    import amrlib

    sentences = load_input_sentences(args.input)
    if not sentences:
        write_jsonl([], args.output)
        return

    stog_model = amrlib.load_stog_model()
    records = build_records(sentences, stog_model, args.id_prefix, args.batch_size)
    write_jsonl(records, args.output)


if __name__ == "__main__":
    main()
