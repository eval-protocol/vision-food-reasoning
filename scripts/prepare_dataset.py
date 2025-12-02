#!/usr/bin/env python3
"""
Utility script to convert the raw fireworks vision-food-reasoning dataset into native
Eval Protocol EvaluationRow JSONL files so that the default dataset adapter can be used.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

DATASET_SOURCE_ID = "fireworks-ai/vision-food-reasoning-dataset"

_BOLD_LABEL_PATTERN = re.compile(r"\*\*(?P<label>[^*]+)\*\*")
_APPEARS_PATTERN = re.compile(r"appears to be\s+(?P<label>[A-Za-z0-9_\- ]+)", re.IGNORECASE)
_IS_PATTERN = re.compile(r"is\s+(?:a|an|the)?\s*(?P<label>[A-Za-z0-9_\- ]+)", re.IGNORECASE)
_SECTION_HEADINGS = {
    "visual characteristics",
    "texture and shape",
    "texture",
    "shape",
    "cooking method or preparation style",
    "cooking method",
    "preparation style",
    "cultural context or typical presentation",
    "cultural context",
    "presentation",
    "distinguishing features",
    "ingredients",
    "aroma",
    "flavor profile",
}


def _normalize_label(label: str | None) -> str:
    if not label:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_val = part.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(parts)
    return ""


def _extract_label_from_text(text: str) -> str | None:
    if not text:
        return None
    bold_matches = _BOLD_LABEL_PATTERN.findall(text)
    if bold_matches:
        for candidate in reversed(bold_matches):
            normalized = candidate.strip().lower()
            if normalized not in _SECTION_HEADINGS:
                return candidate.strip()
    for pattern in (_APPEARS_PATTERN, _IS_PATTERN):
        match = pattern.search(text)
        if match:
            label = match.group("label").strip()
            if len(label.split()) <= 5:
                return label
    sentences = [segment.strip() for segment in re.split(r"[.!?\n]+", text) if segment.strip()]
    if sentences:
        tail = sentences[-1]
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\- ]+", tail)
        if tokens:
            return tokens[-1].strip()
    return None


def convert_dataset(input_path: Path, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    with input_path.open() as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    converted_rows: list[dict[str, Any]] = []
    skipped = 0

    for idx, raw in enumerate(rows):
        messages_payload = raw.get("messages")
        if not isinstance(messages_payload, list) or len(messages_payload) < 2:
            skipped += 1
            continue

        assistant_reference = messages_payload[-1]
        prompt_messages = [
            message
            for message in messages_payload[:-1]
            if isinstance(message, dict) and message.get("role") in {"system", "user"}
        ]
        if not prompt_messages:
            skipped += 1
            continue

        reference_text = _content_to_text(assistant_reference.get("content"))
        raw_label = _extract_label_from_text(reference_text)
        normalized_label = _normalize_label(raw_label)
        if not normalized_label:
            skipped += 1
            continue

        row_id = str(raw.get("id") or f"vision_food_reasoning_{idx}")
        converted_rows.append(
            {
                "messages": prompt_messages,
                "ground_truth": {
                    "label": normalized_label,
                    "raw_label": raw_label or "",
                    "reference_answer": reference_text,
                },
                "input_metadata": {
                    "row_id": row_id,
                    "dataset_info": {
                        "source": DATASET_SOURCE_ID,
                        "normalized_label": normalized_label,
                    },
                },
            }
        )

    with output_path.open("w") as outfile:
        for row in converted_rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted_rows)} rows (skipped {skipped}) from {input_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw vision food reasoning dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the raw JSONL dataset.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL path for the converted EvaluationRow dataset.",
    )
    args = parser.parse_args()
    convert_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
