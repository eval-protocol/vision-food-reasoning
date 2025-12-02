import json
import os
import re
from pathlib import Path
from typing import Any

import litellm

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    Message,
    MetricResult,
    ChatCompletionContentPartTextParam,
)
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.pytest.evaluation_test import evaluation_test

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "vision_food_reasoning_sample.jsonl"

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


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, ChatCompletionContentPartTextParam):
                parts.append(part.text)
        return "\n".join(parts)
    return ""


def _normalize_label(label: str | None) -> str:
    if not label:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


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


def _extract_prediction(row: EvaluationRow) -> tuple[str, str]:
    assistant_messages = [m for m in row.messages if m.role == "assistant"]
    if not assistant_messages:
        return "", ""
    text = _content_to_text(assistant_messages[-1].content)
    label = _normalize_label(_extract_label_from_text(text))
    return label, text


def _llm_equivalence_check(
    ground_truth_label: str,
    prediction_text: str,
    *,
    judge_model: str | None = None,
) -> tuple[bool, str]:
    model_name = (
        judge_model
        or os.getenv("VISION_FOOD_REASONING_JUDGE_MODEL")
        or "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"
    )
    prediction_text = prediction_text.strip()
    if not prediction_text:
        return False, "LLM judge skipped: prediction text is empty."

    system_prompt = (
        "You are a strict food classification judge. "
        "Given the ground-truth dish label and a model response, decide whether the response "
        "unambiguously identifies the same dish. "
        "Only consider the final answer portion; ignore speculation or unrelated commentary. "
        'Respond with compact JSON like {"equivalent": true, "reason": "..."}.'
    )
    user_prompt = (
        "GROUND TRUTH LABEL: {label}\n"
        'MODEL RESPONSE:\n"""\n{response}\n"""\n\n'
        "If the model clearly identifies the same dish, set equivalent=true, otherwise false. "
        "Explain the decision in the reason."
    ).format(label=ground_truth_label, response=prediction_text)

    try:
        completion = litellm.completion(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_content = completion["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - depends on external service
        return False, f"LLM judge failed: {exc}"

    if not raw_content:
        return False, "LLM judge returned empty response."

    parsed = _parse_json_blob(raw_content)
    if not isinstance(parsed, dict):
        return False, f"LLM judge did not return JSON: {raw_content}"

    decision = parsed.get("equivalent")
    reason = parsed.get("reason") or "LLM judge provided no reason."
    if isinstance(decision, str):
        decision = decision.strip().lower() in {"true", "yes", "1"}
    elif not isinstance(decision, bool):
        decision = False
        reason = f"LLM judge missing boolean decision. Raw: {raw_content}"
    return bool(decision), reason


def _parse_json_blob(blob: str) -> Any:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        start = blob.find("{")
        end = blob.rfind("}")
        if start != -1 and end != -1 and start < end:
            snippet = blob[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def _ground_truth_label(row: EvaluationRow) -> str:
    if isinstance(row.ground_truth, dict):
        return _normalize_label(row.ground_truth.get("label"))
    if isinstance(row.ground_truth, str):
        return _normalize_label(row.ground_truth)
    return ""


@evaluation_test(
    input_dataset=[str(DATASET_PATH)],
    completion_params=[
        {
            "model": "fireworks_ai/accounts/fireworks/models/qwen3-vl-235b-a22b-instruct",
            # "max_tokens": 512,
            # "model": "openrouter/qwen/qwen3-vl-30b-a3b-instruct",
            # "model": "gpt-4.1-mini",
            "max_tokens": 512,
            # "extra_body": {
            #     'provider': {
            #         'order': ['novita'],
            #         'allow_fallbacks': False,
            #     },
            # },
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=None,
    max_dataset_rows=10,
    num_runs=1,
    mode="pointwise",
)
def test_vision_food_reasoning(row: EvaluationRow) -> EvaluationRow:
    predicted_label, raw_prediction = _extract_prediction(row)
    ground_truth_label = _ground_truth_label(row)

    is_valid = bool(predicted_label)
    is_correct = is_valid and predicted_label == ground_truth_label and bool(ground_truth_label)

    llm_equivalent = False
    llm_reason = "LLM judge not triggered."
    if not is_correct and ground_truth_label:
        llm_equivalent, llm_reason = _llm_equivalence_check(ground_truth_label, raw_prediction)
        if llm_equivalent:
            is_correct = True
            is_valid = True

    score = 1.0 if is_correct else 0.0
    reason = "Prediction matches ground truth" if is_correct else "Prediction did not match"
    if llm_equivalent:
        reason = "LLM judge considered the prediction equivalent to the ground truth."

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=reason,
        is_score_valid=is_valid,
        metrics={
            "exact_match": MetricResult(
                score=score,
                is_score_valid=is_valid,
                reason="Normalized label comparison",
                data={
                    "predicted_label": predicted_label,
                    "ground_truth_label": ground_truth_label,
                    "raw_prediction": raw_prediction,
                },
            ),
            "llm_equivalence": MetricResult(
                score=1.0 if llm_equivalent else 0.0,
                is_score_valid=llm_equivalent,
                reason=llm_reason,
            ),
        },
    )
    return row
