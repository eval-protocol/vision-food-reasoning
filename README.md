# Vision Food Reasoning Dataset

This example demonstrates how to evaluate multimodal models on the
[fireworks-ai/vision-food-reasoning-dataset](https://huggingface.co/datasets/fireworks-ai/vision-food-reasoning-dataset).
Each row contains a food photograph (embedded as a `data:image/...` URI) and the
prompt *"What's in this image?"*. The assistant response includes structured
reasoning followed by a final classification (for example: `crab_cakes`,
`mussels`, `crème brûlée`, `grilled_cheese_sandwich`, `dumplings`, or
`frozen_yogurt`).

## Repository layout

```
examples/vision_food_reasoning_dataset/
├── data/vision_food_reasoning_sample.jsonl
└── tests/test_vision_food_reasoning.py
```

- `data/vision_food_reasoning_sample.jsonl` contains the first eight rows of the
  public dataset and is small enough to commit. Use it for smoke tests or swap
  in the full JSONL if you need broader coverage. The helper below downloads a
  fresh slice:

  ```bash
  python - <<'PY'
  import urllib.request, json
  url = "https://huggingface.co/datasets/fireworks-ai/vision-food-reasoning-dataset/resolve/main/food_reasoning.jsonl"
  out = "examples/vision_food_reasoning_dataset/data/vision_food_reasoning_sample.jsonl"
  with urllib.request.urlopen(url) as resp, open(out, "w", encoding="utf-8") as fh:
      for idx, line in enumerate(resp):
          if idx == 8:
              break
          fh.write(line.decode("utf-8"))
  print("wrote", idx, "rows to", out)
  PY
  ```

- `tests/test_vision_food_reasoning.py` defines the evaluation via
  `@evaluation_test`. The adapter strips the reference assistant answer,
  preserves only the user prompt (image + question), and stores the normalized
  food label inside `row.ground_truth`.

## Running locally

The default completion params target a model that accepts image inputs
(`gpt-4.1-mini`). Provide whichever vision-capable model you use by overriding
`--ep-completion-params-json`.

```bash
export OPENAI_API_KEY=...
pytest examples/vision_food_reasoning_dataset/tests/test_vision_food_reasoning.py \
  --ep-max-rows 2 \
  --ep-completion-params-json '{"model":"gpt-4.1-mini","max_tokens":512}'
```

- Use `--ep-max-rows` to control the evaluation size (the adapter supports up to
  however many rows you pass in the JSONL).
- To use a Fireworks model, replace the completion params with the appropriate
  identifier and ensure `FIREWORKS_API_KEY` is present in the environment.

## Notes

- Only the text segments of the user/assistant messages are logged or used for
  scoring; the `image_url` entries stay in the row so they can be forwarded to
  a multimodal rollout processor.
- The evaluation function extracts the final label by looking for bold text,
  `appears to be ...` phrases, or (as a fallback) the last sentence in the
  assistant response. Both prediction and ground truth are normalized before an
  exact-match comparison.
