"""
Frozen per-exercise encoder for a (small) VLM.

Goal
----
Turn each exercise (text description + screenshot) into ONE fixed-size vector.
These vectors are cached to a Polars/Parquet table keyed by exercise_id (same
shape as the exercises table: one row per exercise), and later consumed as
input features by a sequence Knowledge-Tracing model:

    KT model input  = [emb(ex_0), emb(ex_1), ..., emb(ex_{t-1}), emb(ex_t)]
                      (+ per-attempt scalars: correct/incorrect, duration, ...)
    KT model output = P(student succeeds on ex_t)

Why a frozen encoder (vs the LoRA fine-tune in the KT prompt design)?
  * Each exercise is encoded ONCE, not once-per-attempt -> far cheaper.
  * There are ~hundreds of exercises but ~millions of attempts.
  * The backbone gets no gradients, so embeddings are deterministic and cacheable.
  * The KT model trains on top of frozen features -> fast iteration.

Trade-off: SmolVLM-256M-Instruct is a *generative* model, not trained for
similarity. Its pooled hidden states are fine as learned-on-top features
(your case), less good for off-the-shelf semantic search.

Prototyping target: HuggingFaceTB/SmolVLM-256M-Instruct (Idefics3 processor).
Switching to Qwen2.5-VL later changes only the model/processor IDs.
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Helpers for building the exercise description text                          #
# --------------------------------------------------------------------------- #
def _strip_html(text):
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _format_exercise(meta):
    """Text description of a single exercise (no student/attempt info)."""
    parts = [
        f"Exercise id: {meta['exercise_id']}",
        f"Gameplay type: {meta['gameplay_type']}",
        f"Module: {meta['module_name']}",
        f"Objective: {meta['objective_name']}",
        f"Objective goal: {_strip_html(meta.get('objective_pedagogical_intent'))}",
        f"Activity: {meta['activity_name']}",
        f"Activity goal: {_strip_html(meta.get('activity_pedagogical_intent'))}",
    ]

    content = meta.get("content")
    try:
        c = json.loads(content)
        if c.get("instruction"):
            parts.append(f"Instruction: {c['instruction']}")
        if c.get("question"):
            parts.append(f"Question: {c['question']}")
        if c.get("correct_answer"):
            parts.append(f"Correct answer: {c['correct_answer']}")
        if c.get("exercise_type"):
            parts.append(f"Exercise type: {c['exercise_type']}")
    except (json.JSONDecodeError, TypeError):
        parts.append("Content: not available")

    parts.append(f"Source: {meta['source']}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Dataset: one item == one exercise (text + optional screenshot)              #
# --------------------------------------------------------------------------- #
class ExerciseDataset(Dataset):
    def __init__(self, exercises_dataframe, screenshots_dir):
        self.screenshots_dir = Path(screenshots_dir)

        ex = exercises_dataframe
        self.rows = list(ex.iter_rows(named=True))

        # exercise_id -> screenshot path (searches nested source dirs)
        self._screenshot_cache = self._index_screenshots()

    def _index_screenshots(self):
        cache = {}
        for source_dir in self.screenshots_dir.iterdir():
            if not source_dir.is_dir():
                continue
            for f in source_dir.iterdir():
                cache[f.stem] = f
        return cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        meta = self.rows[idx]
        ex_id = meta["exercise_id"]

        path = self._screenshot_cache.get(ex_id)
        image = Image.open(path).convert("RGB") if path is not None else None

        return {
            "exercise_id": ex_id,
            "text": _format_exercise(meta),
            "image": image,          # PIL.Image or None
            "has_image": image is not None,
        }


# --------------------------------------------------------------------------- #
# Collate: build chat messages, run processor on the whole batch              #
# --------------------------------------------------------------------------- #
def make_collate_fn(processor):
    def _build_messages(sample):
        content = []
        if sample["has_image"]:
            content.append({"type": "image"})
        content.append({"type": "text", "text": sample["text"]})
        return [{"role": "user", "content": content}]

    def collate_fn(batch):
        messages = [_build_messages(b) for b in batch]
        prompts = [
            processor.apply_chat_template(m, add_generation_prompt=False)
            for m in messages
        ]
        # images must be list[list[Image]]; omit entirely if no sample has one.
        images = [[b["image"]] for b in batch if b["has_image"]]
        kwargs = dict(text=prompts, return_tensors="pt", padding=True)
        if images:
            kwargs["images"] = [[b["image"]] for b in batch]
        inputs = processor(**kwargs)

        inputs["_exercise_ids"] = [b["exercise_id"] for b in batch]
        return inputs

    return collate_fn


# --------------------------------------------------------------------------- #
# Pooling                                                                     #
# --------------------------------------------------------------------------- #
def pool_hidden_states(last_hidden, attention_mask, mode="mean"):
    """
    last_hidden:    (B, T, H) final-layer hidden states
    attention_mask: (B, T)    1 for real tokens, 0 for padding
    returns:        (B, H)
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)

    if mode == "mean":
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    if mode == "last":
        # index of the last non-pad token per row (handles left/right padding)
        lengths = attention_mask.sum(dim=1) - 1            # (B,)
        idx = lengths.clamp(min=0).long()
        return last_hidden[torch.arange(last_hidden.size(0)), idx]

    raise ValueError(f"unknown pooling mode: {mode}")


# --------------------------------------------------------------------------- #
# Encode all exercises -> {exercise_id: vector}, saved to disk                #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_all(
    exercises_dataframe,
    screenshots_dir,
    model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    pooling="last",                   # "mean" or "last"
    batch_size=8,
    num_workers=2,
    device=None,
    dtype=torch.bfloat16,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device).eval()

    ds = ExerciseDataset(exercises_dataframe, screenshots_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(processor),
        num_workers=num_workers,
    )

    ids, vecs = [], []
    for inputs in tqdm(loader, desc="Encoding exercises", unit="batch"):
        ex_ids = inputs.pop("_exercise_ids")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]                 # (B, T, H)
        pooled = pool_hidden_states(
            last_hidden, inputs["attention_mask"], mode=pooling
        )

        ids.extend(ex_ids)
        vecs.append(pooled.float().cpu().numpy())

    matrix = np.concatenate(vecs, axis=0)                    # (N, H)
    dim = matrix.shape[1]

    # One row per exercise; embedding stored as a fixed-length list column.
    df = pl.DataFrame(
        {
            "exercise_id": ids,
            "embedding": matrix.tolist(),
        },
        schema={
            "exercise_id": pl.Utf8,
            "embedding": pl.Array(pl.Float32, dim),
        },
    )
    return df


