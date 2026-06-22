"""
Knowledge-tracing dataset for a (small) VLM.

One training example:
    input  = textual history of student attempts 0..t-1
             + description + screenshot of the target exercise t
    target = whether the student answered exercise t correctly (0/1)

Design notes
------------
* __getitem__ returns RAW, lightweight data (a PIL image + text strings + label).
  Tokenisation / image processing happens in collate_fn via the HF processor,
  so padding is done per-batch (cheaper) and the Dataset stays model-agnostic.
* We DO NOT precompute embeddings: a LoRA fine-tune changes the backbone's
  forward pass, so frozen embeddings would be wrong. Raw pixel_values + input_ids
  must be produced every step.
* Prototyping target: HuggingFaceTB/SmolVLM-256M-Instruct (Idefics3 processor).
  Switching to Qwen2.5-VL later only changes the model/processor IDs and the
  chat-template call below, not the Dataset logic.
"""

import json
import random
import re
from pathlib import Path

import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Helpers for building the prompt text                                        #
# --------------------------------------------------------------------------- #
def _strip_html(text):
    return re.sub(r"<[^>]+>", "", text).strip()


def _format_exercise(meta, outcome=None, include_answer=True, attempt=None):
    """Text description of a single exercise.

    `outcome`: "correct"/"incorrect" for history items, None for the target.
    `include_answer`: set False to omit correct_answer (avoid label leakage).
    `attempt`: the raw attempt struct, for per-attempt fields (history only).
    """
    parts = []
    parts.append(f"Exercise id: {meta['exercise_id']}")
    parts.append(f"Gameplay type: {meta['gameplay_type']}")
    parts.append(f"Module: {meta['module_name']}")
    parts.append(f"Objective: {meta['objective_name']}")
    parts.append(f"Objective goal: {_strip_html(meta['objective_pedagogical_intent'])}")
    parts.append(f"Activity: {meta['activity_name']}")
    parts.append(f"Activity goal: {_strip_html(meta['activity_pedagogical_intent'])}")

    content = meta.get("content")
    try:
        c = json.loads(content)
        if c.get('instruction'):
            parts.append(f"Instruction: {c['instruction']}")
        if c.get('question'):
            parts.append(f"Question: {c['question']}")
        if c.get('correct_answer'):
            parts.append(f"Correct answer: {c['correct_answer']}")
        if c.get('exercise_type'):
            parts.append(f"Exercise type: {c['exercise_type']}")
    except (json.JSONDecodeError, TypeError):
        parts.append("Content: not available")
    
    parts.append(f"Source: {meta['source']}")

    if attempt is not None:
        parts.append(f"Work mode: {attempt['work_mode']}")
        parts.append(f"Answer given: {attempt['data_answer']}")
        parts.append(f"Timestamp: {attempt['created_at']}")
        parts.append(f"Duration: {attempt['data_duration']} ms")

    if outcome is not None:
        parts.append(f"Outcome: {outcome}")

    return "\n".join(parts)


def _format_history(history_attempts, ex_meta, max_history=40):
    """Turn a list of past attempt-structs into a compact text summary.

    Most recent `max_history` attempts only, oldest-first, to bound prompt length.
    """
    history_attempts = history_attempts[-max_history:]
    blocks = []
    for a in history_attempts:
        meta = ex_meta.get(a["exercise_id"], {})
        outcome = "correct" if a["data_correct"] == 1 else "incorrect"
        blocks.append(_format_exercise(meta, outcome=outcome, attempt=a))
    return "\n\n".join(blocks)


def _format_target_description(meta):
    """Text description of the exercise to be predicted (no outcome, answer hidden)."""
    return _format_exercise(meta, outcome=None, include_answer=False)


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
class KTVLMDataset(Dataset):
    def __init__(
        self,
        sequences_path,      # parquet: per-student grouped attempts (list-of-structs)
        exercises_path,      # parquet: static exercise metadata
        screenshots_dir,     # DATASET / "data/screenshots/compressed"
        min_history=3,       # need at least this many past attempts to form an example
        max_history=40,      # cap on attempts summarised in the prompt
        history_window=6,    # cap on the attempts with screenshot
        seed=0,
    ):
        self.screenshots_dir = Path(screenshots_dir)
        self.min_history = min_history
        self.max_history = max_history
        self.history_window = history_window

        self._rng = random.Random(seed)

        # --- per-student sequences ---
        seq = pl.read_parquet(sequences_path)
        self.users = seq["user_id_int"].to_list()
        self.attempts = seq["attempts"].to_list()   # list[list[struct-as-dict]]

        # --- static exercise metadata, keyed by exercise_id ---
        ex = pl.read_parquet(exercises_path)
        self.ex_meta = {r["exercise_id"]: r for r in ex.iter_rows(named=True)}

        # --- build a flat index of valid (student_idx, split_point t) examples ---
        # An example needs >= min_history past attempts AND a screenshot for the
        # target exercise t.
        self._screenshot_cache = self._index_screenshots()
        self.index = []
        self.index = []
        for s_idx, student_attempts in enumerate(self.attempts):
            for t in range(self.min_history, len(student_attempts)):
                target = student_attempts[t]
                windowed = student_attempts[:t][-self.history_window:]
                # every image this example will load: windowed history + target
                needed = [a["exercise_id"] for a in windowed]
                needed.append(target["exercise_id"])
                if all(ex in self._screenshot_cache for ex in needed):
                    self.index.append((s_idx, t))


    def _index_screenshots(self):
        """Map exercise_id -> screenshot path (searches the nested source dirs)."""
        cache = {}
        for source_dir in self.screenshots_dir.iterdir():
            for f in source_dir.iterdir():
                cache[f.stem] = f
        return cache

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_idx, t = self.index[idx]
        attempts = self.attempts[s_idx]

        history = attempts[:t]
        target = attempts[t]

        # Recent window of PAST attempts that will contribute a screenshot.
        windowed = history[-self.history_window:]

        # Load images: recent-history images first (oldest->newest), target last.
        images = [
            Image.open(self._screenshot_cache[a["exercise_id"]]).convert("RGB")
            for a in windowed
        ]
        images.append(
            Image.open(self._screenshot_cache[target["exercise_id"]]).convert("RGB")
        )

        # Text: full history summary (text-only, as before) + target description.
        history_text = _format_history(history, self.ex_meta, self.max_history)
        target_meta = self.ex_meta.get(target["exercise_id"], {})
        target_text = _format_target_description(target_meta)

        # Per-image captions for the windowed history, so the model can align
        # each image with which exercise it was and whether it was answered right.
        windowed_meta = []
        for a in windowed:
            meta = self.ex_meta.get(a["exercise_id"], {})
            name = a["exercise_id"]
            outcome = "correct" if a["data_correct"] == 1 else "incorrect"
            windowed_meta.append(f"{name} ({outcome})")

        label = float(target["data_correct"])

        return {
            "images": images,                 # list, length = len(windowed)+1
            "n_history_images": len(windowed), # so collate/messages know the split
            "windowed_meta": windowed_meta,    # captions for the history images
            "history_text": history_text,
            "target_text": target_text,
            "label": label,
        }


# --------------------------------------------------------------------------- #
# Collate: build chat messages, run the processor on the whole batch           #
# --------------------------------------------------------------------------- #
def make_collate_fn(processor):
    """Return a collate_fn bound to a given HF processor (e.g. SmolVLM's)."""

    def _build_messages(sample):
        content = []

        # Recent-history images, each tagged with what it was + outcome.
        content.append({"type": "text", "text": "Recent exercises the student attempted:"})
        for caption in sample["windowed_meta"]:
            content.append({"type": "image"})
            content.append({"type": "text", "text": caption})

        # Older history as text only.
        content.append({"type": "text", "text":
            f"\nEarlier history (oldest to newest):\n{sample['history_text']}"})

        # Target exercise: its image + description + the question.
        content.append({"type": "text", "text": "\nNext exercise to predict:"})
        content.append({"type": "image"})
        content.append({"type": "text", "text":
            f"{sample['target_text']}\n\n"
            "Will the student answer this next exercise correctly? Answer Yes or No."})

        return [{"role": "user", "content": content}]

    def collate_fn(batch):
        messages = [_build_messages(b) for b in batch]
        prompts = [processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
        # images: one list per sample, already oldest-history -> target order
        images = [b["images"] for b in batch]

        inputs = processor(
            text=prompts,
            images=images,            # list[list[Image]] — variable length per sample
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        return inputs

    return collate_fn


# --------------------------------------------------------------------------- #
# Usage sketch                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader

    DATASET = Path("/path/to/your/dataset")  # <-- set this

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

    ds = KTVLMDataset(
        sequences_path=DATASET / "sequences.parquet",
        exercises_path=DATASET / "exercises.parquet",
        screenshots_dir=DATASET / "data/screenshots/compressed",
        min_history=3,
        max_history=40,
    )
    print(f"{len(ds)} training examples")

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        collate_fn=make_collate_fn(processor),
        num_workers=2,
    )

    batch = next(iter(loader))
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in batch.items()})