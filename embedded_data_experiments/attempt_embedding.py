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
from exercise_embedding import pool_hidden_states



def _format_attempt(attempt):
    parts = []
    parts.append(f"Work mode: {attempt['work_mode']}")
    parts.append(f"Answer given: {attempt['data_answer']}")
    parts.append(f"Timestamp: {attempt['created_at']}")
    parts.append(f"Duration: {attempt['data_duration']} ms")
    parts.append(f"Outcome: {attempt['data_correct']}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Dataset: one item == one interaction (text only, no screenshot)             #
# --------------------------------------------------------------------------- #
class InteractionDataset(Dataset):
    def __init__(self, interactions_dataframe):
        # row index becomes the interaction_id
        self.rows = list(
            interactions_dataframe.with_row_index("interaction_id").iter_rows(named=True)
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "interaction_id": row["interaction_id"],
            "text": _format_attempt(row),
        }


def make_interaction_collate_fn(processor):
    def collate_fn(batch):
        messages = [
            [{"role": "user", "content": [{"type": "text", "text": b["text"]}]}]
            for b in batch
        ]
        prompts = [
            processor.apply_chat_template(m, add_generation_prompt=False)
            for m in messages
        ]
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs["_interaction_ids"] = [b["interaction_id"] for b in batch]
        return inputs

    return collate_fn


# --------------------------------------------------------------------------- #
# Encode all interactions -> DataFrame(interaction_id, embedding)             #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_interactions(
    interactions_dataframe,
    model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    pooling="last",
    batch_size=16,
    num_workers=2,
    device=None,
    dtype=torch.bfloat16,
):
    """
    Encode each row of interactions_dataframe into a fixed-size vector using
    _format_attempt for the text representation (no images).

    Returns a Polars DataFrame with columns:
        interaction_id  UInt32  — row index of the input dataframe
        embedding       Array[Float32, H]
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device).eval()

    ds = InteractionDataset(interactions_dataframe)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_interaction_collate_fn(processor),
        num_workers=num_workers,
    )

    ids, vecs = [], []
    for inputs in tqdm(loader, desc="Encoding interactions", unit="batch"):
        interaction_ids = inputs.pop("_interaction_ids")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]                  # (B, T, H)
        pooled = pool_hidden_states(
            last_hidden, inputs["attention_mask"], mode=pooling
        )

        ids.extend(interaction_ids)
        vecs.append(pooled.float().cpu().numpy())
        torch.cuda.empty_cache()

    matrix = np.concatenate(vecs, axis=0)                    # (N, H)
    dim = matrix.shape[1]

    return pl.DataFrame(
        {
            "interaction_id": ids,
            "embedding": matrix.tolist(),
        },
        schema={
            "interaction_id": pl.UInt32,
            "embedding": pl.Array(pl.Float32, dim),
        },
    )