"""
Training script for the KT Transformer.

Run:  python train.py
"""

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import polars as pl
from tqdm import tqdm

from data_processing import KTDataset, kt_collate_fn
from models import KTTransformer

from types import SimpleNamespace
import yaml
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return SimpleNamespace(**raw)


def main():
    cfg = load_config()

    train_df = pl.read_parquet("data/train_interactions_embedded.parquet")
    val_df = pl.read_parquet("data/val_interactions_embedded.parquet")
    exercises_df = pl.read_parquet("data/exercises_embedded.parquet")

    train_dataset = KTDataset(train_df, exercises_df, max_exercises = cfg.max_exercises)
    val_dataset   = KTDataset(val_df,   exercises_df, max_exercises = cfg.max_exercises)
    print(f"{len(train_dataset)} training examples")
    print(f"{len(val_dataset)} validation examples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=kt_collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=kt_collate_fn,
        num_workers=2,
    )

    model = KTTransformer(
        d_input=cfg.d_input,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))
    loss_fn = nn.BCEWithLogitsLoss()

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        total_steps = min(cfg.max_steps, len(train_loader)) if cfg.max_steps else len(train_loader)
        pbar = tqdm(train_loader, total=total_steps, desc=f"Epoch {epoch}", unit="batch")

        for histories, targets, mask in pbar:
            histories = histories.to(cfg.device)
            targets = targets.to(cfg.device)
            mask = mask.to(cfg.device)
            
            logits = model(histories, attention_mask=mask)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix(step=step, loss=f"{loss.item():.4f}")
            

            if cfg.max_steps and step >= cfg.max_steps:
                pbar.close()
                print("reached MAX_STEPS, stopping")
                break

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for histories, targets, mask in tqdm(val_loader, desc=f"Epoch {epoch} val", unit="batch"):
                histories = histories.to(cfg.device)
                targets = targets.to(cfg.device)
                mask = mask.to(cfg.device)

                logits = model(histories, attention_mask=mask)
                val_loss += loss_fn(logits, targets).item() * targets.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"epoch {epoch}  val loss {val_loss:.4f}  val acc {val_acc:.4f}")

        if cfg.max_steps and step >= cfg.max_steps:
            return


if __name__ == "__main__":
    main()