"""
Training script — works with either the linear-probe or the LoRA model.
Select which one via MODEL_TYPE in config.py.

Run:  python train.py
"""

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import Idefics3Processor
from tqdm import tqdm

from data_processing import KTVLMDataset, make_collate_fn
from models import build_model

from types import SimpleNamespace
import yaml
from pathlib import Path
   

def load_config(path="config.yaml"):
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Resolve runtime-only values that don't belong in YAML.
    raw["dataset"] = Path(raw["dataset"])
    raw["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Nested dicts -> namespaces so you can write cfg.lora.r
    raw["lora"] = SimpleNamespace(**raw["lora"])
    raw["lr"] = SimpleNamespace(**raw["lr"])

    return SimpleNamespace(**raw)



def main():
    
    cfg = load_config()
    
    processor = Idefics3Processor.from_pretrained(cfg.model_id)

    train_dataset = KTVLMDataset(
        sequences_path=cfg.dataset / "train_sequences.parquet",
        exercises_path=cfg.dataset / "exercises.parquet",
        screenshots_dir=cfg.screenshots_dir,
        min_history=cfg.min_history,
        max_history=cfg.max_history,
        history_window=cfg.history_window,
    )
    val_dataset = KTVLMDataset(
        sequences_path=cfg.dataset / "val_sequences.parquet",
        exercises_path=cfg.dataset / "exercises.parquet",
        screenshots_dir=cfg.screenshots_dir,
        min_history=cfg.min_history,
        max_history=cfg.max_history,
        history_window=cfg.history_window,
    )
    print(f"{len(train_dataset)} training examples")
    print(f"{len(val_dataset)} validation examples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(processor),
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(processor),
        num_workers=2,
    )

    model = build_model(cfg).to(cfg.device)
    
    # Optimize everything trainable: head only (probe) or adapters + head (LoRA).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=getattr(cfg.lr, cfg.model_type))
    loss_fn = nn.BCEWithLogitsLoss()

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        total_steps = min(cfg.max_steps, len(train_loader)) if cfg.max_steps else len(train_loader)
        pbar = tqdm(train_loader, total=total_steps, desc=f"Epoch {epoch}", unit="batch")
        for batch in pbar:
            labels = batch.pop("labels").to(cfg.device, dtype=torch.bfloat16)
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            logits = model(**batch)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix(step=step, loss=f"{loss.item():.4f}")

            if cfg.max_steps and step >= cfg.max_steps:
                pbar.close()
                print("reached MAX_STEPS, stopping")
                break

        # Validation pass at end of epoch
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} val", unit="batch"):
                labels = batch.pop("labels").to(cfg.device, dtype=torch.bfloat16)
                batch = {k: v.to(cfg.device) for k, v in batch.items()}

                logits = model(**batch)
                val_loss += loss_fn(logits, labels).item() * labels.size(0)
                preds = (torch.sigmoid(logits) > 0.5).to(torch.bfloat16)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"epoch {epoch}  val loss {val_loss:.4f}  val acc {val_acc:.4f}")

        if cfg.max_steps and step >= cfg.max_steps:
            return


if __name__ == "__main__":
    main()