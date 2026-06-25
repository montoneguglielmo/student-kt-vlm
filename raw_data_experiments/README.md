# Raw Data Experiments

## Goal

Train a small Vision-Language Model (VLM) to predict the outcome of a student's next exercise attempt given the student's history of past attempts. The model receives raw multimodal input: screenshots of the exercises and text describing the exercises and the attempts — no hand-crafted features.

Two training regimes are supported and can be switched by editing `config.yaml`:

- **Probe** — the VLM backbone is frozen; only the final classification head is trained.
- **LoRA fine-tuning** — lightweight LoRA adapters are injected into the attention and MLP projections of the backbone and trained alongside the head.

The base model is [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), but any HuggingFace VLM supported by `Idefics3Processor` can be used by changing `model_id` in `config.yaml`.

---

## Code Overview

### `data_preparation/create_per_student_sequence.ipynb`
Prepares the dataset by grouping raw interaction logs into per-student chronological sequences of exercise attempts. The output is saved to the `data/` directory and consumed by the training pipeline.

### `data_processing.py`
Builds batches of multimodal data for the VLM. Implements `KTVLMDataset`, which samples a sliding history window of attempts for each student, and `make_collate_fn`, which tokenises text and stacks image tensors into model-ready batches.

### `models.py`
Defines the two model architectures:
- **Probe** — frozen VLM backbone + trainable linear classification head.
- **LoRA** — VLM backbone with LoRA adapters injected via `peft` + trainable head.

A single `build_model(cfg)` factory selects the right architecture based on `cfg.model_type`.

### `train.py`
End-to-end training script. Loads the config, instantiates the dataset and model, and runs the training loop. Select the model type by setting `model_type: probe` or `model_type: lora` in `config.yaml`.

```bash
python train.py
```

---

## Configuration (`config.yaml`)

| Key | Description |
|---|---|
| `model_id` | HuggingFace model identifier |
| `model_type` | `probe` or `lora` |
| `dataset` | Path to the prepared dataset directory |
| `screenshots_dir` | Path to the exercise screenshot files |
| `min_history` / `max_history` | Min/max number of past attempts to include in context |
| `history_window` | Number of the most recent past attempts shown with a screenshot; older attempts beyond this window are included as text only |
| `lr.probe` / `lr.lora` | Learning rates (probe trains fewer parameters so uses a higher LR) |
| `lora.r`, `lora.alpha`, `lora.dropout` | LoRA rank, scaling factor, and dropout |
| `lora.target_modules` | Which projection layers receive LoRA adapters |
