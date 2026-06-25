# Embedded Data Experiments

## Goal

Train a small Transformer to predict whether a student will succeed on their next exercise, given the student's full history of past attempts. Instead of feeding raw multimodal content to the model at training time, each exercise and each attempt is pre-encoded once into a fixed-size embedding vector using a frozen VLM backbone. The Transformer then learns to reason over sequences of these vectors.

---

## How the Input Sequence is Built

This is the core design choice of this experiment.

For a student who has completed `t` exercises, the model receives a sequence of `2t + 1` embedding vectors:

```
[ emb(ex_0), emb(att_0), emb(ex_1), emb(att_1), ..., emb(ex_{t-1}), emb(att_{t-1}), emb(ex_t) ]
```

- **`emb(ex_i)`** — the embedding of exercise `i`, produced by passing the exercise's text description and screenshot through the frozen VLM and pooling its hidden states.
- **`emb(att_i)`** — the embedding of the student's attempt on exercise `i`, produced by passing the attempt's text (answer given, duration, timestamp, outcome) through the same frozen VLM.
- The final token **`emb(ex_t)`** is the *target* exercise the student is about to attempt. There is no attempt embedding for it yet.

The Transformer reads this interleaved sequence and predicts from the final token (`emb(ex_t)`) whether the student will answer correctly. The prediction head is applied to the last real token in the sequence.

The sequence is variable-length (different students have different history lengths) and is capped at `max_exercises` past attempts by the dataset loader. Sequences in a batch are zero-padded to the same length and a boolean attention mask is passed to the model so padding positions are ignored.

---

## Why Pre-compute Embeddings?

- Each exercise is encoded **once** regardless of how many students attempt it — far cheaper than feeding screenshots to the VLM for every training step.
- Each attempt is likewise encoded once. With millions of interactions, this makes training feasible.
- The frozen backbone produces deterministic vectors, so embeddings can be cached to disk and reused across experiments.
- The lightweight Transformer trains on top of fixed features, enabling fast iteration on architecture and hyperparameters without re-running the VLM.

---

## Code Overview

### `data_preparation/exercise_embedding.py`
Defines `ExerciseDataset` (one item = one exercise, text + optional screenshot) and `encode_all`, which runs the frozen VLM over all exercises in batches and pools the final hidden states into a single vector per exercise. Pooling mode is `"last"` by default (hidden state of the last non-padding token).

### `data_preparation/attempt_embedding.py`
Defines `InteractionDataset` (one item = one attempt, text only — no image) and `encode_interactions`, which runs the same frozen VLM over all raw interaction rows to produce one embedding vector per attempt.

### `data_preparation/create_exercises_embedding.ipynb`
Calls `encode_all` on the exercises table and saves the result to `data/exercises_embedded.parquet` (one row per exercise, columns: `exercise_id`, `embedding`).

### `data_preparation/create_interactions_embedding.ipynb`
Calls `encode_interactions` on the full interactions table and saves the result to partitioned files `data/interactions_embedded_part_*.parquet` (columns: `interaction_id`, `embedding`).

### `data_preparation/create_train_val_interactions.ipynb`
Joins the interaction embeddings back onto the full interactions table (adding the `interaction_embedding` column), then performs an 80/20 student-level split and saves:
- `data/train_interactions_embedded.parquet`
- `data/val_interactions_embedded.parquet`

### `data_processing.py`
Implements `KTDataset`, which:
- Loads exercise and interaction embeddings into contiguous NumPy arrays for fast lookup.
- Groups interactions by student and sorts them chronologically.
- For each `__getitem__` call, samples a random time step `t` in a student's history and builds the interleaved sequence `[emb(ex_0), emb(att_0), ..., emb(ex_{t-1}), emb(att_{t-1}), emb(ex_t)]` as a single `(2t+1, D)` tensor.

`kt_collate_fn` pads variable-length sequences in a batch to the same length and constructs the boolean attention mask.

### `models.py`
Defines `KTTransformer`:
- A linear projection maps each `D`-dimensional input vector to the internal model dimension `d_model`.
- Learnable positional embeddings are added.
- A stack of standard Transformer encoder layers (pre-norm, causal masking is **not** used — all tokens in the history are visible to each other) processes the sequence.
- The hidden state at the position of the last real token (the target exercise embedding) is passed through a two-layer MLP head to produce a scalar logit for binary classification.

### `train.py`
End-to-end training script. Loads config, builds datasets and data loaders, instantiates `KTTransformer`, and runs training with `BCEWithLogitsLoss`. Logs validation loss and accuracy at the end of each epoch.

```bash
python train.py
```

---

## Configuration (`config.yaml`)

| Key | Description |
|---|---|
| `dataset` | Path to the directory containing the prepared parquet files |
| `max_exercises` | Maximum number of past attempts to include in the history sequence |
| `batch_size` | Training batch size |
| `epochs` | Number of training epochs |
| `max_steps` | If set, stops training after this many gradient steps |
| `lr` | Learning rate for AdamW |
| `d_input` | Dimension of the pre-computed embedding vectors (576 for SmolVLM-256M) |
| `d_model` | Internal Transformer hidden dimension |
| `n_heads` | Number of attention heads |
| `n_layers` | Number of Transformer encoder layers |
| `d_ff` | Feed-forward hidden dimension inside the encoder |
| `dropout` | Dropout rate |
| `max_seq_len` | Maximum sequence length for positional embeddings (must be ≥ `2 * max_exercises + 1`) |

---

## Running the Experiments

### 1. Create a virtual environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the embeddings — run notebooks in this order

#### Step 1 — Exercise embeddings

Open `data_preparation/create_exercises_embedding.ipynb`, set the paths to the exercises table and the screenshots directory at the top, and run all cells. This produces:

- `data/exercises_embedded.parquet` — one row per exercise with a `576`-dimensional `embedding` column

#### Step 2 — Interaction embeddings

Open `data_preparation/create_interactions_embedding.ipynb`, set the path to the raw interactions table, and run all cells. Because the interactions table can be large the output is written in chunks:

- `data/interactions_embedded_part_0.parquet` … `interactions_embedded_part_N.parquet`

#### Step 3 — Train / validation split

Open `data_preparation/create_train_val_interactions.ipynb` and run all cells. This joins the interaction embeddings onto the original interactions rows (adding the `interaction_embedding` column) and performs a student-level 80/20 split, producing:

- `data/train_interactions_embedded.parquet` — training set
- `data/val_interactions_embedded.parquet` — validation set

These are the files consumed by `train.py`.

### 3. Update `config.yaml`

The default paths already point to the `data/` directory. If you saved files elsewhere, update:

```yaml
dataset: path/to/your/data/folder
```

Adjust the model size or sequence budget as needed:

```yaml
max_exercises: 512   # truncate histories longer than this
d_model: 256
n_layers: 4
```

### 4. Train

```bash
cd embedded_data_experiments
python train.py
```
