import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random


class KTDataset(Dataset):
    def __init__(self, interactions_df, exercises_df):
        """
        Args:
            interactions_df: Polars DataFrame with columns:
                user_id_int (u32), exercise_id (str), created_at (datetime),
                data_correct (bool), interaction_embedding (array[f32, 576])
            exercises_df: Polars DataFrame with columns:
                exercise_id (str), embedding (array[f32, 576])
        """
        self.interactions_df = interactions_df
        self.exercises_df = exercises_df

        # Build exercise embedding lookup: exercise_id -> tensor
        self.exercise_embeddings = {
            row["exercise_id"]: torch.tensor(row["embedding"], dtype=torch.float32)
            for row in exercises_df.iter_rows(named=True)
        }

        # Only store the list of student IDs and filter out those with < 2 interactions
        student_counts = interactions_df.group_by("user_id_int").len()
        student_counts = student_counts.filter(student_counts["len"] >= 2)
        self.student_ids = student_counts["user_id_int"].to_list()

    def __len__(self):
        return len(self.student_ids)

    def __getitem__(self, idx):
        uid = self.student_ids[idx]

        # Filter and sort only this student's interactions
        student_df = (
            self.interactions_df
            .filter(self.interactions_df["user_id_int"] == uid)
            .sort("created_at")
        )

        n = len(student_df)
        t = random.randint(1, n - 1)

        # Build history: [ex_0, att_0, ex_1, att_1, ..., att_{t-1}, ex_t]
        history = []
        for i in range(t):
            row = student_df.row(i, named=True)
            ex_emb = self.exercise_embeddings[row["exercise_id"]]
            att_emb = torch.tensor(row["interaction_embedding"], dtype=torch.float32)
            history.append(ex_emb)
            history.append(att_emb)

        # Append target exercise (without its attempt — that's what we predict)
        target_row = student_df.row(t, named=True)
        ex_t = self.exercise_embeddings[target_row["exercise_id"]]
        history.append(ex_t)

        target = float(target_row["data_correct"])

        history_tensor = torch.stack(history)  # (2t + 1, D)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return history_tensor, target_tensor
    
def kt_collate_fn(batch):
    histories, targets = zip(*batch)
    seq_lengths = torch.tensor([h.size(0) for h in histories])
    padded_histories = pad_sequence(histories, batch_first=True)
    targets = torch.stack(targets)

    # Boolean mask: True where real tokens are, False where padding is
    max_len = padded_histories.size(1)
    attention_mask = torch.arange(max_len).unsqueeze(0) < seq_lengths.unsqueeze(1)  # (B, max_seq_len)

    return padded_histories, targets, attention_mask


if __name__ == '__main__':
    
    import polars as pl
    
    train_df = pl.read_parquet("../data/train_interactions_embedded.parquet")
    val_df = pl.read_parquet("../data/val_interactions_embedded.parquet")
    exercises_df = pl.read_parquet("../data/exercises_embedded.parquet")

    train_dataset = KTDataset(train_df, exercises_df)
    val_dataset = KTDataset(val_df, exercises_df)
    print(f"{len(train_dataset)} training examples")
    print(f"{len(val_dataset)} validation examples")
    
    history_tensor, target_tensor = train_dataset[0]
    print(f"Student ID: {train_dataset.student_ids[0]}")
    print(f"History tensor shape: {history_tensor.shape}")  # (2t+1, 576)
    print(f"Target score: {target_tensor.item()}")
    print(f"Number of past attempts: {(history_tensor.shape[0] - 1) // 2}")