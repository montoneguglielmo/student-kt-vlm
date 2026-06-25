import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
from torch.utils.data import DataLoader
import numpy as np

class KTDataset(Dataset):
    def __init__(self, interactions_df, exercises_df, max_exercises=512):
        
        self.max_exercises = max_exercises
        
        # Exercise embeddings: small, keep as one array + id->row map
        self.ex_emb = np.stack(exercises_df["embedding"].to_numpy())  # (M, 576)
        self.ex_id2row = {eid: i for i, eid in enumerate(exercises_df["exercise_id"].to_list())}

        # Interaction embeddings: one big contiguous array, NO python objects
        sorted_df = interactions_df.sort("created_at")
        self.inter_emb = np.stack(sorted_df["interaction_embedding"].to_numpy())  # (N, 576)

        # Per-student: only integers (row idx into inter_emb, row idx into ex_emb, label)
        ex_rows = np.array([self.ex_id2row[e] for e in sorted_df["exercise_id"].to_list()], dtype=np.int64)
        labels  = sorted_df["data_correct"].to_numpy()
        uids    = sorted_df["user_id_int"].to_numpy()

        self.student_data = {}
        for i in range(len(uids)):
            self.student_data.setdefault(int(uids[i]), []).append((i, int(ex_rows[i]), bool(labels[i])))
        self.student_data = {u: r for u, r in self.student_data.items() if len(r) >= 2}
        self.student_ids = list(self.student_data.keys())
        
    def __len__(self):
        return len(self.student_ids)

    def __getitem__(self, idx):
        rows = self.student_data[self.student_ids[idx]]
        n = len(rows); t = random.randint(1, n - 1)
        
        hist_rows = rows[:t][-self.max_exercises:]
        
        history = []
        for inter_i, ex_i, _ in hist_rows:
            history.append(torch.from_numpy(self.ex_emb[ex_i].copy()))
            history.append(torch.from_numpy(self.inter_emb[inter_i].copy()))
        ex_i = rows[t][1]; correct = rows[t][2]
        history.append(torch.from_numpy(self.ex_emb[ex_i].copy()))
        return torch.stack(history), torch.tensor(float(correct))
    
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
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=kt_collate_fn,
        num_workers=8,
    )
    
    for histories, targets, mask in train_loader:
        print(histories.shape)