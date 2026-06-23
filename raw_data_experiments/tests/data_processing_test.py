import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from torch.utils.data import DataLoader
from pathlib import Path
from data_processing import KTVLMDataset

DATASET = Path("data")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
).to(device)
model.eval()

ds = KTVLMDataset(
    sequences_path=DATASET / "sequences.parquet",
    exercises_path=DATASET / "exercises.parquet",
    screenshots_dir="../MIAAM/data/screenshots/compressed",
    min_history=3,
    max_history=40,
    history_window=1
)

sample = ds[100]
print('THIS IS THE SAMPLE:')
print(sample)
print('\n\n')
img = sample["images"][0]
img.save("model_input.png")

content = [
    {"type": "image"},
    {"type": "text", "text": "Describe this image in great detail."},
]
messages = [{"role": "user", "content": content}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=[prompt],
    images=[[img]],
    return_tensors="pt",
).to(device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)

# Strip the prompt tokens so you only print the generated continuation.
gen = out[:, inputs["input_ids"].shape[1]:]
print('THIS THE MODEL ANSWER:')
print(processor.batch_decode(gen, skip_special_tokens=True)[0])