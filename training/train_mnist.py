from dit import DiT_teeny_tiny, DiT_tiny
from datasets import load_dataset

model = DiT_tiny()

dataset = load_dataset("mnist")

print(dataset)