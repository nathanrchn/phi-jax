from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/no_robots")["train_sft"]

a = dataset.map(lambda x: x, batched=True, batch_size=8)

print(a[:8])