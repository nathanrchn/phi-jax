import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("HuggingFaceH4/no_robots")["train_sft"]

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
tokenizer.pad_token = tokenizer.eos_token

print(type(tokenizer(["caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi", "caca", "pipi"], padding="max_length", max_length=40, )["input_ids"]))