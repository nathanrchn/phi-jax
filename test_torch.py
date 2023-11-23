from torch import load
from requests import get

model = load("pytorch_model.bin", map_location="cpu")

for param_name in model:
    print(param_name)