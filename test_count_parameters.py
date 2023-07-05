import torch
import torch.nn as nn

from models import MLPModel
from models import TransformerModel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model1 = MLPModel(n_dims=5, n_positions=11, n_embd=1024, n_layer=20)
model2 = TransformerModel(n_dims=5, n_positions=11, n_embd=256, n_layer=8)

num_params1 = count_parameters(model1)
num_params2 = count_parameters(model2)
print("MLP number of parameters: {:,}".format(num_params1))
print("Transformer number of parameters: {:,}".format(num_params2))

# print(f"MLP model:{model1}")
# print(f"Transformer model:{model2}")
