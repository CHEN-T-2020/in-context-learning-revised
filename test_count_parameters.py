import torch
import torch.nn as nn

from models import MLPModel
from models import TransformerModel
from models import LSTMModel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model1 = MLPModel(n_dims=20, n_positions=41, n_embd=1024, n_layer=22)
model2 = TransformerModel(n_dims=20, n_positions=101, n_embd=256, n_layer=12)
model3 = LSTMModel(n_dims=20, n_positions=101, n_embd=512, n_layer=10)

num_params1 = count_parameters(model1)
num_params2 = count_parameters(model2)
num_params3 = count_parameters(model3)
print("MLP number of parameters: {:,}".format(num_params1))
print("Transformer number of parameters: {:,}".format(num_params2))
print("LSTM number of parameters: {:,}".format(num_params3))

# print(f"MLP model:{model1}")
# print(f"Transformer model:{model2}")
