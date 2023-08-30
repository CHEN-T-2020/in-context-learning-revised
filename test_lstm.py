import torch
import torch.nn as nn

class MultiLayerLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, batch_first=True, bidirectional=False, p_dropout=0.0):
        super(MultiLayerLSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                batch_first=batch_first,
                dropout=p_dropout,
                
            ) for layer in range(num_layers)
        ])

    def forward(self, x):
        outputs = []
        for layer in self.lstm_layers:
            x, _ = layer(x)
            outputs.append(x)
        return outputs

# 定义输入参数
batch_size = 64
seq_length = 82
input_size = 256
hidden_size = 256
num_layers = 10  # 3层的多层LSTM

# 创建多层LSTM模型
multi_layer_lstm = MultiLayerLSTM(num_layers, input_size, hidden_size)

# 生成随机输入数据
input_data = torch.randn(batch_size, seq_length, input_size)

# 前向传播
outputs = multi_layer_lstm(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {outputs[-1].shape}")  # 输出最后一层的输出形状

# 输出每一层的输出形状
for layer_idx, output in enumerate(outputs):
    print(f"Layer {layer_idx} output shape:", output.shape)
