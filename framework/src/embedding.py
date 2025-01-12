import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class FcEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, cond_in_size=None):
        super(FcEncoderLayer, self).__init__()
     
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.fc = nn.Linear(d_model, cond_in_size) if cond_in_size is not None else nn.Identity()
       
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None,is_causal=False):
        
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        src = self.fc(src)
        return src

class MLP_X(nn.Module):
    def __init__(self, dim = 2, data_size = 128, cond_in_size = 128):
        super(MLP_X, self).__init__()
        self.fc1 = nn.Linear(dim * data_size, 2 * dim * data_size)
        self.fc2 = nn.Linear(2 * dim * data_size, cond_in_size)

    def forward(self, x):
        x = x.view(-1)
        x = F.celu(self.fc1(x))
        x = self.fc2(x)
        return x

# class ResidualMLP(nn.Module):
#     def __init__(self, dim=2, cond_in_size=128, layers=4):
#         super(ResidualMLP, self).__init__()

#         self.fc1 = nn.Linear(dim, 2 * dim)

#         self.intermediate_layers = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layers-2)])

#         self.fc_last = nn.Linear(2 * dim, cond_in_size)

#         if dim == cond_in_size:
#             self.shortcut = nn.Identity()
#         else:
#             self.shortcut = nn.Linear(dim, cond_in_size)

#     def forward(self, x):
#         identity = self.shortcut(x)
#         x = F.celu(self.fc1(x))

#         for layer in self.intermediate_layers:
#             x = F.celu(layer(x))
 
#         x = self.fc_last(x)


#         x = torch.mean(x, dim=0, keepdim=False)  
#         identity = identity.mean(dim=0) 

#         x += identity
#         return x

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=2):
        super(ResidualMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # First layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        # Second layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        # Third layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()

        # Output layer, ensuring output dimensions match the input dimensions
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Forward pass through the first two layers
        x1 = self.relu1(self.fc1(x))
        x2 = self.relu2(self.fc2(x1))
        x3 = self.relu3(self.fc3(x2))
        
        # Output layer
        x4 = self.fc4(x3)

        # Residual connection and layer normalization
        # x_out = x + x3  # Residual connection
        x_out = x4
        x_out = self.layer_norm(x_out)  # Layer normalization

        return x_out

# class Transformer_X(nn.Module):
#     def __init__(self, input_dim=2, cond_in_size=128, data_size=128, nhead=2, dim_feedforward=8):
#         super(Transformer_X, self).__init__()
#         self.input_dim = input_dim
#         self.cond_in_size = cond_in_size
#         encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
#         encoder_final_layer = FcEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, cond_in_size=cond_in_size)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=5)
#         self.transformer_encoder_final = nn.TransformerEncoder(encoder_final_layer, num_layers=1)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.transformer_encoder(x)
#         x = self.transformer_encoder_final(x)

#         x = x.squeeze(1)

#         x = torch.mean(x, dim=0)
#         return x

class Transformer_X(nn.Module):
    def __init__(self, input_dim=2, cond_in_size=128, data_size=128, nhead=2, dim_feedforward=64):
        super(Transformer_X, self).__init__()
        self.input_dim = input_dim
        self.cond_in_size = cond_in_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        # 定义Transformer的Encoder层
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # 定义自定义的Encoder层，假设FcEncoderLayer已经正确实现
        # encoder_final_layer = FcEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, cond_in_size=cond_in_size)
        # self.transformer_encoder_final = nn.TransformerEncoder(encoder_final_layer, num_layers=1)
        
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # 增加维度以匹配Transformer的输入要求
        x = x.unsqueeze(1)
        
        # 应用Transformer Encoder，并添加残差连接和层归一化
        residual = x
        x = self.transformer_encoder(x)
        x = x + residual  # 残差连接
        x = self.layer_norm(x)  # 层归一化

        # 应用最终的Transformer Encoder层
        # residual = x
        # x = self.transformer_encoder_final(x)
        # x = x + residual  # 残差连接
        # x = self.layer_norm(x)  # 层归一化

        # 修改最终的池化操作
        # 这里使用平均池化作为例子，您可以根据需要替换为其他类型的池化
        x = torch.mean(x, dim=1)  # 序列维度上的平均池化

        # 压缩到数据的维度
        x = x.squeeze(1)

        return x
    
class Embedding(nn.Module):
    def __init__(self, dim=64, cond_in_size=256):
        super(Embedding, self).__init__()
        # self.mlp_x =ResidualMLP(input_dim=dim, hidden_dim=cond_in_size, output_dim=dim)
        #self.linear_layer = nn.Linear(in_features=dim, out_features=dim)
        self.transformer = Transformer_X(input_dim=dim, cond_in_size=cond_in_size, nhead=dim)
    
    
    def forward(self, x, method='mlp'):
        if method == 'mlp':
            # x = self.mlp_x(x)
            # x = x.unsqueeze(0)
            return x
        elif method == 'mean':
            x = torch.mean(x, dim=0)
            # x = self.linear_layer(x)
            x = x.unsqueeze(0)
            return x
        elif method == 'transformer':
            x = self.transformer(x)
            # x = x.unsqueeze(0)
            return x
