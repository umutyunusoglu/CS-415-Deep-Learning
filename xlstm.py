import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class mLSTMBlock(nn.Module):
    def __init__(self, d_model, head_dim=64, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, head_dim)
        self.k_proj = nn.Linear(d_model, head_dim)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.i_gate = nn.Linear(d_model, 1)
        self.f_gate = nn.Linear(d_model, 1)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) # mLSTM için Pre-Norm yapısı

    def forward(self, x):
        # Pre-Norm
        residual = x
        x = self.norm(x)
        
        B, T, D = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # Kapılar ve Stabilite
        i = torch.exp(torch.clamp(self.i_gate(x), max=8))
        f = torch.exp(torch.clamp(self.f_gate(x), max=8))
        
        # Matrix Memory Mekanizması
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1]**0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        gated_attn = F.softmax(attn_weights, dim=-1) * i.transpose(-2, -1)
        out = torch.matmul(gated_attn, v)
        
        return residual + self.dropout(self.out_proj(out))

class CNN_xLSTM_AMT(nn.Module):
    def __init__(self, n_mels=229, d_model=512, n_layers=4, n_output=88):
        super().__init__()
        
        # 1. CNN Frontend (Transformer modelindeki güçlü yapıyı koruyoruz)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.MaxPool2d((2, 1)), # 229 -> 114
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ELU(),
            nn.MaxPool2d((2, 1)), # 114 -> 57
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ELU(),
            nn.MaxPool2d((2, 1))  # 57 -> 28
        )
        
        cnn_out_dim = 256 * 28 
        self.projection = nn.Linear(cnn_out_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.ln_after_cnn = nn.LayerNorm(d_model)
        
        # 2. xLSTM Layers (mLSTM Blokları)
        self.xlstm_layers = nn.ModuleList([
            mLSTMBlock(d_model, dropout=0.1) for _ in range(n_layers)
        ])
        
        # 3. Classifier Head (Transformer modelindeki gibi derin kafa)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_output)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        
        # CNN
        x = self.cnn(x) # [B, 256, 28, T]
        
        # Reshape & Projection
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.projection(x)
        x = self.ln_after_cnn(x)
        x = self.pos_encoder(x)
        
        # xLSTM Layers
        for layer in self.xlstm_layers:
            x = layer(x)
            
        # Classifier
        x = self.classifier(x) # [B, T, 88]
        
        return x.permute(0, 2, 1).unsqueeze(1) # [B, 1, 88, T]
