import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Transformer zaman kavramını bilmez (paralel işlediği için).
    Bu katman, sesin hangi saniyesinde olduğumuzu modele matematiksel olarak söyler.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [Batch, Time, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 1)) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class TranscriptionNetSmall(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 1. CNN (Görüntü İşleme / Özellik Çıkarma) ---
        # Frekans eksenini sıkıştıracağız, Zaman eksenine dokunmayacağız.
        self.conv_blocks = nn.Sequential(
            CNNBlock(1, 32),  # [B, 32, 229, T]
            CNNBlock(32, 64, pool=True),  # [B, 64, 114, T]
            CNNBlock(64, 128, pool=True),  # [B, 128, 57, T]
            CNNBlock(128, 256, pool=True),  # [B, 256, 28, T]
        )

        # CNN Çıkış boyutu: 512 kanal * 28 frekans = 14336
        cnn_out_size = 512 * 28
        d_model = 512  # Transformer'ın çalışacağı boyut

        # --- 2. PROJECTION & POSITIONAL ENCODING ---
        self.projection = nn.Linear(cnn_out_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)

        # --- 3. TRANSFORMER ENCODER (LSTM Yerine Gelen Güç) ---
        # GPU dostu, paralel, stabil.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,  # 8 farklı açıdan bakar
            dim_feedforward=512,  # Ara katman genişliği
            dropout=0.1,
            batch_first=True,  # [Batch, Time, Feat] formatı için şart
            norm_first=True,  # Pre-Norm: Eğitim kararlılığını artırır (NaN düşmanıdır)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # --- 4. CLASSIFIER ---
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),  # Son bir normalizasyon
            nn.Linear(d_model, 1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 88),  # 88 Piyano tuşu
        )

    def forward(self, x):
        # x: [Batch, 1, 229, Time]

        # 1. CNN
        x = self.conv_blocks(x)  # [Batch, 512, 28, Time]

        # 2. Reshape
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)  # [Batch, Time, Channels, Freq]
        x = x.reshape(b, t, c * f)  # [Batch, Time, 14336]

        # 3. Projection & Positional Encoding
        x = self.projection(x)  # [Batch, Time, 512]
        x = self.pos_encoder(x)  # Konum bilgisini ekle

        # 4. Transformer (LSTM'in yaptığı işi yapar ama paralel)
        x = self.transformer(x)  # [Batch, Time, 512]

        # 5. Classifier
        x = self.fc(x)  # [Batch, Time, 88]

        # 6. Output Shape Fix
        x = x.permute(0, 2, 1).unsqueeze(1)  # [Batch, 1, 88, Time]

        return x
