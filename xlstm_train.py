import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from tqdm import tqdm
import random
import yaml
import warnings
import pretty_midi
# Custom Imports
from Data.dataset import SlakhChunkedDataset
from xlstm import CNN_xLSTM_AMT

# Uyarıları sessize al
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "raw_data_dir": "c:\\Users\\Jade\\Documents\\GPUvenv\\slakh2100_flac_redux",
    "root_dir": "processed_guitar_slakh",
    "save_path": "model_xlstm_guitar.pth",
    "target_class": "Electric Guitar",
    "sequence_length": 128,
    "batch_size": 32,
    "learning_rate": 0.0001, # xLSTM genelde daha düşük LR sever
    "pos_weight": 3.0,
    "label_smoothing": 0.1,
    "epochs": 35,
    "patience": 12,
    "threshold": 0.35,
    "num_workers": 4,
    "sample_rate": 16000,
    "hop_length": 512,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

def preprocess_dataset(input_data: str):
    # Sadece Train klasörünü baz alıyoruz, çünkü validation'ı random split ile ayıracağız
    input_dir = os.path.join(CONFIG["raw_data_dir"], input_data)
    output_dir = os.path.join(CONFIG["root_dir"], input_data)
    os.makedirs(output_dir, exist_ok=True)

    tracks = sorted(glob.glob(os.path.join(input_dir, "Track*")))
    print(f"🔄 Preprocessing Başlıyor: {len(tracks)} parça -> {output_dir}")

    count = 0
    for track_path in tqdm(tracks):
        track_name = os.path.basename(track_path)
        save_path = os.path.join(output_dir, track_name + ".pt")

        if os.path.exists(save_path):
            continue

        mix_path = os.path.join(track_path, "mix.flac")
        if not os.path.exists(mix_path):
            continue

        # Ses İşleme
        waveform, sr = torchaudio.load(mix_path)
        if sr != CONFIG["sample_rate"]:
            resampler = torchaudio.transforms.Resample(sr, CONFIG["sample_rate"])
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # MIDI İşleme
        meta_path = os.path.join(track_path, "metadata.yaml")
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)

        fs = CONFIG["sample_rate"] / CONFIG["hop_length"]
        total_frames = int(waveform.shape[1] / CONFIG["hop_length"])
        piano_roll_combined = np.zeros((88, total_frames), dtype=np.float32)
        has_target = False

        for stem_key, info in meta["stems"].items():
            should_process = (
                 CONFIG["target_class"] in info["midi_program_name"] 
            )
            if should_process:
                mid_path = os.path.join(track_path, "MIDI", f"{stem_key}.mid")
                if os.path.exists(mid_path):
                    try:
                        pm = pretty_midi.PrettyMIDI(mid_path)
                        pr = pm.get_piano_roll(fs=fs)
                        pr = pr[21:109, :]
                        common_len = min(pr.shape[1], piano_roll_combined.shape[1])
                        piano_roll_combined[:, :common_len] += pr[:, :common_len]
                        has_target = True
                    except:
                        pass

        piano_roll_combined = (piano_roll_combined > 0).astype(np.float32)
        target = torch.from_numpy(piano_roll_combined).unsqueeze(0)
        if has_target:
            torch.save(
                {"waveform": waveform.clone(), "target": target.clone().bool()}, 
                save_path
            )
            count += 1

    print(f"✅ Preprocessing bitti. {count} yeni dosya oluşturuldu.")

# ==========================================
# 2. METRIC FUNCTION
def calculate_f1(preds, targets):
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return (2 * precision * recall / (precision + recall + 1e-8)).item()

# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
def train_model():
    print(f"🚀 Device: {CONFIG['device']} | Architecture: xLSTM | Precision: FP32")
    
    # 1. Dataset & Loader Setup
    train_dataset = SlakhChunkedDataset(
        root_dir=os.path.join(CONFIG["root_dir"], "train"),
        sequence_length=CONFIG["sequence_length"],
    )

    val_dataset = SlakhChunkedDataset(
        root_dir=os.path.join(CONFIG["root_dir"], "validation"),
        sequence_length=CONFIG["sequence_length"],
    )   

    print(
        f"📊 Veri Bölümü: {len(train_dataset)} Train | {len(val_dataset)} Validation"
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    # 2. Model & Loss Setup (n_mels=229 ve d_model=512 senin mimarinle uyumlu)
    model = CNN_xLSTM_AMT(n_mels=229, d_model=512).to(CONFIG["device"])
    
    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"], n_fft=2048, hop_length=CONFIG["hop_length"], n_mels=229
    ).to(CONFIG["device"])

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG["pos_weight"]]).to(CONFIG["device"]))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # --- RESUME LOGIC ---
    start_epoch = 0
    best_val_loss = 1
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    if os.path.exists(CONFIG["save_path"]):
        print(f"🔄 Checkpoint bulundu: {CONFIG['save_path']}")
        checkpoint = torch.load(CONFIG["save_path"], map_location=CONFIG["device"])
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            history = checkpoint.get('history', history)
            print(f"✅ Epoch {start_epoch} üzerinden devam ediliyor.")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Sadece ağırlıklar yüklendi.")

    print("🔥 Eğitim Başlıyor...")
    early_stop_counter = 0

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        train_loss, train_f1_sum = 0.0, 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False)

        for waveform, target in loop:
            waveform = waveform.to(CONFIG["device"], non_blocking=True)
            target = target.to(CONFIG["device"], non_blocking=True)
            
            # Label Smoothing

            with torch.no_grad():
                spec = torch.log(mel_layer(waveform) + 1e-5)
                spec = (spec - spec.mean()) / (spec.std() + 1e-5)
                if spec.shape[-1] > CONFIG["sequence_length"]:
                    spec = spec[..., :CONFIG["sequence_length"]]

            # Standart FP32 Training
            optimizer.zero_grad(set_to_none=True)
            
            preds = model(spec)
            loss = criterion(preds, target)
            
            loss.backward()
            
            # xLSTM için gradyan kırpma (clipping) hayati önem taşır
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                pred_bin = (torch.sigmoid(preds) > CONFIG["threshold"]).float()
                train_f1_sum += calculate_f1(pred_bin, target)
            
            loop.set_postfix(loss=loss.item())

        # --- VALIDATION ---
        model.eval()
        val_loss, val_f1_sum = 0.0, 0.0
        with torch.no_grad():
            for waveform, target in val_loader:
                waveform, target = waveform.to(CONFIG["device"]), target.to(CONFIG["device"])
                
                spec = torch.log(mel_layer(waveform) + 1e-5)
                spec = (spec - spec.mean()) / (spec.std() + 1e-5)
                
                preds = model(spec[..., :CONFIG["sequence_length"]])
                v_loss = criterion(preds, target.float())
                
                val_loss += v_loss.item()
                pred_bin = (torch.sigmoid(preds) > CONFIG["threshold"]).float()
                val_f1_sum += calculate_f1(pred_bin, target)

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_f1": train_f1_sum / len(train_loader),
            "val_f1": val_f1_sum / len(val_loader)
        }
        
        for k, v in metrics.items(): history[k].append(v)
        
        print(f"Epoch {epoch+1}: T-Loss: {metrics['train_loss']:.4f} V-Loss: {metrics['val_loss']:.4f} | T-F1: {metrics['train_f1']:.4f} V-F1: {metrics['val_f1']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
  

        scheduler.step(metrics['val_loss'])

        if metrics['val_loss'] < best_val_loss:
            print(f"⭐ New Best Loss! Saving... ({best_val_loss:.4f} -> {metrics['val_loss']:.4f})")
            best_val_loss = metrics['val_loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history
            }, CONFIG["save_path"])
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= CONFIG["patience"]:
                print("🛑 Early Stopping!")
                break
        print("-" * 50)

if __name__ == "__main__":
    # train ve validation ayrı ayrı preprocess edilir
    for split in ["train", "validation"]:
        processed_files = glob.glob(os.path.join(CONFIG["root_dir"], split, "*.pt"))

        if len(processed_files) == 0:
            print(f"⚠️ {split} için işlenmiş veri bulunamadı. Preprocessing başlatılıyor...")
            preprocess_dataset(split)
        else:
            print(f"✅ {split}: {len(processed_files)} adet işlenmiş dosya bulundu. Preprocess atlanıyor.")
    
    train_model()
