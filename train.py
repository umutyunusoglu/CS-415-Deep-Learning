import glob
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import torchaudio
from tqdm import tqdm
import scipy.io.wavfile as wavfile
import yaml

# Custom Imports
# (Eğer SlakhChunkedDataset'i ayrı dosyaya koymadıysan bu class'ı scriptin içine yapıştır)
from Data.dataset import SlakhChunkedDataset
from model import TranscriptionNet

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "raw_data_dir": r"C:\Users\teoma\Downloads\slakh_raw",
    "root_dir": r"C:\Users\teoma\Downloads\slakh_processed",
    "save_path": "model_piano.pth",
    "target_class": "Piano",
    "sequence_length": 128,
    "batch_size": 16,
    "learning_rate": 0.0003,
    "pos_weight": 5.0,
    "epochs": 100,
    "threshold": 0.3,
    "num_workers": 4,
    "sample_rate": 16000,
    "hop_length": 512,
    "patience" :10,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
print(f"🚀 Running on device: {CONFIG['device']}")


# ==========================================
# 2. HELPER METRICS
# ==========================================
def calculate_f1(preds, targets):
    """Batch için F1 Skoru hesaplar (Binary)"""
    # Sigmoid ve Threshold uygulanmış preds beklenir (0 veya 1)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1.item()


def plot_training_results(history):
    """Loss ve F1 grafiklerini çizer ve kaydeder"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss Grafiği
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # F1 Grafiği
    ax2.plot(epochs, history["train_f1"], "b-", label="Training F1")
    ax2.plot(epochs, history["val_f1"], "r-", label="Validation F1")
    ax2.set_title("Training vs Validation F1 Score")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("📈 Grafik kaydedildi: training_curves.png")


# ==========================================
# 3. PREPROCESSING
# ==========================================
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
            should_process = (CONFIG["target_class"] == "All") or (
                info["inst_class"] == CONFIG["target_class"]
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

        torch.save(
            {"waveform": waveform.clone(), "target": target.clone().bool()}, save_path
        )
        count += 1

    print(f"✅ Preprocessing bitti. {count} yeni dosya oluşturuldu.")


# ==========================================
# 4. TRAINING LOOP WITH VALIDATION
# ==========================================
def train_model():
    print(f"🚀 Veri Seti Yükleniyor...")

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

    model = TranscriptionNet().to(CONFIG["device"])

    # Mel Spectrogram Setup
    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=229,
    ).to(CONFIG["device"])

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([CONFIG["pos_weight"]]).to(CONFIG["device"])
    )
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_loss = float("inf")

    print("🔥 Eğitim Başlıyor...")
    early_stop_counter=0
    for epoch in range(CONFIG["epochs"]):
        # --- TRAIN STEP ---
        model.train()
        train_loss, train_f1_sum = 0.0, 0.0
        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]",
            leave=False,
        )

        for waveform, target in loop:
            waveform, target = (
                waveform.to(CONFIG["device"]),
                target.to(CONFIG["device"]),
            )

            # Preprocessing (On-the-fly)
            with torch.no_grad():
                spec = mel_layer(waveform)
                spec = torch.log(spec + 1e-5)
                mean, std = (
                    spec.mean(dim=(1, 2), keepdim=True),
                    spec.std(dim=(1, 2), keepdim=True),
                )
                spec = (spec - mean) / (std + 1e-5)
                if spec.shape[-1] > CONFIG["sequence_length"]:
                    spec = spec[..., : CONFIG["sequence_length"]]

            optimizer.zero_grad()
            preds = model(spec)
            loss = criterion(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            train_loss += loss.item()
            # F1 Metric Calculation
            with torch.no_grad():
                pred_bin = (torch.sigmoid(preds) > CONFIG["threshold"]).float()
                train_f1_sum += calculate_f1(pred_bin, target)

            loop.set_postfix(loss=loss.item())

        # --- VALIDATION STEP ---
        model.eval()
        val_loss, val_f1_sum = 0.0, 0.0
        with torch.no_grad():
            for waveform, target in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Val]",
                leave=False,
            ):
                waveform, target = (
                    waveform.to(CONFIG["device"]),
                    target.to(CONFIG["device"]),
                )

                spec = mel_layer(waveform)
                spec = torch.log(spec + 1e-5)
                mean, std = (
                    spec.mean(dim=(1, 2), keepdim=True),
                    spec.std(dim=(1, 2), keepdim=True),
                )
                spec = (spec - mean) / (std + 1e-5)
                if spec.shape[-1] > CONFIG["sequence_length"]:
                    spec = spec[..., : CONFIG["sequence_length"]]

                preds = model(spec)
                loss = criterion(preds, target)
                val_loss += loss.item()

                pred_bin = (torch.sigmoid(preds) > CONFIG["threshold"]).float()
                val_f1_sum += calculate_f1(pred_bin, target)

        # --- METRICS & LOGGING ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_f1 = train_f1_sum / len(train_loader)
        avg_val_f1 = val_f1_sum / len(val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_f1"].append(avg_train_f1)
        history["val_f1"].append(avg_val_f1)

        print(f"Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(
            f"   Train F1:   {avg_train_f1:.4f} | Val F1:   {avg_val_f1:.4f} | LR: {current_lr:.6f}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(
                f"   ⭐ New Best Model! Saving... ({best_val_loss:.4f} -> {avg_val_loss:.4f})"
            )
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG["save_path"])
            early_stop_counter=0
        else:
            early_stop_counter += 1
            print(f"   (No improvement for {early_stop_counter} epochs)")
            if early_stop_counter >= CONFIG["patience"]:
                print("🛑 Early Stopping Triggered!")
                break

        print("-" * 50)

    plot_training_results(history)
    print("✅ Eğitim Tamamlandı.")


# ==========================================
# 5. MAIN (AUTO DETECT)
# ==========================================
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
