import glob
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import IPython.display as ipd
import torchaudio
from tqdm import tqdm
import scipy.io.wavfile as wavfile
import yaml

# Custom Imports
from Data.dataset import SlakhTranscriptionDataset
from model import TranscriptionNet

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "raw_data_dir": r"D:\Ana\Projeler\CS 415\CS-415-Deep-Learning\Data\slakh\dataset",  # Ham verinin yolu
    "root_dir": r"D:\Ana\Projeler\CS 415\CS-415-Deep-Learning\Data\slakh\processed",
    "save_path": "model_piano.pth",
    "target_class": "Piano",
    "sequence_length": 128,
    "batch_size": 32,  # Transformer için uygun
    # --- KRİTİK DEĞİŞİKLİKLER ---
    "learning_rate": 0.0003,  # 0.005'ten 0.0001'e düşürdük (En önemli düzeltme)
    "pos_weight": 5.0,  # 10.0 çok agresifti, 1.0 yaparak dengeledik.
    "epochs": 100,  # Düşük learning rate ile öğrenmesi için epoch artmalı (5 yetmez)
    # ---------------------------
    "threshold": 0.3,
    "num_workers": 4,
    "sample_rate": 16000,
    "hop_length": 512,
    "split": "train",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
print(f"🚀 Running on device: {CONFIG['device']}")


# ==========================================
# 2. PREPROCESSING (ONE TIME RUN)
# ==========================================
def preprocess_dataset():
    input_dir = os.path.join(CONFIG["raw_data_dir"], CONFIG["split"])
    output_dir = CONFIG["root_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tracks = sorted(glob.glob(os.path.join(input_dir, "Track*")))
    print(f"🔄 Preprocessing: {len(tracks)} parça işlenecek -> {output_dir}")

    if len(tracks) == 0:
        print("❌ Hata: Raw veri bulunamadı. raw_data_dir yolunu kontrol et!")
        return

    for track_path in tqdm(tracks):
        track_name = os.path.basename(track_path)
        save_path = os.path.join(output_dir, track_name + ".pt")

        if os.path.exists(save_path):
            continue

        # Ses Yükle
        mix_path = os.path.join(track_path, "mix.flac")
        if not os.path.exists(mix_path):
            continue

        waveform, sr = torchaudio.load(mix_path)
        if sr != CONFIG["sample_rate"]:
            resampler = torchaudio.transforms.Resample(sr, CONFIG["sample_rate"])
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # MIDI Yükle
        meta_path = os.path.join(track_path, "metadata.yaml")
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)

        fs = CONFIG["sample_rate"] / CONFIG["hop_length"]
        total_frames = int(waveform.shape[1] / CONFIG["hop_length"])
        piano_roll_combined = np.zeros((88, total_frames), dtype=np.float32)

        for stem_key, info in meta["stems"].items():
            if CONFIG["target_class"] == "All":
                mid_path = os.path.join(track_path, "MIDI", f"{stem_key}.mid")
                if os.path.exists(mid_path):
                    try:
                        pm = pretty_midi.PrettyMIDI(mid_path)
                        pr = pm.get_piano_roll(fs=fs)
                        pr = pr[21:109, :]  # 88 tuş
                        common_len = min(pr.shape[1], piano_roll_combined.shape[1])
                        piano_roll_combined[:, :common_len] += pr[:, :common_len]
                    except:
                        pass

        piano_roll_combined = (piano_roll_combined > 0).astype(np.float32)
        target = torch.from_numpy(piano_roll_combined).unsqueeze(0)

        torch.save(
            {
                "waveform": waveform.clone(),  # Float16 tasarruf
                "target": target.clone().bool(),  # Bool tasarruf
            },
            save_path,
        )


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    print(f"🚀 Training Başlıyor: {CONFIG['device']} (TF32 + Scheduler)")

    # --- HIZ VE KARARLILIK AYARLARI ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dataset = SlakhTranscriptionDataset(
        root_dir=CONFIG["root_dir"],
        split="train",
        target_class=CONFIG["target_class"],
        sequence_length=CONFIG["sequence_length"],
    )

    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    if len(dataset) == 0:
        print("❌ Hata: Dataset boş! Önce preprocess modunu çalıştırın.")
        return None, None, False

    model = TranscriptionNet().to(CONFIG["device"])

    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=229,
    ).to(CONFIG["device"])

    # Pos Weight: 2.0 yaparak gerçek notalara biraz daha ağırlık veriyoruz
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([2.0]).to(CONFIG["device"])
    )

    # Başlangıç hızı: 0.0001
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # --- YENİ EKLENEN SCHEDULER ---
    # Sabır (patience): 3 epoch boyunca loss düşmezse devreye girer.
    # Faktör (factor): 0.5 -> Hızı yarıya indirir (0.0001 -> 0.00005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_loss = float("inf")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}", file=sys.stdout)

        for batch_idx, (waveform, target) in enumerate(loop):
            waveform = waveform.to(CONFIG["device"], non_blocking=True)
            target = target.to(CONFIG["device"], non_blocking=True)

            if torch.isnan(waveform).any():
                continue

            optimizer.zero_grad(set_to_none=True)

            # Forward
            spec = mel_layer(waveform)
            spec = torch.log(spec + 1e-5)

            # Normalizasyon
            mean = spec.mean(dim=(1, 2), keepdim=True)
            std = spec.std(dim=(1, 2), keepdim=True)
            spec = (spec - mean) / (std + 1e-5)

            if spec.shape[-1] > CONFIG["sequence_length"]:
                spec = spec[..., : CONFIG["sequence_length"]]

            preds = model(spec)
            loss = criterion(preds, target)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if torch.isnan(loss):
                continue

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- EPOCH SONU ---
        avg_loss = running_loss / len(loader) if len(loader) > 0 else 0

        # Şu anki Learning Rate'i öğrenelim
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\tEpoch {epoch + 1} Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # --- SCHEDULER ADIMI ---
        # Scheduler'a bu epoch'un loss değerini söylüyoruz.
        # Eğer loss iyileşmediyse, LR'yi düşürecek.
        scheduler.step(avg_loss)

        if avg_loss < best_loss and avg_loss > 0:
            print(f"\t🔥 Improved: {best_loss:.4f} -> {avg_loss:.4f}. Saving...")
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG["save_path"])

    return model, loader, True


# ==========================================
# 5. VISUALIZATION & AUDIO GENERATION
# ==========================================


def save_audio_compatible(filename, audio_data, sample_rate):
    """
    Sesi Float32 formatından 16-bit PCM formatına çevirir ve normalizasyon yapar.
    """
    # 1. Normalizasyon (Sesi -1 ile 1 arasına çek)
    max_val = np.abs(audio_data).max()
    if max_val > 0:
        audio_data = audio_data / max_val

    # 2. 16-bit Tam Sayıya Çevirme
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # 3. Kaydet
    wavfile.write(filename, sample_rate, audio_int16)
    print(f"💾 Ses Kaydedildi: {filename}")


def piano_roll_to_pretty_midi(piano_roll, fs=31.25, program=0):
    """
    Piano Roll matrisini (88 x Zaman) alır, MIDI objesine çevirir.
    fs: Saniyedeki kare sayısı (Frame Rate)
    """
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)  # 0: Akustik Piyano

    # Bitişleri yakalamak için padding
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")
    velocity_changes = np.diff(piano_roll).T
    note_on_time = np.zeros(notes)

    for time, row in enumerate(velocity_changes):
        velocity = row
        change_indices = np.where(velocity != 0)[0]

        for note_idx in change_indices:
            if velocity[note_idx] > 0:
                note_on_time[note_idx] = time
            else:
                note_number = note_idx + 21  # MIDI numarasına çevir (0 -> 21)
                start_time = note_on_time[note_idx] / fs
                end_time = time / fs

                # Nota oluştur (Velocity 100 standarttır)
                note = pretty_midi.Note(
                    velocity=100, pitch=note_number, start=start_time, end=end_time
                )
                instrument.notes.append(note)

    pm.instruments.append(instrument)
    return pm


def generate_visualization():
    print(f"🚀 Görselleştirme ve Ses Üretimi Başlıyor...")

    # 1. Modeli Yükle
    model = TranscriptionNet().to(CONFIG["device"])
    try:
        state = torch.load(CONFIG["save_path"], map_location=CONFIG["device"])
        model.load_state_dict(state)
        print(f"✅ Model Yüklendi: {CONFIG['save_path']}")
    except:
        print("❌ Model dosyası bulunamadı, eğitim yapılmamış olabilir.")
        return

    model.eval()

    # 2. Veri Yükleyici
    dataset = SlakhTranscriptionDataset(
        root_dir=CONFIG["root_dir"],
        split="train",
        target_class=CONFIG["target_class"],
        sequence_length=CONFIG["sequence_length"],
    )

    # Piyano içeren bir örnek bulana kadar dene (Max 20 deneme)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    found = False

    with torch.no_grad():
        for i, (waveform, target) in enumerate(loader):
            if i > 20:
                break  # Çok aramayalım
            if target.sum() == 0:
                continue  # Boş ise geç

            found = True
            print(f"🎹 Piyano içeren örnek bulundu (İterasyon: {i})")

            waveform = waveform.to(CONFIG["device"])

            # --- PREPROCESSING (Eğitimle Birebir Aynı) ---
            mel_layer = torchaudio.transforms.MelSpectrogram(
                sample_rate=CONFIG["sample_rate"],
                n_fft=2048,
                hop_length=CONFIG["hop_length"],
                n_mels=229,
            ).to(CONFIG["device"])

            spec = mel_layer(waveform)
            spec = torch.log(spec + 1e-5)

            # Normalizasyon
            mean = spec.mean(dim=(1, 2), keepdim=True)
            std = spec.std(dim=(1, 2), keepdim=True)
            spec = (spec - mean) / (std + 1e-5)

            if spec.shape[-1] > CONFIG["sequence_length"]:
                spec = spec[..., : CONFIG["sequence_length"]]

            # Tahmin
            logits = model(spec)
            probs = torch.sigmoid(logits)
            preds = (probs > CONFIG["threshold"]).float()

            # Loop'tan çık
            break

    if not found:
        print("⚠️ Uyarı: Piyano içeren örnek bulunamadı, rastgele bir tane çiziliyor.")

    # 3. Grafik Çizimi
    spec_np = spec[0, 0].cpu().numpy()
    target_np = target[0, 0].cpu().numpy()
    probs_np = probs[0, 0].cpu().numpy()
    pred_np = preds[0, 0].cpu().numpy()

    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax[0].imshow(spec_np, aspect="auto", origin="lower", cmap="inferno")
    ax[0].set_title("Spectrogram")
    ax[1].imshow(target_np, aspect="auto", origin="lower", cmap="magma")
    ax[1].set_title("Target")
    ax[2].imshow(probs_np, aspect="auto", origin="lower", cmap="viridis")
    ax[2].set_title("Probabilities (Model Output)")
    ax[3].imshow(pred_np, aspect="auto", origin="lower", cmap="magma")
    ax[3].set_title(f"Prediction (Threshold: {CONFIG['threshold']})")

    plt.tight_layout()
    plt.savefig("output_comparison.png")
    print("✅ Grafik Kaydedildi: output_comparison.png")

    # 4. SES ÜRETİMİ (AUDIO SYNTHESIS)
    print("🎧 Ses Dosyaları Oluşturuluyor...")

    # Frame Rate Hesabı (Önemli: Yoksa ses çok hızlı/yavaş çalar)
    # Örn: 16000 / 512 = 31.25 kare/saniye
    fs_calc = CONFIG["sample_rate"] / CONFIG["hop_length"]

    # A) Gerçek Ses (Ground Truth)
    pm_true = piano_roll_to_pretty_midi(target_np, fs=fs_calc)
    audio_true = pm_true.synthesize(fs=CONFIG["sample_rate"])
    save_audio_compatible("ground_truth.wav", audio_true, CONFIG["sample_rate"])

    # B) Tahmin Edilen Ses (Prediction)
    pm_pred = piano_roll_to_pretty_midi(pred_np, fs=fs_calc)
    audio_pred = pm_pred.synthesize(fs=CONFIG["sample_rate"])
    save_audio_compatible("prediction.wav", audio_pred, CONFIG["sample_rate"])

    print(
        "\n✨ İşlem Tamam! 'ground_truth.wav' ve 'prediction.wav' dosyalarını dinleyebilirsin."
    )


# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    mode = "eval"  # "preprocess", "train", veya "eval" seç

    if mode == "preprocess":
        preprocess_dataset()
    elif mode == "train":
        # preprocess_dataset() # İstersen burayı açabilirsin
        model, loader, success = train_model()
        if success:
            generate_visualization()
    elif mode == "eval":
        generate_visualization()
