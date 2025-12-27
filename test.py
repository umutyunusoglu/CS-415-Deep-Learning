import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import pretty_midi
import os
from model import TranscriptionNetSmall  # Model dosyanın yanında olmalı

# ================= CONFIG =================
CONFIG = {
    "model_path": "model_piano.pth",  # Eğitilmiş model
    "test_file_path": None,  # None ise datasetten rastgele seçer
    "processed_dir": r"D:\Ana\Projeler\CS 415\CS-415-Deep-Learning\Data\slakh\processed",
    "sequence_length": 128,
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.4,  # Testte biraz daha kesin sonuç isteyebiliriz
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def save_audio(filename, audio_data, sr):
    audio_int16 = (audio_data / np.abs(audio_data).max() * 32767).astype(np.int16)
    wavfile.write(filename, sr, audio_int16)


def piano_roll_to_pretty_midi(piano_roll, fs):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")
    velocity_changes = np.diff(piano_roll).T
    note_on_time = np.zeros(notes)

    for time, row in enumerate(velocity_changes):
        for note_idx in np.where(row != 0)[0]:
            if row[note_idx] > 0:
                note_on_time[note_idx] = time
            else:
                start = note_on_time[note_idx] / fs
                end = time / fs
                inst.notes.append(pretty_midi.Note(100, note_idx + 21, start, end))
    pm.instruments.append(inst)
    return pm


def run_test():
    print(f"🧪 Test Modu Başlatılıyor... ({CONFIG['device']})")

    # 1. Modeli Yükle
    model = TranscriptionNetSmall().to(CONFIG["device"])
    try:
        model.load_state_dict(
            torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
        )
        model.eval()
        print("✅ Model başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return

    # 2. Veri Yükle
    if CONFIG["test_file_path"]:
        data = torch.load(CONFIG["test_file_path"])
    else:
        import glob
        import random

        files = glob.glob(os.path.join(CONFIG["processed_dir"], "*.pt"))
        if not files:
            print("❌ Veri bulunamadı.")
            return

        # Piyano içeren dosya bulana kadar dene
        for _ in range(50):
            f = random.choice(files)
            data = torch.load(f)
            if data["target"].sum() > 0:
                print(f"📂 Rastgele dosya seçildi: {os.path.basename(f)}")
                break

    waveform = data["waveform"].to(CONFIG["device"])
    target = data["target"].to(CONFIG["device"])

    # Rastgele bir kesit al (Chunk)
    if waveform.shape[1] > CONFIG["sequence_length"] * CONFIG["hop_length"]:
        start_frame = np.random.randint(0, target.shape[-1] - CONFIG["sequence_length"])
        end_frame = start_frame + CONFIG["sequence_length"]
        start_sample = start_frame * CONFIG["hop_length"]
        end_sample = end_frame * CONFIG["hop_length"]

        waveform = waveform[:, start_sample:end_sample]
        target = target[:, :, start_frame:end_frame]

    # 3. Model Tahmini
    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=229,
    ).to(CONFIG["device"])

    with torch.no_grad():
        spec = mel_layer(waveform)
        spec = torch.log(spec + 1e-5)
        spec = (spec - spec.mean()) / (spec.std() + 1e-5)

        logits = model(spec)
        probs = torch.sigmoid(logits)
        preds = (probs > CONFIG["threshold"]).float()

    # 4. Görselleştirme
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].imshow(spec[0].cpu(), aspect="auto", origin="lower", cmap="inferno")
    ax[0].set_title("Input Spectrogram")
    ax[1].imshow(target[0].cpu(), aspect="auto", origin="lower", cmap="gray_r")
    ax[1].set_title("Ground Truth (Gerçek)")
    ax[2].imshow(preds[0].cpu(), aspect="auto", origin="lower", cmap="magma")
    ax[2].set_title(f"Prediction (Tahmin) - Thr: {CONFIG['threshold']}")
    plt.tight_layout()
    plt.savefig("test_result.png")
    print("🖼️ Grafik kaydedildi: test_result.png")

    # 5. Ses Üretimi
    fs = CONFIG["sample_rate"] / CONFIG["hop_length"]
    audio_pred = piano_roll_to_pretty_midi(preds[0].cpu().numpy(), fs).synthesize(
        fs=16000
    )
    save_audio("test_prediction.wav", audio_pred, 16000)
    print("🎧 Ses kaydedildi: test_prediction.wav")


if __name__ == "__main__":
    run_test()
