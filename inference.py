import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import pretty_midi
import os
import glob
import random

# Model dosyanızın (model.py) yanımızda olduğundan emin olun
from model import TranscriptionNet

# ================= CONFIG =================
CONFIG = {
    # BURAYI KENDİ DOSYA YOLLARINA GÖRE DÜZENLE
    # Program numaraları General MIDI standardına göredir.
    "models": {
        "Piano": {
            "path": "D:\Ana\Projeler\CS 415\model_piano_transformer_teoman.pth",
            "program": 0,  # 0: Acoustic Grand Piano (Genel Piyano sesi)
        },
        "Bass": {
            "path": "D:\Ana\Projeler\CS 415\model_bass_transformer.pth",
            "program": 33,  # 33: Electric Bass (Finger) (Hem akustik hem elektrik bası kapsar)
        },
        "Guitar": {
            "path": "model_guitar_transformer.pth",
            "program": 25,  # 25: Acoustic Guitar (Steel) veya 27: Electric Clean
            # Tüm gitarlar birleştiği için 25 (Akustik) veya 27 (Clean) en temizidir.
        },
        "Strings": {
            "path": "D:\Ana\Projeler\CS 415\model_strings_transformer.pth",
            "program": 48,  # 48: String Ensemble 1 (Keman, Viyola, Çello hepsi içinde olduğu için Grup sesi)
        },
    },
    # Test verilerinin olduğu klasör
    "processed_dir": r"D:\Ana\Projeler\CS 415\processed_test_slakh",
    "test_file_path": None,  # None ise rastgele seçer
    "sequence_length": 128,  # Yaklaşık 4 saniye
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.4,  # Note On için eşik değeri
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# ================= HELPER FUNCTIONS =================


def load_models(device):
    """Tanımlı tüm modelleri yükler."""
    loaded_models = {}
    print(f"🔄 Modeller yükleniyor ({device})...")

    for name, info in CONFIG["models"].items():
        if not os.path.exists(info["path"]):
            print(f"   ⚠️ UYARI: {name} modeli bulunamadı ({info['path']}). Atlanıyor.")
            loaded_models[name] = None
            continue

        try:
            model = TranscriptionNet().to(device)
            # weights_only=False, eski PyTorch sürümleri veya tam model kaydı için gerekebilir
            model.load_state_dict(torch.load(info["path"], map_location=device))
            model.eval()
            loaded_models[name] = model
            print(f"   ✅ {name} modeli yüklendi.")
        except Exception as e:
            print(f"   ❌ {name} modeli yüklenirken hata: {e}")
            loaded_models[name] = None

    return loaded_models


def save_audio(filename, audio_data, sr):
    if len(audio_data) == 0:
        return
    # Normalize et (patlamayı önle)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_int16 = (audio_data / max_val * 32767).astype(np.int16)
        wavfile.write(filename, sr, audio_int16)


def piano_roll_to_pretty_midi(piano_roll_dict, fs):
    """
    Her sınıf için ayrı kanal açar ve MIDI oluşturur.
    """
    pm = pretty_midi.PrettyMIDI()

    for inst_name, prediction in piano_roll_dict.items():
        program_num = CONFIG["models"][inst_name]["program"]

        # Enstrümanı oluştur
        inst = pretty_midi.Instrument(
            program=program_num, is_drum=False, name=inst_name
        )

        # [88, Time] -> [Time, 88] yapıp diff alıyoruz
        piano_roll = np.pad(prediction, [(0, 0), (1, 1)], "constant")  # Kenar dolgusu
        velocity_changes = np.diff(piano_roll).T
        note_on_time = np.zeros(prediction.shape[0])

        for time, row in enumerate(velocity_changes):
            for note_idx in np.where(row != 0)[0]:
                if row[note_idx] > 0:
                    # Note ON
                    note_on_time[note_idx] = time
                else:
                    # Note OFF
                    start = note_on_time[note_idx] / fs
                    end = time / fs
                    # Çok kısa notaları atla (gürültü engelleme)
                    if end - start > 0.05:
                        inst.notes.append(
                            pretty_midi.Note(100, note_idx + 21, start, end)
                        )

        pm.instruments.append(inst)

    return pm


# ================= MAIN PIPELINE =================


def run_ensemble_test():
    print(f"🚀 Ensemble (Merged Classes) Test Başlatılıyor...")

    # 1. Modelleri Yükle
    models = load_models(CONFIG["device"])

    # Eğer hiçbir model yüklenemediyse dur
    if all(m is None for m in models.values()):
        print("❌ Hiçbir model yüklenemedi. Dosya yollarını kontrol et.")
        return

    # 2. Test Dosyası Bul
    if CONFIG["test_file_path"]:
        files = [CONFIG["test_file_path"]]
    else:
        files = glob.glob(os.path.join(CONFIG["processed_dir"], "*.pt"))

    if not files:
        print(f"❌ '{CONFIG['processed_dir']}' klasöründe .pt dosyası bulunamadı.")
        return

    # Rastgele dosya seçimi
    waveform, filename = None, ""
    print("🔍 Karışık ses içeren bir dosya aranıyor...")

    # 50 deneme yap, tamamen sessiz olmayan bir dosya bul
    for _ in range(50):
        f = random.choice(files)
        try:
            data = torch.load(f, map_location=CONFIG["device"], weights_only=False)
            # Hedefte (Target) biraz aktivite olsun ki görselde bir şey görelim
            if data["target"].sum() > 50:
                waveform = data["waveform"].to(CONFIG["device"])
                filename = os.path.basename(f)
                break
        except:
            continue

    if waveform is None:
        print("❌ Uygun test dosyası bulunamadı.")
        return

    print(f"📂 Seçilen Dosya: {filename}")

    # 3. Kesit Al (Chunking)
    if waveform.shape[1] > CONFIG["sequence_length"] * CONFIG["hop_length"]:
        # Rastgele bir yerden 4 saniye kes
        start_frame = np.random.randint(
            0, waveform.shape[1] // CONFIG["hop_length"] - CONFIG["sequence_length"]
        )
        end_frame = start_frame + CONFIG["sequence_length"]
        start_sample = start_frame * CONFIG["hop_length"]
        end_sample = end_frame * CONFIG["hop_length"]

        waveform_chunk = waveform[:, start_sample:end_sample]
    else:
        waveform_chunk = waveform

    # --- YENİ EKLENEN KISIM: Orijinal Sesi Kaydet ---
    print("💾 Orijinal ses parçası kaydediliyor...")
    original_audio_np = waveform_chunk.cpu().numpy().squeeze()
    save_audio("original_input_chunk.wav", original_audio_np, CONFIG["sample_rate"])
    print("🎧 Orijinal Kayıt: original_input_chunk.wav")
    # ------------------------------------------------

    # 4. Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=229,
    ).to(CONFIG["device"])

    with torch.no_grad():
        spec = mel_transform(waveform_chunk)
        spec = torch.log(spec + 1e-5)
        spec = (spec - spec.mean()) / (spec.std() + 1e-5)

        if spec.dim() == 2:
            spec = spec.unsqueeze(0).unsqueeze(0)
        elif spec.dim() == 3:
            spec = spec.unsqueeze(1)

    # 5. Inference (Her Sınıf İçin Ayrı)
    predictions = {}
    print("\n🔍 Tahminler yürütülüyor...")

    for name, model in models.items():
        if model is None:
            continue

        with torch.no_grad():
            logits = model(spec)
            probs = torch.sigmoid(logits)
            # Threshold uygulama
            preds = (probs > CONFIG["threshold"]).float().squeeze().cpu().numpy()
            predictions[name] = preds
            print(f"   🎹 {name}: Tamamlandı (Max Prob: {probs.max():.2f})")

    # 6. Görselleştirme (Matplotlib)
    print("📊 Grafikler oluşturuluyor...")
    fig, axes = plt.subplots(len(predictions) + 1, 1, figsize=(10, 12), sharex=True)

    # Giriş Spectrogramı
    axes[0].imshow(spec.squeeze().cpu(), aspect="auto", origin="lower", cmap="inferno")
    axes[0].set_title(f"Input Spectrogram ({filename})")
    axes[0].set_ylabel("Freq")

    # Model Çıktıları
    for i, (name, pred_map) in enumerate(predictions.items()):
        ax = axes[i + 1]
        ax.imshow(
            pred_map,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_title(f"Prediction: {name}")
        ax.set_ylabel("MIDI Pitch")
        ax.grid(color="white", alpha=0.1)

    plt.xlabel("Time Frames")
    plt.tight_layout()
    plt.savefig("ensemble_merged_result.png")
    print("🖼️  Görsel kaydedildi: ensemble_merged_result.png")

    # 7. MIDI ve Ses
    fs = CONFIG["sample_rate"] / CONFIG["hop_length"]
    combined_midi = piano_roll_to_pretty_midi(predictions, fs)

    combined_midi.write("ensemble_merged_output.mid")
    print("💾 MIDI kaydedildi: ensemble_merged_output.mid")

    try:
        # pretty_midi'nin synthesize fonksiyonu
        # Not: Kaliteli ses için sisteminizde FluidSynth olması gerekebilir.
        audio_data = combined_midi.synthesize(fs=16000)
        save_audio("ensemble_merged_prediction.wav", audio_data, 16000)
        print("🎧 Ses (WAV) kaydedildi: ensemble_merged_prediction.wav")
    except Exception as e:
        print(f"⚠️ Ses sentezleme uyarısı: {e}")
        print("   (Sadece .mid dosyasını kullanabilirsiniz)")


if __name__ == "__main__":
    run_ensemble_test()
