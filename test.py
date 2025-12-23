import glob
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import pretty_midi
import os
import tqdm
import yaml
import sys
import random

# Model dosyasının (model.py) yanınızda olduğundan emin olun
try:
    from model import TranscriptionNet
except ImportError:
    print(
        "⚠️ UYARI: 'model.py' bulunamadı. Sadece preprocessing yapacaksanız sorun yok."
    )

# ================= CONFIG =================
CONFIG = {
    "raw_data_dir": r"D:\Datasets\Slakh2100",  # Slakh veri setinin ana klasörü
    "processed_dir": r"processed_test_slakh",  # İşlenmiş dosyaların yeri
    "model_path": "model_piano.pth",
    "sequence_length": 256,
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "target_class": "Piano",
}


# ================= 1. YARDIMCI FONKSİYONLAR =================
def save_audio(filename, audio_data, sr):
    """NumPy array'i WAV dosyası olarak kaydeder."""
    audio_int16 = (audio_data / np.abs(audio_data).max() * 32767).astype(np.int16)
    wavfile.write(filename, sr, audio_int16)


def piano_roll_to_pretty_midi(piano_roll, fs):
    """Piano roll matrisini MIDI nesnesine çevirir."""
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


# ================= 2. PREPROCESSING (ÖN İŞLEME) =================
def preprocess_test_dataset():
    """Test verilerini işleyip .pt dosyalarına çevirir."""
    input_dir = os.path.join(CONFIG["raw_data_dir"], "test")
    output_dir = CONFIG["processed_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tracks = sorted(glob.glob(os.path.join(input_dir, "Track*")))
    if not tracks:
        print(f"❌ HATA: {input_dir} bulunamadı!")
        return

    print(f"🔄 Test Verisi İşleniyor ({len(tracks)} parça)...")

    for track_path in tqdm.tqdm(tracks):
        track_name = os.path.basename(track_path)
        save_path = os.path.join(output_dir, track_name + ".pt")

        if os.path.exists(save_path):
            continue

        try:
            mix_path = os.path.join(track_path, "mix.flac")
            meta_path = os.path.join(track_path, "metadata.yaml")

            if not os.path.exists(mix_path):
                continue

            # Ses Yükle & Resample
            waveform, sr = torchaudio.load(mix_path)
            if sr != CONFIG["sample_rate"]:
                waveform = torchaudio.transforms.Resample(sr, CONFIG["sample_rate"])(
                    waveform
                )
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Mono

            # Metadata & MIDI
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)

            fs = CONFIG["sample_rate"] / CONFIG["hop_length"]
            total_frames = int(waveform.shape[1] / CONFIG["hop_length"]) + 1
            piano_roll = np.zeros((88, total_frames), dtype=np.float32)

            for stem_key, info in meta["stems"].items():
                if info["inst_class"] == CONFIG["target_class"]:
                    mid_path = os.path.join(track_path, "MIDI", f"{stem_key}.mid")
                    if os.path.exists(mid_path):
                        pm = pretty_midi.PrettyMIDI(mid_path)
                        pr = pm.get_piano_roll(fs=fs)[21:109, :]
                        common_len = min(pr.shape[1], piano_roll.shape[1])
                        piano_roll[:, :common_len] += pr[:, :common_len]

            target = torch.from_numpy(piano_roll > 0).unsqueeze(0)
            torch.save({"waveform": waveform, "target": target.bool()}, save_path)

        except Exception as e:
            print(f"Hata {track_name}: {e}")


# ================= 3. F1 SCORE DEĞERLENDİRME =================
# ================= 3. F1 SCORE DEĞERLENDİRME (DÜZELTİLMİŞ) =================
def evaluate_dataset():
    """Tüm dataset üzerinde Precision, Recall ve F1 Score hesaplar."""
    print(f"\n📊 F1 Değerlendirmesi Başlıyor... Device: {CONFIG['device']}")

    # Chunk boyutu (Bellek hatası almamak için 4000 frame)
    EVAL_CHUNK_SIZE = 4000

    model = TranscriptionNet().to(CONFIG["device"])
    try:
        model.load_state_dict(
            torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
        )
        model.eval()
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return

    files = glob.glob(os.path.join(CONFIG["processed_dir"], "*.pt"))
    if not files:
        print("❌ Test dosyaları bulunamadı.")
        return

    total_tp, total_fp, total_fn = 0, 0, 0

    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=229,
    ).to(CONFIG["device"])

    with torch.no_grad():
        for file_path in tqdm.tqdm(files, desc="Evaluating"):
            data = torch.load(file_path)
            waveform = data["waveform"].to(CONFIG["device"])
            target = data["target"].to(CONFIG["device"])

            # 1. Spektrogram Oluştur
            spec = mel_layer(waveform)
            spec = torch.log(spec + 1e-5)
            spec = (spec - spec.mean()) / (spec.std() + 1e-5)

            total_time = spec.shape[-1]

            # 2. CHUNKING
            all_preds = []
            for start in range(0, total_time, EVAL_CHUNK_SIZE):
                end = min(start + EVAL_CHUNK_SIZE, total_time)
                chunk = spec[:, :, start:end]

                logits = model(chunk.unsqueeze(0))
                if logits.dim() == 4:
                    logits = logits.squeeze(1)

                probs = torch.sigmoid(logits)
                preds_chunk = (probs > CONFIG["threshold"]).float()
                all_preds.append(preds_chunk)

            full_preds = torch.cat(all_preds, dim=-1)

            # 3. Boyut Eşitleme ve TÜR DÖNÜŞÜMÜ (Hatayı Çözen Yer)
            min_len = min(full_preds.shape[-1], target.shape[-1])
            full_preds = full_preds[..., :min_len]

            # BURASI DEĞİŞTİ: .float() eklendi.
            # Boolean tensörü Float'a çeviriyoruz ki matematiksel işlem yapabilelim.
            target_slice = target[..., :min_len].float()

            # 4. Hesaplama (Artık hepsi float olduğu için hata vermez)
            tp = (full_preds * target_slice).sum().item()
            fp = (full_preds * (1 - target_slice)).sum().item()
            fn = ((1 - full_preds) * target_slice).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

    epsilon = 1e-7
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    print("\n" + "=" * 30)
    print(f"🏁 TEST SONUÇLARI (Threshold: {CONFIG['threshold']})")
    print("=" * 30)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("=" * 30)


# ================= 4. TEK ÖRNEK GÖRSELLEŞTİRME =================
def visualize_single_example():
    """Mutlaka notanın olduğu bir kesiti bulup görselleştirir."""
    print("\n🎨 Görselleştirme Başlıyor (Nota içeren kesit aranıyor)...")

    model = TranscriptionNet().to(CONFIG["device"])
    try:
        model.load_state_dict(
            torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
        )
        model.eval()
    except:
        print("Model yüklenemedi, rastgele weight ile devam ediliyor.")

    files = glob.glob(os.path.join(CONFIG["processed_dir"], "*.pt"))
    if not files:
        print("❌ Dosya yok.")
        return

    # 1. İçinde piyano olan bir DOSYA bul
    data_path = None
    # 100 deneme hakkı verelim
    random.shuffle(files)  # Her seferinde farklı dosyalara baksın
    for f in files[:100]:
        temp = torch.load(f)
        if temp["target"].sum() > 0:  # Dosyada en az 1 nota var mı?
            data_path = f
            break

    if not data_path:
        print("❌ Hiçbir dosyada piyano notası bulunamadı! Preprocess hatalı olabilir.")
        return

    print(f"📂 Seçilen Dosya: {os.path.basename(data_path)}")
    data = torch.load(data_path)
    waveform = data["waveform"].to(CONFIG["device"])
    target = data["target"].to(CONFIG["device"])

    # 2. Dosya içinde piyano olan bir KESİT (Chunk) bul
    found_chunk = False
    max_attempts = 200  # 200 kere rastgele denesin

    seq_len = CONFIG["sequence_length"]
    hop = CONFIG["hop_length"]

    for _ in range(max_attempts):
        # Rastgele bir başlangıç noktası seç
        start_frame = np.random.randint(0, max(1, target.shape[-1] - seq_len))
        end_frame = start_frame + seq_len

        target_slice = target[:, :, start_frame:end_frame]

        # Eğer bu kesitte en az 5 tane '1' (nota) varsa bunu seç
        if target_slice.sum() > 5:
            start_sample = start_frame * hop
            end_sample = end_frame * hop

            waveform = waveform[:, start_sample:end_sample]
            target = target_slice  # Güncelle
            found_chunk = True
            break

    if not found_chunk:
        print(
            "⚠️ Uyarı: Bu dosyada nota var ama seçilen kesitlere denk gelmedi. Rastgele bir yer gösteriliyor."
        )
        # Yine de kesit alalım ki hata vermesin
        start_frame = 0
        waveform = waveform[:, : seq_len * hop]
        target = target[:, :, :seq_len]

    # --- Tahmin ve Çizim ---
    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"], n_fft=2048, hop_length=hop, n_mels=229
    ).to(CONFIG["device"])

    with torch.no_grad():
        spec = mel_layer(waveform)
        spec = torch.log(spec + 1e-5)
        spec = (spec - spec.mean()) / (spec.std() + 1e-5)

        logits = model(spec.unsqueeze(0))
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > CONFIG["threshold"]).float()

    # Grafik Ayarları (Daha anlaşılır renkler)
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Spectrogram
    ax[0].imshow(spec[0].cpu(), aspect="auto", origin="lower", cmap="inferno")
    ax[0].set_title("Input Spectrogram (Giriş Sesi)")
    ax[0].set_ylabel("Frekans")

    # 2. Ground Truth (Artık SİYAH arka plan, BEYAZ nota)
    # cmap="gray" yaparsak: 0=Siyah, 1=Beyaz olur. Daha anlaşılır.
    ax[1].imshow(
        target[0].cpu(), aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1
    )
    ax[1].set_title("Ground Truth (Gerçek Notalar)")
    ax[1].set_ylabel("Piyano Tuşları")

    # 3. Prediction
    ax[2].imshow(
        preds[0].cpu(), aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
    )
    ax[2].set_title(f"Prediction (Tahmin) - Threshold: {CONFIG['threshold']}")
    ax[2].set_ylabel("Piyano Tuşları")

    plt.tight_layout()
    plt.savefig("test_visualization.png")
    print("🖼️ Grafik kaydedildi: test_visualization.png")

    # Ses Kaydet
    fs = CONFIG["sample_rate"] / hop
    pm = piano_roll_to_pretty_midi(preds[0].cpu().numpy(), fs)
    audio_pred = pm.synthesize(fs=16000)
    save_audio("test_prediction.wav", audio_pred, 16000)
    print("🎧 Ses kaydedildi: test_prediction.wav")


# ================= MAIN =================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        CONFIG["raw_data_dir"] = sys.argv[1]

    # 1. Veri Hazırla (Yoksa)
    if not os.path.exists(CONFIG["processed_dir"]) or not os.listdir(
        CONFIG["processed_dir"]
    ):
        preprocess_test_dataset()
    else:
        print("✅ İşlenmiş veri bulundu, preprocess atlanıyor.")

    # 2. Genel Başarıyı Ölç (F1 Score)
    evaluate_dataset()

    # 3. Bir Tane Örnek Çiz ve Kaydet
    visualize_single_example()
