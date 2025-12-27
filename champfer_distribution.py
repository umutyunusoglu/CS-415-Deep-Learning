"""
TEST SCRIPT: CONTINUOUS ERROR LANDSCAPE (KDE CONTOUR MAP)
- Fixes the 'PositionalEncoding' crash by using Windowing.
- Uses Gaussian KDE to plot a smooth, topographic error map.
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
from scipy.stats import gaussian_kde
import scipy.ndimage  # Bunu en tepeye importlara ekle

# Custom Imports
from model import TranscriptionNet

# ================= CONFIG =================
CONFIG = {
    "model_path": "D:\Ana\Projeler\CS 415\piyano transformer\model_piano.pth",
    "processed_dir": "D:\Ana\Projeler\CS 415\processed_test_slakh",
    "sequence_length": 128,
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# ================= CORE LOGIC =================


def get_displacement_vectors(preds, targets):
    """Tahmin ve Hedef arasındaki vektörel farkı (dt, dp) hesaplar"""
    batch_size = preds.shape[0]
    preds = preds.squeeze(1)
    targets = targets.squeeze(1)

    batch_dt = []
    batch_dp = []

    for b in range(batch_size):
        p_indices = torch.nonzero(preds[b], as_tuple=False)
        t_indices = torch.nonzero(targets[b], as_tuple=False)

        if p_indices.size(0) == 0 or t_indices.size(0) == 0:
            continue

        p_coords = p_indices.float()
        t_coords = t_indices.float()

        dists = torch.cdist(p_coords, t_coords, p=2)
        _, min_indices = torch.min(dists, dim=1)
        closest_targets = t_coords[min_indices]

        diffs = p_coords - closest_targets

        # diffs[:, 0] -> Pitch (Y), diffs[:, 1] -> Time (X)
        batch_dp.extend(diffs[:, 0].cpu().numpy().tolist())
        batch_dt.extend(diffs[:, 1].cpu().numpy().tolist())

    return batch_dt, batch_dp


def plot_contour_landscape(
    time_errors, pitch_errors, output_path="error_contour_map.png"
):
    """
    5 Milyon nokta için optimize edilmiş Hızlı Contour Map (Histogram + Gaussian Blur).
    """
    print("🎨 Hızlı Contour haritası hesaplanıyor (Histogram Method)...")

    # Veriyi Numpy array'e çevir
    x = np.array(time_errors)
    y = np.array(pitch_errors)

    # Odaklanma Alanı
    xlim = 8
    ylim = 8
    bins = 100  # Çözünürlük (100x100 ızgara)

    # --- OPTİMİZASYON BURADA ---
    # 1. KDE yerine 2D Histogram al (Milisaniyeler sürer)
    # range=[[-xlim, xlim], [-ylim, ylim]] diyerek sadece merkeze odaklanıyoruz
    H, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[-xlim, xlim], [-ylim, ylim]]
    )

    # 2. Histogramı yumuşat (KDE efekti verir)
    # sigma=1.5 veya 2.0 pürüzsüzlük ayarıdır.
    import scipy.ndimage

    H_smooth = scipy.ndimage.gaussian_filter(H.T, sigma=1.5)

    # Grid merkezlerini ayarla
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    xx, yy = np.meshgrid(xc, yc)

    # --- ÇİZİM ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#121212")

    # Dolgulu Contour
    cf = ax.contourf(xx, yy, H_smooth, levels=60, cmap="inferno")

    # Çizgili Contour
    ax.contour(xx, yy, H_smooth, levels=15, colors="white", linewidths=0.5, alpha=0.3)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Hata Yoğunluğu (Density)", rotation=270, labelpad=15)

    # Eksenler
    ms_per_frame = (CONFIG["hop_length"] / CONFIG["sample_rate"]) * 1000
    ax.set_title(
        f"Sürekli Hata Yoğunluk Haritası (N={len(x)})\nHistogram Density Estimation",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(f"Zaman Kayması (Frame)\n1 Frame ≈ {ms_per_frame:.1f}ms", fontsize=12)
    ax.set_ylabel("Pitch Kayması (Semitone)", fontsize=12)

    ax.axhline(0, color="white", linestyle=":", alpha=0.4)
    ax.axvline(0, color="white", linestyle=":", alpha=0.4)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"🔥 Contour Map kaydedildi: {output_path}")


# ================= RUNNER (WINDOWING FIXED) =================


def run_analysis():
    print(f"🚀 Contour Analizi Başlıyor... ({CONFIG['device']})")

    model = TranscriptionNet().to(CONFIG["device"])
    model.load_state_dict(
        torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    )
    model.eval()

    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=229,
    ).to(CONFIG["device"])

    files = glob.glob(os.path.join(CONFIG["processed_dir"], "*.pt"))
    all_dt = []
    all_dp = []

    print(f"📊 {len(files)} dosya taranıyor...")

    for filepath in tqdm(files):
        try:
            data = torch.load(
                filepath, map_location=CONFIG["device"], weights_only=False
            )
            waveform = data["waveform"].to(CONFIG["device"]).squeeze(0)
            target = data["target"].to(CONFIG["device"]).squeeze(0)

            # --- WINDOWING (Bu kısım modelin patlamasını önler) ---
            seq_len = CONFIG["sequence_length"]
            total_frames = target.shape[1]
            num_windows = total_frames // seq_len

            if num_windows == 0:
                continue

            valid_waveforms = []
            valid_targets = []

            for i in range(num_windows):
                start_frame = i * seq_len
                end_frame = start_frame + seq_len
                t_window = target[:, start_frame:end_frame]

                if t_window.sum() == 0:
                    continue  # Sessiz kısımları atla

                start_sample = start_frame * CONFIG["hop_length"]
                end_sample = end_frame * CONFIG["hop_length"]
                w_window = waveform[start_sample:end_sample]

                valid_targets.append(t_window)
                valid_waveforms.append(w_window)

            if not valid_targets:
                continue

            # Batch oluştur
            batch_waveforms = torch.stack(valid_waveforms)
            batch_targets = torch.stack(valid_targets)

            # --- INFERENCE ---
            with torch.no_grad():
                spec = mel_layer(batch_waveforms)
                spec = torch.log(spec + 1e-5)
                mean = spec.mean(dim=(1, 2), keepdim=True)
                std = spec.std(dim=(1, 2), keepdim=True)
                spec = (spec - mean) / (std + 1e-8)

                if spec.shape[-1] > seq_len:
                    spec = spec[..., :seq_len]
                if spec.dim() == 3:
                    spec = spec.unsqueeze(1)

                logits = model(spec)
                preds = (torch.sigmoid(logits) > CONFIG["threshold"]).float()

                if preds.shape[-1] != seq_len and preds.shape[-2] == seq_len:
                    preds = preds.transpose(1, 2)
                if preds.dim() == 4:
                    preds = preds.squeeze(1)

            # --- VEKTÖR TOPLAMA ---
            dt, dp = get_displacement_vectors(preds, batch_targets)
            all_dt.extend(dt)
            all_dp.extend(dp)

        except Exception as e:
            # Hatalı dosyayı atla ama tüm script durmasın
            print(f"\nDosya hatası ({os.path.basename(filepath)}): {e}")
            continue

    print(f"📊 Toplam {len(all_dt)} hata noktası ile Contour Haritası çiziliyor...")

    if len(all_dt) > 0:
        plot_contour_landscape(all_dt, all_dp)
    else:
        print("❌ Hata verisi toplanamadı.")


if __name__ == "__main__":
    run_analysis()
