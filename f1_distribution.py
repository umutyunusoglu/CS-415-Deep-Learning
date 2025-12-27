"""
TEST SCRIPT FOR TRANSCRIPTION (F1 + Chamfer Distance)
Iterates through EVERY 4-second window in every file.
Filters out silent windows individually.
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm

# Custom Imports
from model import TranscriptionNet

# ================= CONFIG =================
CONFIG = {
    "model_path": "piyano transformer/model_piano.pth",
    "processed_dir": "processed_test_slakh",  # Test verisi yolu
    "sequence_length": 128,  # approx 4 seconds
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.5,  # Chamfer için genelde 0.5 iyidir
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# ================= METRIC FUNCTIONS =================


def calculate_f1_batch(preds, targets):
    """
    Calculates F1 scores for a BATCH. Returns a LIST of floats.
    preds, targets: [Batch, 88, Time]
    """
    batch_size = preds.shape[0]
    scores = []

    for i in range(batch_size):
        p = preds[i].float().reshape(-1)
        t = targets[i].float().reshape(-1)

        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        scores.append(f1.item())

    return scores


def calculate_chamfer_batch(preds, targets):
    """
    Calculates Move-Based (Chamfer) Distance for a BATCH.
    Returns a LIST of distances (lower is better).

    preds, targets: [Batch, 88, Time]
    """
    batch_size = preds.shape[0]
    distances = []

    # Thresholding is already done before this function, but let's be safe
    # preds is 0 or 1 float here

    preds = preds.squeeze(1)  # [Batch, 88, Time] (if dim 1 exists)
    targets = targets.squeeze(1)

    for b in range(batch_size):
        # Koordinatları al: (Pitch, Time)
        p_coords = torch.nonzero(preds[b]).float()
        t_coords = torch.nonzero(targets[b]).float()

        # --- CASE 1: İkisi de boş (Sessizlik doğru tahmin edilmiş) ---
        if p_coords.size(0) == 0 and t_coords.size(0) == 0:
            distances.append(0.0)
            continue

        # --- CASE 2: Biri boş biri dolu (Tamamen yanlış) ---
        if p_coords.size(0) == 0 or t_coords.size(0) == 0:
            # Ceza: Sabit bir yüksek değer veya o anki frame sayısı kadar ceza
            # Burada 'ortalama kaydırma' mantığı çöküyor,
            # o yüzden keyfi bir 'Time Length / 2' cezası verelim.
            distances.append(CONFIG["sequence_length"] / 2.0)
            continue

        # --- CASE 3: Normal Hesaplama (Pairwise Distance) ---
        # [Num_Preds, Num_Targets] matrisi
        dists = torch.cdist(p_coords, t_coords, p=2)  # Euclidean distance

        # Tahminler gerçeğe ne kadar uzak? (Her tahmin noktası için en yakın gerçeği bul)
        min_dist_p2t, _ = torch.min(dists, dim=1)

        # Gerçekler tahmine ne kadar uzak? (Her gerçek nokta için en yakın tahmini bul)
        min_dist_t2p, _ = torch.min(dists, dim=0)

        # İki yönlü ortalama (Chamfer Distance)
        chamfer_dist = torch.mean(min_dist_p2t) + torch.mean(min_dist_t2p)

        distances.append(chamfer_dist.item())

    return distances


# ================= PROCESSING LOOP =================


def process_file_windows(model, mel_layer, filepath):
    """
    Returns: (list_of_f1s, list_of_distances, skipped_count)
    """
    try:
        data = torch.load(filepath, map_location=CONFIG["device"], weights_only=False)
        waveform = data["waveform"].to(CONFIG["device"]).squeeze(0)
        target = data["target"].to(CONFIG["device"]).squeeze(0)  # [88, Frames]

        seq_len = CONFIG["sequence_length"]
        total_frames = target.shape[1]
        num_windows = total_frames // seq_len

        if num_windows == 0:
            return [], [], 0

        valid_waveforms = []
        valid_targets = []

        # --- SLICING ---
        for i in range(num_windows):
            start_frame = i * seq_len
            end_frame = start_frame + seq_len

            t_window = target[:, start_frame:end_frame]

            # Filter Silence
            if t_window.sum() == 0:
                continue

            start_sample = start_frame * CONFIG["hop_length"]
            end_sample = end_frame * CONFIG["hop_length"]
            w_window = waveform[start_sample:end_sample]

            valid_targets.append(t_window)
            valid_waveforms.append(w_window)

        if not valid_targets:
            return [], [], num_windows

        # Stack Batch
        batch_waveforms = torch.stack(valid_waveforms)  # [Batch, Samples]
        batch_targets = torch.stack(valid_targets)  # [Batch, 88, Frames]

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

            # Fix output shape if needed [Batch, 88, Time]
            if preds.shape[-1] != seq_len and preds.shape[-2] == seq_len:
                preds = preds.transpose(1, 2)
            if preds.dim() == 4:  # [B, 1, 88, T] -> [B, 88, T]
                preds = preds.squeeze(1)

        # --- CALCULATE METRICS ---
        f1s = calculate_f1_batch(preds, batch_targets)
        dists = calculate_chamfer_batch(preds, batch_targets)

        return f1s, dists, (num_windows - len(f1s))

    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        return [], [], 0


# ================= MAIN RUNNER =================
def run_full_test():
    print(f"🚀 Initializing Eval with Chamfer Distance on {CONFIG['device']}...")

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

    test_files = glob.glob(os.path.join(CONFIG["processed_dir"], "*.pt"))

    all_f1_scores = []
    all_dist_scores = []
    total_skipped = 0

    print(f"📊 Scanning {len(test_files)} files...")

    for filepath in tqdm(test_files):
        f1s, dists, skipped = process_file_windows(model, mel_layer, filepath)
        all_f1_scores.extend(f1s)
        all_dist_scores.extend(dists)
        total_skipped += skipped

    # --- REPORT & PLOT ---
    mean_f1 = np.mean(all_f1_scores) if all_f1_scores else 0
    mean_dist = np.mean(all_dist_scores) if all_dist_scores else 0

    print("\n" + "=" * 40)
    print(f"📈 RESULTS SUMMARY")
    print(f"   Mean F1 Score:       {mean_f1:.4f} (Higher is better)")
    print(f"   Mean Chamfer Dist:   {mean_dist:.4f} px (Lower is better)")
    print(f"   Windows Tested:      {len(all_f1_scores)}")
    print(f"   Silent Skipped:      {total_skipped}")
    print("=" * 40)

    # 1. Plot F1 Distribution
    if all_f1_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(
            all_f1_scores, bins=40, color="royalblue", alpha=0.8, edgecolor="black"
        )
        plt.axvline(
            mean_f1,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_f1:.3f}",
        )
        plt.title("F1 Score Distribution")
        plt.xlabel("F1 Score")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("distribution_f1.png")
        print("✨ Saved: distribution_f1.png")

    # 2. Plot Distance Distribution
    if all_dist_scores:
        # Outlier'ları grafiği bozmasın diye %95 percentile ile kırpabiliriz görsel amaçlı
        limit = (
            np.percentile(all_dist_scores, 95)
            if len(all_dist_scores) > 0
            else max(all_dist_scores)
        )

        plt.figure(figsize=(10, 6))
        plt.hist(
            all_dist_scores,
            bins=40,
            range=(0, limit),
            color="darkorange",
            alpha=0.8,
            edgecolor="black",
        )
        plt.axvline(
            mean_dist,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_dist:.2f} px",
        )
        plt.title("Chamfer (Move-Based) Distance Distribution")
        plt.xlabel("Average Pixel Shift Error")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("distribution_chamfer.png")
        print("✨ Saved: distribution_chamfer.png")


if __name__ == "__main__":
    run_full_test()
