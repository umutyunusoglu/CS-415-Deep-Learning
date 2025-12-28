import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import scipy.ndimage
from tqdm import tqdm

# Custom Imports (model.py dosyanın yanında olmalı)
from model import TranscriptionNet

# ================= STATIC CONFIG =================
# Değişmeyen hiperparametreler
CONFIG = {
    "sequence_length": 128,  # approx 4 seconds
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.5,
    "n_mels": 229,
}

# ================= METRIC FUNCTIONS =================


def calculate_f1_batch(preds, targets):
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
    batch_size = preds.shape[0]
    distances = []
    preds = preds.squeeze(1)
    targets = targets.squeeze(1)

    for b in range(batch_size):
        p_coords = torch.nonzero(preds[b]).float()
        t_coords = torch.nonzero(targets[b]).float()

        # Case 1: Both empty (Silence predicted correctly)
        if p_coords.size(0) == 0 and t_coords.size(0) == 0:
            distances.append(0.0)
            continue
        # Case 2: One empty (Total miss)
        if p_coords.size(0) == 0 or t_coords.size(0) == 0:
            distances.append(CONFIG["sequence_length"] / 2.0)  # Penalty
            continue

        # Case 3: Pairwise Distance
        dists = torch.cdist(p_coords, t_coords, p=2)
        min_dist_p2t, _ = torch.min(dists, dim=1)
        min_dist_t2p, _ = torch.min(dists, dim=0)
        chamfer_dist = torch.mean(min_dist_p2t) + torch.mean(min_dist_t2p)
        distances.append(chamfer_dist.item())

    return distances


def get_displacement_vectors(preds, targets):
    """Calculates vector difference (dt, dp) for Error Density Plot"""
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

        # diffs[:, 0] -> Pitch (Y), diffs[:, 1] -> Time (X)
        diffs = p_coords - closest_targets
        batch_dp.extend(diffs[:, 0].cpu().numpy().tolist())
        batch_dt.extend(diffs[:, 1].cpu().numpy().tolist())

    return batch_dt, batch_dp


# ================= PROCESS LOOP =================


def process_file(model, mel_layer, filepath, device):
    """Runs inference ONCE and returns all metrics together."""
    try:
        data = torch.load(filepath, map_location=device, weights_only=False)
        waveform = data["waveform"].to(device).squeeze(0)
        target = data["target"].to(device).squeeze(0)

        seq_len = CONFIG["sequence_length"]
        total_frames = target.shape[1]
        num_windows = total_frames // seq_len

        if num_windows == 0:
            return [], [], [], []

        valid_waveforms = []
        valid_targets = []

        # --- SLICING & FILTERING ---
        for i in range(num_windows):
            start_frame = i * seq_len
            end_frame = start_frame + seq_len
            t_window = target[:, start_frame:end_frame]

            if t_window.sum() == 0:
                continue  # Skip silence

            start_sample = start_frame * CONFIG["hop_length"]
            end_sample = end_frame * CONFIG["hop_length"]
            w_window = waveform[start_sample:end_sample]

            valid_targets.append(t_window)
            valid_waveforms.append(w_window)

        if not valid_targets:
            return [], [], [], []

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

        # --- CALC METRICS ---
        f1s = calculate_f1_batch(preds, batch_targets)
        dists = calculate_chamfer_batch(preds, batch_targets)
        dt, dp = get_displacement_vectors(preds, batch_targets)

        return f1s, dists, dt, dp

    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        return [], [], [], []


# ================= PLOTTING =================


def plot_combined_results(
    f1_scores, chamfer_scores, time_errors, pitch_errors, instrument_name
):
    """Generates 3 side-by-side plots."""

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3)

    # --- PLOT 1: F1 Distribution ---
    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    axs[0].hist(f1_scores, bins=40, color="royalblue", alpha=0.8, edgecolor="black")
    axs[0].axvline(
        mean_f1, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_f1:.3f}"
    )
    axs[0].set_title(f"{instrument_name}: F1 Score Distribution")
    axs[0].set_xlabel("F1 Score")
    axs[0].set_ylabel("Count")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # --- PLOT 2: Chamfer Distance ---
    mean_dist = np.mean(chamfer_scores) if chamfer_scores else 0
    limit = np.percentile(chamfer_scores, 95) if chamfer_scores else max(chamfer_scores)
    axs[1].hist(
        chamfer_scores,
        bins=40,
        range=(0, limit),
        color="darkorange",
        alpha=0.8,
        edgecolor="black",
    )
    axs[1].axvline(
        mean_dist,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_dist:.2f} px",
    )
    axs[1].set_title(f"{instrument_name}: Chamfer Distance")
    axs[1].set_xlabel("Average Pixel Shift Error")
    axs[1].set_ylabel("Count")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # --- PLOT 3: Error Density Contour ---
    if len(time_errors) > 0:
        x, y = np.array(time_errors), np.array(pitch_errors)
        xlim, ylim, bins = 8, 8, 100

        # 2D Histogram + Gaussian Smooth (Fast Contour)
        H, xedges, yedges = np.histogram2d(
            x, y, bins=bins, range=[[-xlim, xlim], [-ylim, ylim]]
        )
        H_smooth = scipy.ndimage.gaussian_filter(H.T, sigma=1.5)

        xc = (xedges[:-1] + xedges[1:]) / 2
        yc = (yedges[:-1] + yedges[1:]) / 2
        xx, yy = np.meshgrid(xc, yc)

        axs[2].set_facecolor("#121212")
        cf = axs[2].contourf(xx, yy, H_smooth, levels=60, cmap="inferno")
        axs[2].contour(
            xx, yy, H_smooth, levels=15, colors="white", linewidths=0.5, alpha=0.3
        )

        cbar = fig.colorbar(cf, ax=axs[2], fraction=0.046, pad=0.04)
        cbar.set_label("Error Density", rotation=270, labelpad=15)

        ms_per_frame = (CONFIG["hop_length"] / CONFIG["sample_rate"]) * 1000
        axs[2].set_title(f"{instrument_name}: Error Density Landscape")
        axs[2].set_xlabel(f"Time Shift (Frames)\n1 Frame ≈ {ms_per_frame:.1f}ms")
        axs[2].set_ylabel("Pitch Shift (Semitone)")
        axs[2].axhline(0, color="white", linestyle=":", alpha=0.4)
        axs[2].axvline(0, color="white", linestyle=":", alpha=0.4)
        axs[2].set_xlim(-2, 2)
        axs[2].set_ylim(-2, 2)
    else:
        axs[2].text(0.5, 0.5, "No Error Data", ha="center", va="center")

    save_path = f"evaluation_results_{instrument_name.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✨ Plot saved to: {save_path}")


# ================= MAIN RUNNER FUNCTION =================


def run_evaluation(model_path, processed_dir, instrument_name="Piano"):
    """
    Main entry point for evaluation.
    Args:
        model_path (str): Path to .pth model file.
        processed_dir (str): Folder containing .pt test files.
        instrument_name (str): Instrument name for plots (e.g. "Piano", "Guitar").
    """

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Eval for {instrument_name} on {device}")
    print(f"   Model: {model_path}")
    print(f"   Data:  {processed_dir}")

    # Load Model
    model = TranscriptionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Mel Transform
    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=CONFIG["n_mels"],
    ).to(device)

    # Load Files
    test_files = glob.glob(os.path.join(processed_dir, "*.pt"))
    print(f"📊 Processing {len(test_files)} files...")

    all_f1, all_chamfer, all_dt, all_dp = [], [], [], []

    for filepath in tqdm(test_files):
        f1s, dists, dt, dp = process_file(model, mel_layer, filepath, device)
        all_f1.extend(f1s)
        all_chamfer.extend(dists)
        all_dt.extend(dt)
        all_dp.extend(dp)

    # Print Summary
    mean_f1 = np.mean(all_f1) if all_f1 else 0
    mean_dist = np.mean(all_chamfer) if all_chamfer else 0
    print("\n" + "=" * 40)
    print(f"📈 RESULTS SUMMARY: {instrument_name}")
    print(f"   Mean F1 Score:       {mean_f1:.4f}")
    print(f"   Mean Chamfer Dist:   {mean_dist:.4f}")
    print(f"   Total Windows:       {len(all_f1)}")
    print(f"   Total Error Points:  {len(all_dt)}")
    print("=" * 40)

    # Plot
    plot_combined_results(all_f1, all_chamfer, all_dt, all_dp, instrument_name)


if __name__ == "__main__":
    # Parametreleri BURADAN değiştir:
    run_evaluation(
        model_path="D:\Ana\Projeler\CS 415\piyano transformer\model_piano.pth",
        processed_dir="D:\Ana\Projeler\CS 415\processed_test_slakh",
        instrument_name="Piano",
    )
