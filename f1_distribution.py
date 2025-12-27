"""
TEST SCRIPT FOR TRANSCRIPTION
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
    "model_path": "model_guitar.pth",
    "processed_dir": "processed_guitar_test_slakh",
    "sequence_length": 128,  # approx 4 seconds
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.35,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# ================= HELPER FUNCTIONS =================
def calculate_metrics_batch(preds, targets):
    """
    Calculates F1 for a BATCH of windows.
    """
    # Flatten batch dimensions for global metrics or keep per-sample? 
    # Let's calculate per-sample F1 to generate the distribution plot.
    
    # preds: [Batch, 88, Time]
    # targets: [Batch, 88, Time]
    
    batch_size = preds.shape[0]
    f1_scores = []
    
    for i in range(batch_size):
        p = preds[i].float().reshape(-1)
        t = targets[i].float().reshape(-1)
        
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1.item())
        
    return f1_scores

def process_file_windows(model, mel_layer, filepath):
    """
    Slices the file into windows, filters silent ones, and runs batch inference.
    Returns a list of F1 scores for all valid windows in this file.
    """
    try:
        data = torch.load(filepath, map_location=CONFIG["device"], weights_only=False)
        waveform = data["waveform"].to(CONFIG["device"]) # [1, Samples]
        target = data["target"].to(CONFIG["device"])     # [1, 88, Frames]
        
        # Squeeze batch dim for slicing
        waveform = waveform.squeeze(0) 
        target = target.squeeze(0)     

        seq_len = CONFIG["sequence_length"]
        hop_samples = seq_len * CONFIG["hop_length"]
        total_frames = target.shape[1]
        
        # Calculate how many full windows we can fit
        num_windows = total_frames // seq_len
        
        if num_windows == 0:
            return [], 0 # File too short
            
        valid_waveforms = []
        valid_targets = []
        
        # --- SLICING LOOP ---
        for i in range(num_windows):
            start_frame = i * seq_len
            end_frame = start_frame + seq_len
            
            # Slice Target
            t_window = target[:, start_frame:end_frame]
            
            # Check for Silence (Filter)
            if t_window.sum() == 0:
                continue # Skip this window
                
            # Slice Waveform
            start_sample = start_frame * CONFIG["hop_length"]
            end_sample = end_frame * CONFIG["hop_length"]
            w_window = waveform[start_sample:end_sample]
            
            valid_targets.append(t_window)
            valid_waveforms.append(w_window)
            
        if not valid_targets:
            return [], num_windows # All windows were silent
            
        # Stack into a batch
        # waveform batch: [Batch, Samples]
        # target batch:   [Batch, 88, Frames]
        batch_waveforms = torch.stack(valid_waveforms)
        batch_targets = torch.stack(valid_targets)
        
        # --- BATCH INFERENCE ---
        with torch.no_grad():
            spec = mel_layer(batch_waveforms)
            spec = torch.log(spec + 1e-5)
            
            # Normalize
            mean = spec.mean(dim=(1, 2), keepdim=True)
            std = spec.std(dim=(1, 2), keepdim=True)
            spec = (spec - mean) / (std + 1e-8)
            
            # Fix dims: [Batch, Channels, Freq, Time]
            if spec.shape[-1] > seq_len:
                spec = spec[..., :seq_len]
            
            # Fix dims: [Batch, Channels, Freq, Time]
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)
                
            logits = model(spec)
            
            preds = (torch.sigmoid(logits) > CONFIG["threshold"]).float()
            
            # Ensure shape [Batch, 88, Time]
            if preds.shape != batch_targets.shape:
                preds = preds.transpose(1, 2)
                
        # Calculate F1s for this batch
        f1s = calculate_metrics_batch(preds, batch_targets)
        
        return f1s, (num_windows - len(f1s)) # Returns scores and count of silent windows dropped
        
    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        return [], 0

# ================= MAIN RUNNER =================
def run_full_test():
    print(f"🚀 Initializing Full Dataset Evaluation on {CONFIG['device']}...")

    model = TranscriptionNet().to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.eval()

    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"], n_fft=2048, hop_length=CONFIG["hop_length"], n_mels=229
    ).to(CONFIG["device"])

    test_files = glob.glob(os.path.join(CONFIG["processed_dir"], "*.pt"))
    if not test_files:
        print("❌ No test files found.")
        return

    all_active_f1_scores = []
    total_windows_processed = 0
    total_silent_windows_skipped = 0

    print(f"📊 Scanning {len(test_files)} files (Windowing & Filtering)...")
    
    for filepath in tqdm(test_files):
        f1s, skipped = process_file_windows(model, mel_layer, filepath)
        
        all_active_f1_scores.extend(f1s)
        total_silent_windows_skipped += skipped
        total_windows_processed += (len(f1s) + skipped)

    # REPORTING
    mean_active_f1 = np.mean(all_active_f1_scores) if all_active_f1_scores else 0
    
    print("\n" + "="*40)
    print(f"📈 MEAN ACTIVE F1: {mean_active_f1:.4f}")
    print(f"🪟 Total Windows Checked: {total_windows_processed}")
    print(f"✅ Active Windows Tested: {len(all_active_f1_scores)}")
    print(f"🔇 Silent Windows Skipped: {total_silent_windows_skipped}")
    print("="*40)

    # PLOT
    if all_active_f1_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(all_active_f1_scores, bins=30, color='royalblue', alpha=0.8, edgecolor='black')
        plt.axvline(mean_active_f1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_active_f1:.3f}')
        plt.title('F1 Score Distribution (All Active 4s Windows)')
        plt.xlabel('F1 Score')
        plt.ylabel('Window Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('full_dataset_f1_distribution.png')
        print("✨ Plot saved to: full_dataset_f1_distribution.png")

if __name__ == "__main__":
    run_full_test()
