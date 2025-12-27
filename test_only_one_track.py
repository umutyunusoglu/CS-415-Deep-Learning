"""
TEST SCRIPT FOR STRINGS TRANSCRIPTION
Tests on preprocessed test data
"""
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import pretty_midi
import os
import glob
import random
from model import TranscriptionNet

# ================= CONFIG =================
CONFIG = {
    "model_path": "model_strings.pth",
    "test_file_path": None,  # None = random selection from test
    "processed_dir": r"C:\Users\Lenovo\Downloads\strings_processed\test",  # Test folder
    "sequence_length": 128,
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def save_audio(filename, audio_data, sr):
    audio_int16 = (audio_data / (np.abs(audio_data).max() + 1e-8) * 32767).astype(np.int16)
    wavfile.write(filename, sr, audio_int16)


def piano_roll_to_pretty_midi(piano_roll, fs):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=40)  # 40 = Violin/Strings
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


def calculate_metrics(preds, targets):
    """Calculate both F1 and Accuracy"""
    # Convert targets to float (they're saved as bool)
    targets = targets.float()
    
    tp = (preds * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    # Accuracy = (TP + TN) / Total
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    
    # F1 Score
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return accuracy.item(), precision.item(), recall.item(), f1.item()


def run_test():
    print(f"🧪 Test Mode Starting... ({CONFIG['device']})")
    print(f"🎻 Testing Strings model on TEST data\n")

    # 1. Load Model
    model = TranscriptionNet().to(CONFIG["device"])
    try:
        model.load_state_dict(
            torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
        )
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

    # 2. Load TEST Data
    if CONFIG["test_file_path"]:
        data = torch.load(CONFIG["test_file_path"])
        selected_file = CONFIG["test_file_path"]
        print(f"📂 Using specified file")
    else:
        files = glob.glob(os.path.join(CONFIG["processed_dir"], "*.pt"))
        
        if not files:
            print(f"❌ No test data found in: {CONFIG['processed_dir']}")
            print("   Run preprocess_test.py first!")
            return

        print(f"📂 Found {len(files)} test files")
        
        # Find file with strings
        data = None
        selected_file = None
        for _ in range(50):
            f = random.choice(files)
            temp_data = torch.load(f)
            if temp_data["target"].sum() > 0:
                data = temp_data
                selected_file = f  # Save the filepath
                track_name = os.path.basename(f).replace('.pt', '')
                print(f"📂 Selected: {track_name}\n")
                break
        
        if data is None:
            print("⚠️ No files with strings found")
            return

    waveform = data["waveform"].to(CONFIG["device"])
    target = data["target"].to(CONFIG["device"])
    original_waveform = waveform.clone()
    # Random chunk
    if target.shape[-1] > CONFIG["sequence_length"]:
        start_frame = np.random.randint(0, target.shape[-1] - CONFIG["sequence_length"])
        end_frame = start_frame + CONFIG["sequence_length"]
        start_sample = start_frame * CONFIG["hop_length"]
        end_sample = end_frame * CONFIG["hop_length"]

        waveform = waveform[:, start_sample:end_sample]
        target = target[:, :, start_frame:end_frame]
    
    # Ensure spec matches target length after mel transform
    print(f"   Waveform shape: {waveform.shape}")
    print(f"   Target shape: {target.shape}")

    # 3. Model Prediction
    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=229,
    ).to(CONFIG["device"])

    with torch.no_grad():
        spec = mel_layer(waveform)
        spec = torch.log(spec + 1e-5)
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True)
        spec = (spec - mean) / (std + 1e-8)
        
        # Add batch dimension if needed
        if spec.dim() == 3:
            spec = spec.unsqueeze(0)  # [C, F, T] -> [1, C, F, T]
        if spec.shape[-1] > CONFIG["sequence_length"]:
            spec = spec[..., :CONFIG["sequence_length"]]

        logits = model(spec)
        probs = torch.sigmoid(logits)
        preds = (probs > CONFIG["threshold"]).float()

    # 4. Calculate Metrics
    accuracy, precision, recall, f1 = calculate_metrics(preds[0], target[0])
    
    print(f"📊 Metrics:")
    print(f"   Accuracy (TP+TN/Total): {accuracy:.4f}")
    print(f"   Precision:              {precision:.4f}")
    print(f"   Recall:                 {recall:.4f}")
    print(f"   F1 Score:               {f1:.4f}")
    print(f"   True Notes:             {target[0].sum().item():.0f}")
    print(f"   Predicted Notes:        {preds[0].sum().item():.0f}\n")

    # 5. Visualization
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    ax[0].imshow(spec[0, 0].cpu(), aspect="auto", origin="lower", cmap="inferno")
    ax[0].set_title("Input Spectrogram (Mix Audio)")
    ax[1].imshow(target[0].cpu(), aspect="auto", origin="lower", cmap="gray_r")
    ax[1].set_title(f"Ground Truth (Strings) - {target[0].sum().item():.0f} notes")
    ax[2].imshow(preds[0, 0].cpu(), aspect="auto", origin="lower", cmap="magma")  # preds[0, 0]
    ax[2].set_title(f"Prediction - Acc: {accuracy:.3f}, F1: {f1:.3f}")
    plt.tight_layout()
    plt.savefig("test_result_strings.png")
    print("🖼️ Visualization saved: test_result_strings.png")

    # 6. Audio Synthesis
    fs = CONFIG["sample_rate"] / CONFIG["hop_length"]
    # Save original input audio (the chunk we tested on)
    try:
        waveform_np = waveform[0].cpu().numpy()
        save_audio("test_input_original.wav", waveform_np, CONFIG["sample_rate"])
        print("🎧 Original input chunk saved: test_input_original.wav")
    except Exception as e:
        print(f"⚠️ Could not save input audio: {e}")
    try:
        audio_pred = piano_roll_to_pretty_midi(preds[0, 0].cpu().numpy(), fs).synthesize(fs=16000)  # preds[0, 0]
        save_audio("test_prediction_strings.wav", audio_pred, 16000)
        print("🎧 Audio saved: test_prediction_strings.wav")
    except Exception as e:
        print(f"⚠️ Could not synthesize: {e}")
    # Save ground truth MIDI as audio
    try:
        audio_gt = piano_roll_to_pretty_midi(target[0].cpu().numpy(), fs).synthesize(fs=16000)
        save_audio("test_groundtruth_strings.wav", audio_gt, 16000)
        print("🎧 Ground truth saved: test_groundtruth_strings.wav")
    except Exception as e:
        print(f"⚠️ Could not synthesize ground truth: {e}")
    
    # Print file info for reference
    if selected_file:
        print(f"\n📁 Tested file: {os.path.basename(selected_file)}")
        print(f"   Full path: {selected_file}")


if __name__ == "__main__":
    run_test()