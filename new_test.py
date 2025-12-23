import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import pretty_midi
import os
import yaml
import glob
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

# Custom Imports
from model import TranscriptionNet
import Data.dataset as dataset

# ================= CONFIG =================
CONFIG = {
    "raw_data_dir": "C:\\Users\\Jade\\Documents\\GPUvenv\\slakh2100_flac_redux",
    "root_dir": "processed_test_slakh", # Test .pt dosyalarının yeri
    "model_path": "model_piano.pth", 
    "save_prefix": "test_results_transformer",
    "target_class": "Piano",
    "sequence_length": 128,
    "batch_size": 32,
    "sample_rate": 16000,
    "hop_length": 512,
    "threshold": 0.35, 
    "num_workers": 6,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# ================= HELPER FUNCTIONS =================
def calculate_f1(preds, targets):
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1.item()

def preprocess_test_dataset():
    input_dir = os.path.join(CONFIG["raw_data_dir"], "test")
    output_dir = CONFIG["root_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tracks = sorted(glob.glob(os.path.join(input_dir, "Track*")))
    if not tracks:
        print(f"❌ HATA: {input_dir} içinde Track klasörü bulunamadı!")
        return

    print(f"🔄 Test Verisi İşleniyor: {len(tracks)} parça...")
    for track_path in tqdm(tracks):
        track_name = os.path.basename(track_path)
        save_path = os.path.join(output_dir, track_name + ".pt")
        if os.path.exists(save_path): continue

        mix_path = os.path.join(track_path, "mix.flac")
        meta_path = os.path.join(track_path, "metadata.yaml")
        if not os.path.exists(mix_path) or not os.path.exists(meta_path): continue

        try:
            waveform, sr = torchaudio.load(mix_path)
            if sr != CONFIG["sample_rate"]:
                waveform = torchaudio.transforms.Resample(sr, CONFIG["sample_rate"])(waveform)
            waveform = torch.mean(waveform, dim=0, keepdim=True)

            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)

            fs = CONFIG["sample_rate"] / CONFIG["hop_length"]
            total_frames = int(waveform.shape[1] / CONFIG["hop_length"])
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
        except Exception as e: print(f"Hata {track_name}: {e}")

def load_inference_model():
    model = TranscriptionNet().to(CONFIG["device"])
    checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"], weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def piano_roll_to_pretty_midi(piano_roll, fs):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")
    diff = np.diff(piano_roll).T
    note_on_time = np.zeros(88)
    for time, row in enumerate(diff):
        for idx in np.where(row != 0)[0]:
            if row[idx] > 0: note_on_time[idx] = time
            else:
                start, end = note_on_time[idx] / fs, time / fs
                if end > start: inst.notes.append(pretty_midi.Note(100, idx + 21, start, end))
    pm.instruments.append(inst)
    return pm

# ================= MAIN RUNNER =================
def run_full_test():
    # 1. Preprocess
    preprocess_test_dataset()

    # 2. Model & Data Setup
    model = load_inference_model()
    test_files = sorted(glob.glob(os.path.join(CONFIG["root_dir"], "*.pt")))
    test_dataset = dataset.SlakhChunkedDataset(CONFIG["root_dir"], test_files, CONFIG["sequence_length"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"], n_fft=2048, hop_length=CONFIG["hop_length"], n_mels=229
    ).to(CONFIG["device"])
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 3. Bulk Evaluation
    print("📊 Evaluating on whole test set...")
    total_loss, total_f1 = 0, 0
    with torch.no_grad():
        for waveform, target in tqdm(test_loader):
            waveform, target = waveform.to(CONFIG["device"]), target.to(CONFIG["device"])
            spec = torch.log(mel_layer(waveform) + 1e-5)
            spec = (spec - spec.mean()) / (spec.std() + 1e-5)
            
            with torch.amp.autocast(device_type='cuda'):
                preds = model(spec[..., :CONFIG["sequence_length"]])
                loss = criterion(preds, target.float())
            
            total_loss += loss.item()
            pred_bin = (torch.sigmoid(preds) > CONFIG["threshold"]).float()
            total_f1 += calculate_f1(pred_bin, target)

    print(f"📈 Test Results -> Loss: {total_loss/len(test_loader):.4f} | F1: {total_f1/len(test_loader):.4f}")

    # 4. Single Sample Inference (Visual & Audio)
    print("🎨 Generating Sample Outputs...")
    sample_path = random.choice(test_files)
    sample_data = torch.load(sample_path, weights_only=False)
    wave_sample = sample_data["waveform"][:, :CONFIG["sequence_length"]*CONFIG["hop_length"]].to(CONFIG["device"])
    total_samples = sample_data["waveform"].shape[1]
    total_frames = sample_data["target"].shape[-1]
    
    # Rastgele bir 4 saniyelik kesit başlangıcı seç
    if total_frames > CONFIG["sequence_length"]:
        start_frame = random.randint(0, total_frames - CONFIG["sequence_length"])
    else:
        start_frame = 0
    
    end_frame = start_frame + CONFIG["sequence_length"]
    start_sample = start_frame * CONFIG["hop_length"]
    end_sample = end_frame * CONFIG["hop_length"]

    # Kesitleri Kırp
    wave_sample = sample_data["waveform"][:, start_sample:end_sample].to(CONFIG["device"])
    target_sample = sample_data["target"][:, :, start_frame:end_frame].float().cpu().numpy()[0]
    
    with torch.no_grad():
        # [1, 1, 229, 128] formatına getir
        spec_s = torch.log(mel_layer(wave_sample.unsqueeze(0)) + 1e-5)
        spec_s = (spec_s - spec_s.mean()) / (spec_s.std() + 1e-5)
        
        # 4 saniyelik tahmin
        out_logits = model(spec_s[..., :CONFIG["sequence_length"]])
        out_preds = (torch.sigmoid(out_logits) > CONFIG["threshold"]).float()[0, 0].cpu().numpy()

    # Save MIDI & WAV
    fs = CONFIG["sample_rate"] / CONFIG["hop_length"]
    pm = piano_roll_to_pretty_midi(out_preds, fs)
    pm.write(f"{CONFIG['save_prefix']}.mid")
    
    # Sentezleme
    audio_synth = pm.synthesize(fs=16000)
    wavfile.write(f"{CONFIG['save_prefix']}.wav", 16000, (audio_synth * 32767).astype(np.int16))

    # Görselleştirme (Gerçek vs Tahmin)
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].imshow(target_sample, aspect='auto', origin='lower', cmap='gray_r')
    ax[0].set_title("Ground Truth (4 Seconds)")
    ax[1].imshow(out_preds, aspect='auto', origin='lower', cmap='magma')
    ax[1].set_title(f"Prediction (4 Seconds) - Threshold: {CONFIG['threshold']}")
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['save_prefix']}_plot.png")
    print(f"✨ Done! Random 4s slice from {os.path.basename(sample_path)} processed.")

if __name__ == "__main__":
    run_full_test()
