import torch
import torchaudio
import numpy as np
import pretty_midi
from tqdm import tqdm

from xlstm import CNN_xLSTM_AMT

# =========================
# CONFIG (MATCH TRAINING)
# =========================
CONFIG = {
    "model_path": "model_xlstm_piano.pth",
    "sample_rate": 16000,
    "hop_length": 512,
    "sequence_length": 128,
    "n_mels": 229,
    "threshold": 0.3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "target_midi_pitch": 21,  # Piano lowest (A0)
    "num_pitches": 88,        # If later extended
}

# =========================
# LOAD MODEL
# =========================
def load_model():
    model = CNN_xLSTM_AMT(n_mels=CONFIG["n_mels"], d_model=512)
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.to(CONFIG["device"])
    model.eval()
    return model


# =========================
# AUDIO → MEL
# =========================
def audio_to_mel(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono

    if sr != CONFIG["sample_rate"]:
        waveform = torchaudio.functional.resample(
            waveform, sr, CONFIG["sample_rate"]
        )

    mel_layer = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_fft=2048,
        hop_length=CONFIG["hop_length"],
        n_mels=CONFIG["n_mels"]
    ).to(CONFIG["device"])

    with torch.no_grad():
        mel = mel_layer(waveform.to(CONFIG["device"]))
        mel = torch.log(mel + 1e-5)
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)

    return mel.squeeze(0)  # (n_mels, T)


# =========================
# SLIDING WINDOW INFERENCE
# =========================
def run_inference(model, mel):
    T = mel.shape[1]
    seq_len = CONFIG["sequence_length"]
    hop = seq_len // 2  # 50% overlap

    frame_probs = torch.zeros(T, device=CONFIG["device"])
    frame_counts = torch.zeros(T, device=CONFIG["device"])

    for start in tqdm(range(0, T - seq_len + 1, hop), desc="Inferencing"):
        chunk = mel[:, start:start + seq_len].unsqueeze(0)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            preds = model(chunk)
            probs = torch.sigmoid(preds).squeeze(0)

        frame_probs[start:start + seq_len] += probs
        frame_counts[start:start + seq_len] += 1

    frame_probs /= torch.clamp(frame_counts, min=1)
    return frame_probs.cpu().numpy()


# =========================
# FRAMES → MIDI
# =========================
def frames_to_midi(frame_probs, out_midi):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

    active = False
    onset = 0.0
    time_per_frame = CONFIG["hop_length"] / CONFIG["sample_rate"]

    for i, p in enumerate(frame_probs):
        t = i * time_per_frame

        if p >= CONFIG["threshold"] and not active:
            onset = t
            active = True

        elif p < CONFIG["threshold"] and active:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=90,
                    pitch=60,  # middle C (single-pitch version)
                    start=onset,
                    end=t
                )
            )
            active = False

    pm.instruments.append(instrument)
    pm.write(out_midi)


# =========================
# FRAMES → WAV MASK
# =========================
def frames_to_wav(audio_path, frame_probs, out_wav):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)

    if sr != CONFIG["sample_rate"]:
        waveform = torchaudio.functional.resample(waveform, sr, CONFIG["sample_rate"])

    mask = torch.zeros_like(waveform)
    samples_per_frame = CONFIG["hop_length"]

    for i, p in enumerate(frame_probs):
        if p >= CONFIG["threshold"]:
            start = i * samples_per_frame
            end = min(start + samples_per_frame, mask.shape[0])
            mask[start:end] = 1.0

    masked_audio = waveform * mask
    torchaudio.save(out_wav, masked_audio.unsqueeze(0), CONFIG["sample_rate"])


# =========================
# MAIN
# =========================
def transcribe(audio_path, out_midi, out_wav):
    model = load_model()
    mel = audio_to_mel(audio_path)
    frame_probs = run_inference(model, mel)

    frames_to_midi(frame_probs, out_midi)
    frames_to_wav(audio_path, frame_probs, out_wav)

    print("✅ Transcription complete")


if __name__ == "__main__":
    # pick random test audio from test set


    transcribe(
        audio_path="input_song.flac",
        out_midi="piano.mid",
        out_wav="piano.wav"
    )
