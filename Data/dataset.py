import os
import torch
import glob
from torch.utils.data import Dataset


class SlakhTranscriptionDataset(Dataset):
    def __init__(
        self, root_dir, split="train", target_class="All", sequence_length=128
    ):
        # İşlenmiş .pt dosyalarını bul
        self.file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.sequence_length = sequence_length
        self.hop_length = 512

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Hedeflenen Boyutlar
        req_frames = self.sequence_length
        req_samples = req_frames * self.hop_length

        try:
            # .pt dosyasını yükle
            data = torch.load(path)
            waveform = data["waveform"].float()
            target = data["target"].float()

            # --- KORUMA 1: BOŞ DOSYA KONTROLÜ ---
            # Eğer dosya boşsa [1, 0] ise hemen "Sessizlik" döndür
            if waveform.shape[-1] == 0 or target.shape[-1] == 0:
                return torch.zeros(1, req_samples), torch.zeros(1, 88, req_frames)

            curr_frames = target.shape[2]

            # --- DURUM A: Veri Uzunsa (KIRP) ---
            if curr_frames > req_frames:
                max_frame_start = curr_frames - req_frames
                # Rastgele bir başlangıç noktası seç
                start_frame = torch.randint(0, max_frame_start, (1,)).item()
                start_sample = start_frame * self.hop_length

                target = target[:, :, start_frame : start_frame + req_frames]
                waveform = waveform[:, start_sample : start_sample + req_samples]

            # --- DURUM B: Veri Kısaysa (DOLDUR/PAD) ---
            else:
                pad_frames = req_frames - curr_frames
                # Dikkat: Sample padding hesabı hassas yapılmalı
                pad_samples = req_samples - waveform.shape[1]

                if pad_frames > 0:
                    target = torch.nn.functional.pad(target, (0, pad_frames))
                if pad_samples > 0:
                    waveform = torch.nn.functional.pad(waveform, (0, pad_samples))

            # --- KORUMA 2: SON BOYUT KONTROLÜ ---
            # Hesaplama hatalarına karşı son bir zorlama yapıyoruz
            # Waveform [1, 65536] olmak ZORUNDA
            if waveform.shape[1] != req_samples:
                if waveform.shape[1] > req_samples:
                    waveform = waveform[:, :req_samples]
                else:
                    diff = req_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, diff))

            # Target [1, 88, 128] olmak ZORUNDA
            if target.shape[2] != req_frames:
                if target.shape[2] > req_frames:
                    target = target[:, :, :req_frames]
                else:
                    diff = req_frames - target.shape[2]
                    target = torch.nn.functional.pad(target, (0, diff))

            return waveform, target

        except Exception as e:
            print(f"⚠️ Hatalı dosya atlandı: {path} - Hata: {e}")
            # Hata durumunda eğitim durmasın diye boş veri döndür
            return torch.zeros(1, req_samples), torch.zeros(1, 88, req_frames)
