"""
ECG Dataset loader for diffusion model training
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Dict


class ECGDataset(Dataset):
    """
    ECG Dataset for conditional diffusion model
    
    Girdiler: amp_init, emb_init
    Çıktılar: freq_embeddings, images
    """
    
    def __init__(
        self,
        dataset_path: str = "dataset",
        split: str = "train",
        file_prefix: Optional[str] = None,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Args:
            dataset_path: Dataset klasör yolu
            split: "train" veya "val"
            file_prefix: Dosya ismi prefix'i (örn: "100_lead0_z0_w000")
            normalize: Veriyi normalize et
            augment: Data augmentation uygula
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.file_prefix = file_prefix
        self.normalize = normalize
        self.augment = augment
        
        # Klasör yolları
        self.amp_init_path = self.dataset_path / "inits" / "amp_init" / split
        self.emb_init_path = self.dataset_path / "inits" / "emb_init" / split
        self.freq_embeddings_path = self.dataset_path / "freq_embeddings" / split
        self.images_path = self.dataset_path / "images" / split
        
        # Dosya listesini oluştur
        self.file_list = self._get_file_list()
        
        # Normalizasyon parametrelerini hesapla
        if self.normalize:
            self.norm_params = self._compute_normalization_params()
        else:
            self.norm_params = None
    
    def _get_file_list(self):
        """Aynı isimdeki dosyaları bul ve eşleştir"""
        file_list = []
        
        # Tüm klasörlerdeki dosyaları topla
        all_files = {}
        
        for folder_name, folder_path in [
            ("amp_init", self.amp_init_path),
            ("emb_init", self.emb_init_path),
            ("freq_embeddings", self.freq_embeddings_path),
            ("images", self.images_path)
        ]:
            if folder_path.exists():
                for file_path in sorted(folder_path.glob("*.npy")):
                    filename = file_path.name
                    if self.file_prefix is None or filename.startswith(self.file_prefix):
                        if filename not in all_files:
                            all_files[filename] = {}
                        all_files[filename][folder_name] = file_path
        
        # Tüm klasörlerde mevcut olan dosyaları filtrele
        for filename, file_dict in sorted(all_files.items()):
            if len(file_dict) == 4:  # Tüm 4 dosya mevcut
                file_list.append({
                    "filename": filename,
                    "amp_init": file_dict["amp_init"],
                    "emb_init": file_dict["emb_init"],
                    "freq_embeddings": file_dict["freq_embeddings"],
                    "images": file_dict["images"]
                })
        
        if len(file_list) == 0:
            # Provide helpful error message
            error_msg = f"No matching files found for split '{self.split}'"
            if self.file_prefix:
                error_msg += f" with prefix '{self.file_prefix}'"
            error_msg += f"\nDataset path: {self.dataset_path}"
            error_msg += f"\nChecked paths:"
            error_msg += f"\n  - {self.amp_init_path} (exists: {self.amp_init_path.exists()})"
            error_msg += f"\n  - {self.emb_init_path} (exists: {self.emb_init_path.exists()})"
            error_msg += f"\n  - {self.freq_embeddings_path} (exists: {self.freq_embeddings_path.exists()})"
            error_msg += f"\n  - {self.images_path} (exists: {self.images_path.exists()})"
            
            # Count files in each directory
            if self.amp_init_path.exists():
                amp_files = list(self.amp_init_path.glob("*.npy"))
                error_msg += f"\n  - Found {len(amp_files)} files in amp_init"
            if self.emb_init_path.exists():
                emb_files = list(self.emb_init_path.glob("*.npy"))
                error_msg += f"\n  - Found {len(emb_files)} files in emb_init"
            if self.freq_embeddings_path.exists():
                freq_files = list(self.freq_embeddings_path.glob("*.npy"))
                error_msg += f"\n  - Found {len(freq_files)} files in freq_embeddings"
            if self.images_path.exists():
                img_files = list(self.images_path.glob("*.npy"))
                error_msg += f"\n  - Found {len(img_files)} files in images"
            
            raise ValueError(error_msg)
        
        return file_list
    
    def _compute_normalization_params(self):
        """Normalizasyon parametrelerini hesapla (tüm dataset üzerinden)"""
        if len(self.file_list) == 0:
            return None
        
        # Örnek dosyalardan istatistikleri hesapla
        sample_size = min(100, len(self.file_list))
        sample_indices = np.linspace(0, len(self.file_list) - 1, sample_size, dtype=int)
        
        stats = {
            "amp_init": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "emb_init": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "freq_embeddings": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "images": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        }
        
        all_data = {key: [] for key in stats.keys()}
        
        for idx in sample_indices:
            file_info = self.file_list[idx]
            for key in stats.keys():
                data = np.load(file_info[key])
                all_data[key].append(data)
        
        # İstatistikleri hesapla
        for key in stats.keys():
            if all_data[key]:
                all_values = np.concatenate([d.flatten() for d in all_data[key]])
                stats[key]["mean"] = float(np.mean(all_values))
                stats[key]["std"] = float(np.std(all_values))
                stats[key]["min"] = float(np.min(all_values))
                stats[key]["max"] = float(np.max(all_values))
        
        return stats
    
    def _normalize(self, data: np.ndarray, key: str) -> np.ndarray:
        """Veriyi normalize et"""
        if not self.normalize or self.norm_params is None:
            return data
        
        params = self.norm_params[key]
        mean = params["mean"]
        std = params["std"]
        
        if std > 1e-8:
            normalized = (data - mean) / std
        else:
            normalized = data - mean
        
        # [-1, 1] aralığına getir
        min_val = params["min"]
        max_val = params["max"]
        if max_val - min_val > 1e-8:
            normalized = 2 * (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized)) - 1
        
        return normalized
    
    def _denormalize(self, data: np.ndarray, key: str) -> np.ndarray:
        """Normalize edilmiş veriyi geri çevir"""
        if not self.normalize or self.norm_params is None:
            return data
        
        # [-1, 1]'den orijinal aralığa
        params = self.norm_params[key]
        min_val = params["min"]
        max_val = params["max"]
        
        if max_val - min_val > 1e-8:
            data = (data + 1) / 2 * (max_val - min_val) + min_val
        
        return data
    
    def _augment(self, *arrays):
        """Data augmentation uygula"""
        if not self.augment or self.split != "train":
            return arrays
        
        # Random horizontal flip (50% olasılık)
        if np.random.rand() > 0.5:
            # Use copy() to avoid negative stride issues
            arrays = tuple(np.flip(arr, axis=1).copy() for arr in arrays)
        
        return arrays
    
    def _load_and_process(self, file_path: Path, key: str) -> np.ndarray:
        """Dosyayı yükle ve işle"""
        data = np.load(file_path)
        
        # 2D'ye getir (gerekirse)
        if len(data.shape) == 1:
            # 1D veriyi 2D'ye reshape et
            data = data.reshape(-1, 1)
        elif len(data.shape) > 2:
            # İlk iki boyutu al
            data = data.reshape(data.shape[0], -1)
        
        # Normalize et
        data = self._normalize(data, key)
        
        return data
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Dataset'ten bir örnek getir"""
        file_info = self.file_list[idx]
        
        # Verileri yükle
        amp_init = self._load_and_process(file_info["amp_init"], "amp_init")
        emb_init = self._load_and_process(file_info["emb_init"], "emb_init")
        freq_embeddings = self._load_and_process(file_info["freq_embeddings"], "freq_embeddings")
        images = self._load_and_process(file_info["images"], "images")
        
        # Shape alignment - farklı boyutları uyumlu hale getir
        # En büyük boyutları al
        max_h = max(amp_init.shape[0], emb_init.shape[0], freq_embeddings.shape[0], images.shape[0])
        max_w = max(amp_init.shape[1], emb_init.shape[1], freq_embeddings.shape[1], images.shape[1])
        
        # Padding ile aynı boyuta getir
        def pad_to_size(arr, target_h, target_w):
            h, w = arr.shape
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            if pad_h > 0 or pad_w > 0:
                arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            # Ensure contiguous array to avoid negative stride issues
            return np.ascontiguousarray(arr[:target_h, :target_w])
        
        amp_init = pad_to_size(amp_init, max_h, max_w)
        emb_init = pad_to_size(emb_init, max_h, max_w)
        freq_embeddings = pad_to_size(freq_embeddings, max_h, max_w)
        images = pad_to_size(images, max_h, max_w)
        
        # Augmentation
        amp_init, emb_init, freq_embeddings, images = self._augment(
            amp_init, emb_init, freq_embeddings, images
        )
        
        # Ensure arrays are contiguous (fix negative stride issue)
        amp_init = np.ascontiguousarray(amp_init)
        emb_init = np.ascontiguousarray(emb_init)
        freq_embeddings = np.ascontiguousarray(freq_embeddings)
        images = np.ascontiguousarray(images)
        
        # Channel dimension ekle ve tensor'a çevir
        amp_init = torch.FloatTensor(amp_init).unsqueeze(0)  # (1, H, W)
        emb_init = torch.FloatTensor(emb_init).unsqueeze(0)  # (1, H, W)
        freq_embeddings = torch.FloatTensor(freq_embeddings).unsqueeze(0)  # (1, H, W)
        images = torch.FloatTensor(images).unsqueeze(0)  # (1, H, W)
        
        return {
            "amp_init": amp_init,
            "emb_init": emb_init,
            "freq_embeddings": freq_embeddings,
            "images": images,
            "filename": file_info["filename"]
        }
    
    def get_denormalized(self, data: torch.Tensor, key: str) -> np.ndarray:
        """Normalize edilmiş veriyi geri çevir (numpy array olarak)"""
        data_np = data.detach().cpu().numpy()
        if len(data_np.shape) == 3:  # (C, H, W)
            data_np = data_np[0]  # İlk channel'ı al
        
        return self._denormalize(data_np, key)

