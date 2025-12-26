"""
Inference sonuçlarını analiz eden ve görselleştiren modül
analyze_dataset.py benzeri yapı ile girdi-çıktı karşılaştırması yapar
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict


class InferenceAnalyzer:
    """Inference sonuçlarını analiz eden ve görselleştiren sınıf"""
    
    def __init__(
        self,
        amp_init: np.ndarray,
        emb_init: np.ndarray,
        generated_freq: np.ndarray,
        generated_images: np.ndarray,
        target_freq: Optional[np.ndarray] = None,
        target_images: Optional[np.ndarray] = None,
        filenames: Optional[List[str]] = None
    ):
        """
        Args:
            amp_init: (N, H, W) veya (N, 1, H, W) amplitude initialization
            emb_init: (N, H, W) veya (N, 1, H, W) embedding initialization
            generated_freq: (N, H, W) veya (N, 1, H, W) generated frequency embeddings
            generated_images: (N, H, W) veya (N, 1, H, W) generated images
            target_freq: (N, H, W) veya (N, 1, H, W) target frequency embeddings (optional)
            target_images: (N, H, W) veya (N, 1, H, W) target images (optional)
            filenames: Dosya isimleri listesi (optional)
        """
        # Channel dimension'ı kaldır
        self.amp_init = self._remove_channel_dim(amp_init)
        self.emb_init = self._remove_channel_dim(emb_init)
        self.generated_freq = self._remove_channel_dim(generated_freq)
        self.generated_images = self._remove_channel_dim(generated_images)
        self.target_freq = self._remove_channel_dim(target_freq) if target_freq is not None else None
        self.target_images = self._remove_channel_dim(target_images) if target_images is not None else None
        
        self.N = self.generated_freq.shape[0]
        self.filenames = filenames if filenames else [f"sample_{i:04d}" for i in range(self.N)]
        
        # Verileri flip et (görselleştirme için - analyze_dataset.py ile uyumlu)
        if len(self.amp_init.shape) == 2:
            self.amp_init = np.flipud(self.amp_init)[np.newaxis, :, :]
        else:
            self.amp_init = np.array([np.flipud(x) for x in self.amp_init])
        
        if len(self.emb_init.shape) == 2:
            self.emb_init = np.flipud(self.emb_init)[np.newaxis, :, :]
        else:
            self.emb_init = np.array([np.flipud(x) for x in self.emb_init])
        
        self.generated_freq = np.array([np.flipud(x) for x in self.generated_freq])
        self.generated_images = np.array([np.flipud(x) for x in self.generated_images])
        if self.target_freq is not None:
            self.target_freq = np.array([np.flipud(x) for x in self.target_freq])
        if self.target_images is not None:
            self.target_images = np.array([np.flipud(x) for x in self.target_images])
    
    @staticmethod
    def _remove_channel_dim(data: np.ndarray) -> np.ndarray:
        """Channel dimension'ı kaldır"""
        if data is None:
            return None
        if len(data.shape) == 4:  # (N, 1, H, W)
            return data[:, 0, :, :]
        elif len(data.shape) == 3:  # (N, H, W)
            return data
        elif len(data.shape) == 2:  # (H, W) - tek örnek
            return data[np.newaxis, :, :]
        else:
            return data
    
    @staticmethod
    def calculate_metrics(data: np.ndarray) -> Dict[str, float]:
        """Bir veri dizisi için metrikleri hesaplar"""
        if data is None or data.size == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "zero_ratio": 0.0
            }
        
        metrics = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "zero_ratio": float(np.sum(data == 0) / data.size) if data.size > 0 else 0.0
        }
        return metrics
    
    @staticmethod
    def calculate_mse(data1: np.ndarray, data2: np.ndarray) -> Optional[float]:
        """İki veri dizisi arasındaki MSE'yi hesaplar"""
        if data1 is None or data2 is None:
            return None
        if data1.shape != data2.shape:
            return None
        return float(np.mean((data1 - data2) ** 2))
    
    def analyze_sample(self, idx: int) -> Dict:
        """Belirli bir örneği analiz eder"""
        if idx >= self.N:
            raise ValueError(f"Index {idx} out of range (N={self.N})")
        
        result = {
            "filename": self.filenames[idx],
            "index": idx,
            "metrics": {},
            "mse": {}
        }
        
        # Metrikleri hesapla
        result["metrics"]["amp_init"] = self.calculate_metrics(self.amp_init[idx])
        result["metrics"]["emb_init"] = self.calculate_metrics(self.emb_init[idx])
        result["metrics"]["generated_freq"] = self.calculate_metrics(self.generated_freq[idx])
        result["metrics"]["generated_images"] = self.calculate_metrics(self.generated_images[idx])
        
        if self.target_freq is not None:
            result["metrics"]["target_freq"] = self.calculate_metrics(self.target_freq[idx])
        if self.target_images is not None:
            result["metrics"]["target_images"] = self.calculate_metrics(self.target_images[idx])
        
        # MSE hesapla
        result["mse"]["generated_freq_vs_target"] = self.calculate_mse(
            self.generated_freq[idx],
            self.target_freq[idx] if self.target_freq is not None else None
        )
        result["mse"]["generated_images_vs_target"] = self.calculate_mse(
            self.generated_images[idx],
            self.target_images[idx] if self.target_images is not None else None
        )
        
        return result
    
    def print_analysis(self):
        """Tüm örnekler için detaylı analiz yazdırır"""
        print("=" * 120)
        print("INFERENCE SONUÇLARI ANALİZİ")
        print("=" * 120)
        
        for idx in range(self.N):
            result = self.analyze_sample(idx)
            filename = result["filename"]
            
            print(f"\n{'='*120}")
            print(f"ÖRNEK {idx+1}/{self.N}: {filename}")
            print(f"{'='*120}")
            
            # Metrikler tablosu
            print(f"\n{'Kategori':<25s} {'MEAN':>15s} {'STD':>15s} {'MIN':>15s} {'MAX':>15s} {'ZERO_RATIO':>15s}")
            print("-" * 120)
            
            categories = ["amp_init", "emb_init", "generated_freq", "generated_images"]
            if self.target_freq is not None:
                categories.append("target_freq")
            if self.target_images is not None:
                categories.append("target_images")
            
            for cat in categories:
                if cat in result["metrics"]:
                    m = result["metrics"][cat]
                    print(f"{cat:<25s} {m['mean']:>15.6f} {m['std']:>15.6f} {m['min']:>15.6f} {m['max']:>15.6f} {m['zero_ratio']:>15.6f}")
            
            # MSE karşılaştırmaları
            if self.target_freq is not None or self.target_images is not None:
                print(f"\nMSE Karşılaştırmaları:")
                print("-" * 120)
                
                if result["mse"]["generated_freq_vs_target"] is not None:
                    print(f"  Generated Freq vs Target Freq:     MSE = {result['mse']['generated_freq_vs_target']:.6f}")
                
                if result["mse"]["generated_images_vs_target"] is not None:
                    print(f"  Generated Images vs Target Images: MSE = {result['mse']['generated_images_vs_target']:.6f}")
            
            # Girdi-çıktı karşılaştırmaları
            print(f"\nGirdi-Çıktı Karşılaştırmaları:")
            print("-" * 120)
            
            mse_amp_freq = self.calculate_mse(self.amp_init[idx], self.generated_freq[idx])
            mse_emb_freq = self.calculate_mse(self.emb_init[idx], self.generated_freq[idx])
            mse_amp_images = self.calculate_mse(self.amp_init[idx], self.generated_images[idx])
            mse_emb_images = self.calculate_mse(self.emb_init[idx], self.generated_images[idx])
            
            if mse_amp_freq is not None:
                print(f"  amp_init vs generated_freq:      MSE = {mse_amp_freq:.6f}")
            if mse_emb_freq is not None:
                print(f"  emb_init vs generated_freq:       MSE = {mse_emb_freq:.6f}")
            if mse_amp_images is not None:
                print(f"  amp_init vs generated_images:    MSE = {mse_amp_images:.6f}")
            if mse_emb_images is not None:
                print(f"  emb_init vs generated_images:    MSE = {mse_emb_images:.6f}")
        
        print("\n" + "=" * 120)
    
    def visualize_comparison(
        self,
        save_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        figsize: Tuple[int, int] = (20, 12)
    ):
        """Girdi-çıktı karşılaştırmasını görselleştirir"""
        num_samples = num_samples if num_samples else self.N
        num_samples = min(num_samples, self.N)
        
        # Sütun sayısını belirle
        has_target = self.target_freq is not None and self.target_images is not None
        num_cols = 6 if has_target else 4  # amp_init, emb_init, generated_freq, generated_images, [target_freq, target_images]
        
        fig, axes = plt.subplots(num_samples, num_cols, figsize=figsize)
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_samples):
            row_axes = axes[idx] if num_samples > 1 else axes
            
            col = 0
            
            # amp_init
            ax = row_axes[col]
            im = ax.imshow(self.amp_init[idx], cmap='hot', aspect='auto')
            ax.set_title(f"amp_init\n{self.filenames[idx]}", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
            col += 1
            
            # emb_init
            ax = row_axes[col]
            im = ax.imshow(self.emb_init[idx], cmap='hot', aspect='auto')
            ax.set_title(f"emb_init", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
            col += 1
            
            # generated_freq
            ax = row_axes[col]
            im = ax.imshow(self.generated_freq[idx], cmap='seismic', aspect='auto')
            ax.set_title(f"Generated\nfreq_embeddings", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
            col += 1
            
            # generated_images
            ax = row_axes[col]
            im = ax.imshow(self.generated_images[idx], cmap='hot', aspect='auto')
            ax.set_title(f"Generated\nimages", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
            col += 1
            
            # target_freq (varsa)
            if has_target:
                ax = row_axes[col]
                im = ax.imshow(self.target_freq[idx], cmap='seismic', aspect='auto')
                ax.set_title(f"Target\nfreq_embeddings", fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.axis('off')
                col += 1
                
                # target_images (varsa)
                ax = row_axes[col]
                im = ax.imshow(self.target_images[idx], cmap='hot', aspect='auto')
                ax.set_title(f"Target\nimages", fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Görselleştirme kaydedildi: {save_path}")
        
        plt.show()
    
    def visualize_detailed(
        self,
        idx: int,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 10)
    ):
        """Belirli bir örnek için detaylı görselleştirme"""
        if idx >= self.N:
            raise ValueError(f"Index {idx} out of range (N={self.N})")
        
        has_target = self.target_freq is not None and self.target_images is not None
        num_rows = 2
        num_cols = 4 if has_target else 2
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # İlk satır: Frequency embeddings
        row = 0
        col = 0
        
        # amp_init
        ax = axes[row, col]
        im = ax.imshow(self.amp_init[idx], cmap='hot', aspect='auto')
        ax.set_title(f"amp_init\n{self.filenames[idx]}", fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
        col += 1
        
        # generated_freq
        ax = axes[row, col]
        im = ax.imshow(self.generated_freq[idx], cmap='seismic', aspect='auto')
        ax.set_title("Generated freq_embeddings", fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
        col += 1
        
        # target_freq (varsa)
        if has_target:
            ax = axes[row, col]
            im = ax.imshow(self.target_freq[idx], cmap='seismic', aspect='auto')
            ax.set_title("Target freq_embeddings", fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
            col += 1
            
            # Fark (generated - target)
            ax = axes[row, col]
            diff = self.generated_freq[idx] - self.target_freq[idx]
            im = ax.imshow(diff, cmap='seismic', aspect='auto')
            mse = self.calculate_mse(self.generated_freq[idx], self.target_freq[idx])
            ax.set_title(f"Difference (MSE: {mse:.6f})", fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
        
        # İkinci satır: Images
        row = 1
        col = 0
        
        # emb_init
        ax = axes[row, col]
        im = ax.imshow(self.emb_init[idx], cmap='hot', aspect='auto')
        ax.set_title("emb_init", fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
        col += 1
        
        # generated_images
        ax = axes[row, col]
        im = ax.imshow(self.generated_images[idx], cmap='hot', aspect='auto')
        ax.set_title("Generated images", fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
        col += 1
        
        # target_images (varsa)
        if has_target:
            ax = axes[row, col]
            im = ax.imshow(self.target_images[idx], cmap='hot', aspect='auto')
            ax.set_title("Target images", fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
            col += 1
            
            # Fark (generated - target)
            ax = axes[row, col]
            diff = self.generated_images[idx] - self.target_images[idx]
            im = ax.imshow(diff, cmap='seismic', aspect='auto')
            mse = self.calculate_mse(self.generated_images[idx], self.target_images[idx])
            ax.set_title(f"Difference (MSE: {mse:.6f})", fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
        
        plt.suptitle(f"Detaylı Analiz: {self.filenames[idx]}", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detaylı görselleştirme kaydedildi: {save_path}")
        
        plt.show()
    
    def run_full_analysis(
        self,
        save_dir: Optional[str] = None,
        visualize: bool = True,
        detailed: bool = True
    ):
        """Tam analiz çalıştırır"""
        print("\n" + "=" * 120)
        print("INFERENCE SONUÇLARI TAM ANALİZİ")
        print("=" * 120)
        
        # Metrik analizi
        self.print_analysis()
        
        # Görselleştirme
        if visualize:
            save_path_comparison = None
            save_path_detailed = None
            
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path_comparison = str(save_dir / "inference_comparison.png")
                if detailed:
                    save_path_detailed = str(save_dir / "inference_detailed.png")
            
            print("\nGörselleştirme oluşturuluyor...")
            self.visualize_comparison(save_path=save_path_comparison)
            
            if detailed:
                print("\nDetaylı görselleştirme oluşturuluyor...")
                # İlk örnek için detaylı görselleştirme
                self.visualize_detailed(0, save_path=save_path_detailed)
        
        print("\n" + "=" * 120)
        print("Analiz tamamlandı!")
        print("=" * 120)

