"""
Dataset klasörünü analiz eden ve images, inits, freq_embeddings klasörlerindeki görüntüleri gösteren script
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


class DatasetVisualizer:
    """Dataset analizi ve görselleştirme sınıfı"""
    
    def __init__(self, file_name, dataset_path="dataset"):
        """
        Args:
            file_name: Analiz edilecek dosya ismi (prefix)
            dataset_path: Dataset klasör yolu
        """
        self.file_name = file_name
        self.dataset_path = Path(dataset_path)
        self.folders = [
            ("images", self.dataset_path / "images"),
            ("freq_embeddings", self.dataset_path / "freq_embeddings"),
            ("amp_init", self.dataset_path / "inits" / "amp_init"),
            ("emb_init", self.dataset_path / "inits" / "emb_init")
        ]
    
    def analyze_dataset(self):
        """Dataset klasörünü analiz eder ve istatistikler verir"""
        print("=" * 60)
        print("DATASET ANALIZI")
        print("=" * 60)
        
        stats = defaultdict(lambda: {"train": 0, "val": 0})
        
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.npy'):
                    rel_path = Path(root).relative_to(self.dataset_path)
                    parts = rel_path.parts
                    
                    if len(parts) >= 2:
                        category = parts[0]
                        split = parts[-1]
                        
                        if split in ["train", "val"]:
                            stats[category][split] += 1
        
        print("\nDosya Istatistikleri:")
        print("-" * 60)
        for category, splits in sorted(stats.items()):
            total = splits["train"] + splits["val"]
            print(f"\n{category}:")
            print(f"  Train: {splits['train']} dosya")
            print(f"  Val:   {splits['val']} dosya")
            print(f"  Toplam: {total} dosya")
        
        return stats
    
    def analyze_folder(self, folder_path, folder_name, num_samples=5):
        """Belirli bir klasördeki dosyaları analiz eder"""
        folder_path = Path(folder_path)
        
        print("\n" + "=" * 60)
        print(f"{folder_name.upper()} KLASORU ANALIZI")
        print("=" * 60)
        
        train_files_all = sorted(list((folder_path / "train").glob("*.npy"))) if (folder_path / "train").exists() else []
        val_files_all = sorted(list((folder_path / "val").glob("*.npy"))) if (folder_path / "val").exists() else []
        
        train_files = [f for f in train_files_all if f.name.startswith(self.file_name)]
        val_files = [f for f in val_files_all if f.name.startswith(self.file_name)]
        
        print(f"\nTrain klasorunde {len(train_files_all)} dosya bulundu ({self.file_name}: {len(train_files)})")
        print(f"Val klasorunde {len(val_files_all)} dosya bulundu ({self.file_name}: {len(val_files)})")
        
        if len(train_files) == 0 and len(val_files) == 0:
            print(f"UYARI: {self.file_name} ile baslayan .npy dosyasi bulunamadi!")
            return None
        
        all_files = train_files[:num_samples] + val_files[:min(num_samples, len(val_files))]
        
        print(f"\nIlk {len(all_files)} dosya analiz ediliyor...")
        print("-" * 60)
        
        image_data = []
        
        for file_path in all_files:
            try:
                data = np.load(file_path)
                data = np.flipud(data).copy()
                split = "train" if "train" in str(file_path) else "val"
                
                print(f"\n{file_path.name} ({split}):")
                print(f"   Shape: {data.shape}")
                print(f"   Dtype: {data.dtype}")
                print(f"   Min: {np.min(data):.4f}")
                print(f"   Max: {np.max(data):.4f}")
                print(f"   Mean: {np.mean(data):.4f}")
                print(f"   Std: {np.std(data):.4f}")
                
                image_data.append({
                    "path": file_path,
                    "data": data,
                    "split": split,
                    "name": file_path.name,
                    "folder": folder_name
                })
            except Exception as e:
                print(f"HATA: {file_path.name} yuklenirken hata: {e}")
        
        return image_data
    
    @staticmethod
    def calculate_metrics(data):
        """Bir veri dizisi için metrikleri hesaplar"""
        metrics = {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "zero_ratio": np.sum(data == 0) / data.size if data.size > 0 else 0.0
        }
        return metrics
    
    @staticmethod
    def calculate_mse(data1, data2):
        """İki veri dizisi arasındaki MSE'yi hesaplar"""
        if data1.shape != data2.shape:
            return None
        return np.mean((data1 - data2) ** 2)
    
    def print_metrics_comparison(self):
        """Aynı isimdeki dosyalar için MSE, STD, MEAN ve ZERO oranlarını karşılaştırır"""
        print("\n" + "=" * 120)
        print(f"METRIK KARSILASTIRMASI ({self.file_name})")
        print("=" * 120)
        
        all_data = {}
        
        for folder_name, folder_path in self.folders:
            for split in ["train", "val"]:
                split_path = folder_path / split
                if split_path.exists():
                    for file_path in sorted(split_path.glob(f"{self.file_name}*.npy")):
                        try:
                            data = np.load(file_path)
                            key = (file_path.name, split)
                            if key not in all_data:
                                all_data[key] = {}
                            all_data[key][folder_name] = {
                                "data": data,
                                "path": file_path
                            }
                        except Exception as e:
                            print(f"HATA: {file_path} yuklenirken hata: {e}")
        
        if not all_data:
            print(f"UYARI: {self.file_name} ile baslayan dosya bulunamadi!")
            return
        
        for (filename, split), file_dict in sorted(all_data.items()):
            print(f"\n{'='*120}")
            print(f"{filename} ({split})")
            print(f"{'='*120}")
            
            metrics_dict = {}
            for folder_name in ["images", "freq_embeddings", "amp_init", "emb_init"]:
                if folder_name in file_dict:
                    data = file_dict[folder_name]["data"]
                    metrics_dict[folder_name] = self.calculate_metrics(data)
            
            print(f"\n{'Klasor':<20s} {'MEAN':>15s} {'STD':>15s} {'MIN':>15s} {'MAX':>15s} {'ZERO_RATIO':>15s}")
            print("-" * 120)
            for folder_name in ["images", "freq_embeddings", "amp_init", "emb_init"]:
                if folder_name in metrics_dict:
                    m = metrics_dict[folder_name]
                    print(f"{folder_name:<20s} {m['mean']:>15.6f} {m['std']:>15.6f} {m['min']:>15.6f} {m['max']:>15.6f} {m['zero_ratio']:>15.6f}")
                else:
                    print(f"{folder_name:<20s} {'YOK':>15s} {'YOK':>15s} {'YOK':>15s} {'YOK':>15s} {'YOK':>15s}")
            
            if "images" in metrics_dict:
                print(f"\nMSE Karsilastirmalari (images referans):")
                print("-" * 120)
                ref_data = file_dict["images"]["data"]
                
                for folder_name in ["freq_embeddings", "amp_init", "emb_init"]:
                    if folder_name in file_dict:
                        comp_data = file_dict[folder_name]["data"]
                        mse = self.calculate_mse(ref_data, comp_data)
                        if mse is not None:
                            print(f"  images vs {folder_name:<20s}: MSE = {mse:.6f}")
            
            print(f"\nTum Klasorler Arasi MSE Karsilastirmalari:")
            print("-" * 120)
            folder_names = [f for f in ["images", "freq_embeddings", "amp_init", "emb_init"] if f in file_dict]
            
            for i, folder1 in enumerate(folder_names):
                for folder2 in folder_names[i+1:]:
                    data1 = file_dict[folder1]["data"]
                    data2 = file_dict[folder2]["data"]
                    mse = self.calculate_mse(data1, data2)
                    if mse is not None:
                        print(f"  {folder1:<20s} vs {folder2:<20s}: MSE = {mse:.6f}")
        
        print("\n" + "=" * 120)
    
    def print_same_filename_files(self):
        """Aynı isimdeki dosyaları yan yana gösterir"""
        print("\n" + "=" * 120)
        print("Ayni Isimdeki Dosyalar (Yan Yana)")
        print("=" * 120)
        
        all_files = {}
        
        for folder_name, folder_path in self.folders:
            for split in ["train", "val"]:
                split_path = folder_path / split
                if split_path.exists():
                    for file_path in sorted(split_path.glob("*.npy")):
                        filename = file_path.name
                        key = (filename, split)
                        if key not in all_files:
                            all_files[key] = {}
                        all_files[key][folder_name] = file_path
        
        for (filename, split), file_dict in sorted(all_files.items()):
            file_list = []
            for folder_name in ["images", "freq_embeddings", "amp_init", "emb_init"]:
                if folder_name in file_dict:
                    file_list.append(f"{folder_name}/{split}/{file_dict[folder_name].name}")
                else:
                    file_list.append("YOK")
            
            print(f"{filename:45s} ({split:5s}):  {'  |  '.join(file_list)}")
        
        print("\n" + "=" * 120)
    
    def visualize(self, output_name="dataset_analysis", num_samples=3):
        """Tüm görüntü verilerini görselleştirir"""
        print("\n" + "=" * 60)
        print("GORSELLESTIRME")
        print("=" * 60)
        
        all_data = []
        
        for folder_name, folder_path in self.folders:
            folder_data = self.analyze_folder(folder_path, folder_name, num_samples=num_samples)
            if folder_data:
                all_data.append(folder_data)
        
        if not all_data or all([data is None for data in all_data]):
            print("UYARI: Gorsellestirilecek veri yok!")
            return
        
        all_image_data = []
        for data_list in all_data:
            if data_list:
                all_image_data.extend(data_list)
        
        if not all_image_data:
            print("UYARI: Gorsellestirilecek veri yok!")
            return
        
        num_images = len(all_image_data)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, img_info in enumerate(all_image_data):
            ax = axes[idx]
            data = img_info["data"]
            folder_name = img_info.get("folder", "unknown")
            
            is_embedding = (
                "emb" in folder_name.lower() or "embedding" in folder_name.lower()
                or "emb" in img_info.get("name", "").lower() or "embedding" in img_info.get("name", "").lower()
            )
            embedding_cmap_2d = "seismic" if is_embedding else "hot"
            embedding_cmap_3d = "seismic" if is_embedding else "hot"
            embedding_cmap_4d = "seismic" if is_embedding else "gray"
            
            if len(data.shape) == 1:
                ax.plot(data)
                ax.set_title(f"{folder_name}\n{img_info['name']}\n({img_info['split']})", fontsize=9)
                ax.grid(True, alpha=0.3)
            elif len(data.shape) == 2:
                im = ax.imshow(data, cmap=embedding_cmap_2d, aspect='auto')
                ax.set_title(f"{folder_name}\n{img_info['name']}\n{data.shape} ({img_info['split']})", fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046)
            elif len(data.shape) == 3:
                if data.shape[2] == 3:
                    ax.imshow(data)
                else:
                    im = ax.imshow(data[:, :, 0], cmap=embedding_cmap_3d, aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f"{folder_name}\n{img_info['name']}\n{data.shape} ({img_info['split']})", fontsize=9)
            else:
                if len(data.shape) == 4:
                    im = ax.imshow(data[0, :, :, 0] if data.shape[3] > 1 else data[0, :, :],
                                   cmap=embedding_cmap_4d, aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                else:
                    ax.text(0.5, 0.5, f"Shape: {data.shape}\nGorsellestirilemedi", 
                           ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{folder_name}\n{img_info['name']}\n{data.shape} ({img_info['split']})", fontsize=9)
            
            ax.axis('off')
        
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = f"{output_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nGorsellestirme kaydedildi: {output_path}")
        
        plt.show()
        
        print("\n" + "=" * 60)
        print("Analiz tamamlandi!")
        print("=" * 60)
    
    def run_full_analysis(self):
        """Tam analiz çalıştırır"""
        self.analyze_dataset()
        self.print_same_filename_files()
        self.print_metrics_comparison()
        self.visualize()


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Dataset analizi ve görselleştirme')
    parser.add_argument('--file-name', type=str, required=True,
                        help='Analiz edilecek dosya ismi (prefix)')
    parser.add_argument('--dataset-path', type=str, default='dataset',
                        help='Dataset klasör yolu (varsayılan: dataset)')
    
    args = parser.parse_args()
    
    visualizer = DatasetVisualizer(args.file_name, args.dataset_path)
    visualizer.run_full_analysis()


if __name__ == "__main__":
    main()
