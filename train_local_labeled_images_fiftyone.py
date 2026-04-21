from collections import Counter
import fiftyone as fo

# =========================
# VISUALIZAÇÃO COM FIFTYONE
# =========================
dataset = fo.Dataset()

# carregar splits
dataset.add_dir(
    dataset_dir="dataset/YOLODataset",
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",
)

dataset.add_dir(
    dataset_dir="dataset/YOLODataset",
    dataset_type=fo.types.YOLOv5Dataset,
    split="val",
)

# abrir interface
session = fo.launch_app(dataset)

# =========================
# 7. ANÁLISE DE BALANCEAMENTO
# =========================
counts = Counter()
empty = 0

for sample in dataset:
    detections = sample.ground_truth.detections

    if len(detections) == 0:
        empty += 1
    else:
        for det in detections:
            counts[det.label] += 1

print("\n===== DISTRIBUIÇÃO =====")
print("Objetos:", counts)
print("Negativas:", empty)

session.wait()