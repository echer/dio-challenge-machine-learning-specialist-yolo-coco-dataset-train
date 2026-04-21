# =========================
# 1. IMPORTS
# =========================
import fiftyone as fo
from ultralytics import YOLO

# =========================
# 2. BAIXAR COCO (cat + dog)
# =========================
classes = ["cat", "dog"]

dataset_train = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=classes,
    max_samples=5000
)

dataset_val = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=classes,
    max_samples=100
)

print("Train:", len(dataset_train))
print("Val:", len(dataset_val))

# =========================
# 3. EXPORTAR PARA YOLO (CORRIGIDO)
# =========================

classes = ["cat", "dog"]

dataset_train.export(
    export_dir="coco_yolo",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="train",
    classes=classes
)

dataset_val.export(
    export_dir="coco_yolo",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="val",
    classes=classes
)


# =========================
# 4. CRIAR dataset.yaml
# =========================
yaml_content = """
path: coco_yolo

train: images/train
val: images/val

names:
  0: cat
  1: dog
"""

with open("coco_yolo/dataset.yaml", "w") as f:
    f.write(yaml_content)

print("dataset.yaml criado")


# =========================
# 5. TREINAMENTO YOLO
# =========================
model = YOLO("yolo26n.pt")

model.train(
    data="coco_yolo/dataset.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    patience=10
)


# =========================
# 6. VALIDAÇÃO
# =========================
metrics = model.val()

print("\n===== MÉTRICAS =====")
print(metrics)


# =========================
# 7. PREDIÇÃO (TESTE)
# =========================
model.predict(
    source="coco_yolo/images/val",
    save=True,
    conf=0.25
)