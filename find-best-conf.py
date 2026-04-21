import pandas as pd
from ultralytics import YOLO

# carregar modelo
model = YOLO("runs/detect/train/weights/best.pt")

# valores de confiança para testar
confs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

results = []

for conf in confs:
    print(f"\nTestando conf={conf}")

    metrics = model.val(
        data="coco_yolo/dataset.yaml",
        conf=conf,
        verbose=False
    )

    precision = metrics.results_dict["metrics/precision(B)"]
    recall = metrics.results_dict["metrics/recall(B)"]
    map50 = metrics.results_dict["metrics/mAP50(B)"]

    # F1 score (equilíbrio entre precision e recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    results.append({
        "conf": conf,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "F1": f1
    })

# =========================
# CRIAR TABELA
# =========================
df = pd.DataFrame(results)

# ordenar pelo melhor F1
df = df.sort_values(by="F1", ascending=False)

print("\n===== TABELA =====")
print(df)

# =========================
# MELHOR CONF AUTOMÁTICO
# =========================
best_conf = df.iloc[0]["conf"]

print("\n===== MELHOR CONF =====")
print(f"Melhor threshold: {best_conf}")

from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

metrics = model.val(
    data="coco_yolo/dataset.yaml",
    conf=best_conf
)

model.predict(
    source="coco_yolo/images/val",
    save=True,
    conf=best_conf
)

print(metrics)