# =========================
# 1. IMPORTS
# =========================
from ultralytics import YOLO

# =========================
# 2. CARREGAR MODELO
# =========================
model = YOLO("yolo26n.pt")  # modelo leve ideal pro seu dataset

# =========================
# 3. TREINAMENTO
# =========================
model.train(
    data="dataset/YOLODataset/dataset.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
)

# =========================
# 4. VALIDAÇÃO
# =========================
metrics = model.val()

print("\n===== MÉTRICAS =====")
print(metrics)

# =========================
# 5. PREDIÇÃO (TESTE)
# =========================
model.predict(
    source="dataset/images/test",
    save=True,
)