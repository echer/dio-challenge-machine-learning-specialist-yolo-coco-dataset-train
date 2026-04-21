from PIL import Image
import os
import math

# =========================
# 1. CAMINHO DAS IMAGENS
# =========================
folder = "runs/detect/predict-3"  # ajuste se for predict2, predict3...

images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]

# ordenar (opcional)
images.sort()

# =========================
# 2. ABRIR IMAGENS
# =========================
imgs = [Image.open(img) for img in images]

# =========================
# 3. DEFINIR GRID
# =========================
cols = 3  # número de colunas
rows = math.ceil(len(imgs) / cols)

# tamanho padrão (pega da primeira)
w, h = imgs[0].size

# =========================
# 4. CRIAR IMAGEM FINAL
# =========================
grid_img = Image.new("RGB", (cols * w, rows * h))

# =========================
# 5. COLAR IMAGENS
# =========================
for i, img in enumerate(imgs):
    row = i // cols
    col = i % cols
    grid_img.paste(img, (col * w, row * h))

# =========================
# 6. SALVAR
# =========================
output_path = "assets/resultado-rotulagem-manual.jpg"
grid_img.save(output_path)

print("Imagem salva em:", output_path)