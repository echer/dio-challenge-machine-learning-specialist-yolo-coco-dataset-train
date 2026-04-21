# Primeira abordagem criar um dataset de imagens e rotular manualmente

Para esta etapa foram copiadas manualmente algumas imagens de `gato` e `cachorro` obtidas do dataset coco-2017 para a pasta `dataset/images/labeled`, depois foram rotuladas utilizando o `labelme` GUI do python, apos a rotulação foram convertidas em formato `Yolo` utilizando o `labelme2yolo`.

Estrutura da pasta: 
- `labeled`: contem as imagens rotuladas manualmente (53 gatos / 67 cachorros)
- `negative`: contém imagens não rotuladas pois não possuem gatos nem cachorros
- `test`: contém imagens utilizadas para testar o reconhecimento de gatos e cachorro do modelo

![dataset structure](./assets/dataset-structure.png)

Aqui temos um exemplo de como a rotulação foi feita:
![labelme](./assets/labelme.png)

Aqui temos todas as imagens rotuladas no fiftyone:
![](./assets/labeled_images.png)

![](./assets/labeled_cat.png)

![](./assets/labeled_dog.png)

O código utilizado para treinamento foi o seguinte:

``` python

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

```

O treino foi realizado na maquina local utilizando a GPU abaixo:

``` shell

(tf) aecher@aecher-RYZEN:~/tf$ nvidia-smi
Tue Apr 21 10:34:17 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060        Off |   00000000:07:00.0  On |                  N/A |
|  0%   39C    P5             11W /  145W |     394MiB /   8151MiB |     15%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

```

O resultado do treino não foi muito satisfatório pois o dataset era pequeno para conseguir obter conhecimento assertivo, mas conseguiu fazer algumas identificações, algumas não identificou nada, algumas identificou um objeto o outro não:

![](./assets/resultado-rotulagem-manual.jpg)

# Segunda abordagem utilizar mais dados já rotulados do COCO-2017

Aqui nesta etapa utilizei uma quantidade maior de dados no treinamento, os dados foram obtidos do dataset coco-2017, o resultado foi um pouco mais satisfatório pois melhorou os resultados principalmente em imagens que tinham mais de um objeto e gatos e cachorros juntos, ainda manteve alguns erros, algumas não foram identificadas, algumas foram identificadas errada.

A estrutura ficou assim:

![](./assets/dataset-coco-yolo.png)

O código obtem 600 imagens de gatos e cachorros para treino e 200 imagens para validação do dataset coco-2017 assim como as suas labels, depois gera o mapa `dataset.yaml` para ser utilizado no modelo, treina e aplica o modelo nas imagens de validação.

``` python
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
    max_samples=600
)

dataset_val = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=classes,
    max_samples=200
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
```

Resultado obtido:

![](assets/resultado-coco-2017-600-yolo26n.jpg)

Pude perceber que quanto mais imagens temos no treino maior é o ganho na identificação de gatos e cachorros, então fiz outro teste aumentando a base de treino para 2000 imagens e a melhora foi significativamente visivel, casos em que identificou objetos incorretos melhoraram, casos em que havia mais de um objeto também e a assertividade, porém ainda há falsos positivos que poderiam ser melhorados com mais imagens ou modelo maior.

![](assets/resultado-coco-2017-2000-yolo26n.jpg)

Agora mantive as mesmas 2000 imagens porém troquei o modelo por um maior, porém não houve mudanças significativas e os mesmos erros ou mais permaneceram aparentemente, porém demorou bem mais.

![](assets/resultado-coco-2017-2000-yolo26s.jpg)

Agora voltei o modelo yolo26n e usei 5000 imagens, porém houveram algumas perdas e alguns ganhos em relação a 2000 com o mesmo modelo.

![](assets/resultado-coco-2017-5000-yolo266n.jpg)
