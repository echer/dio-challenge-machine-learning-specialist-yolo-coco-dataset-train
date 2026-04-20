import fiftyone as fo
from collections import Counter

dataset = fo.Dataset()
# carregar treino
dataset.add_dir(
    dataset_dir="dataset/YOLODataset",
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",
)
# carregar validação
dataset.add_dir(
    dataset_dir="dataset/YOLODataset",
    dataset_type=fo.types.YOLOv5Dataset,
    split="val",
)

counts = Counter()
for sample in dataset:
    if sample.ground_truth:
        for det in sample.ground_truth.detections:
            counts[det.label] += 1

print(counts)

empty = 0

for sample in dataset:
    if not sample.ground_truth or len(sample.ground_truth.detections) == 0:
        empty += 1

print("negativas:", empty)

import collections

img_counts = collections.Counter()

for sample in dataset:
    labels = set()
    if sample.ground_truth:
        for det in sample.ground_truth.detections:
            labels.add(det.label)
    for l in labels:
        img_counts[l] += 1

print(img_counts)

#session = fo.launch_app(dataset)
#session.wait()