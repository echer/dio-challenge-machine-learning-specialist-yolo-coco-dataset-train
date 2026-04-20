import fiftyone as fo
import fiftyone.zoo as foz
from ultralytics import YOLO
import fiftyone.utils.random as four

def export_yolo_data(
    samples,
    export_dir,
    classes,
    label_field,
    split = None
    ):

    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(
                samples,
                export_dir,
                classes,
                label_field,
                split
            )
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split
        )

def getDatasetFromCOCO(classes):
    # obtem 200 imagens do dataset coco-2017
    train_dataset = fo.zoo.load_zoo_dataset(
        "coco-2017",
        label_types=["detections"],
        classes=classes,
    ).clone()
    train_dataset.name = "cat-and-dogs-train"
    train_dataset.persistent = True
    train_dataset.save()
    return train_dataset


if __name__ == '__main__':
    classes = ["cat", "dog"]
    label_field = "yolov26l"

    train_dataset = getDatasetFromCOCO(classes)
    train_dataset.untag_samples(train_dataset.distinct("tags"))
    ## split into train and val
    four.random_split(
        train_dataset,
        {"train": 0.8, "val": 0.2}
    )

    ## export in YOLO format
    export_yolo_data(
        train_dataset,
        "dataset_train",
        classes,
        label_field,
        split=["train", "val"]
    )

    #model26 = YOLO("yolo26n.pt")
    #train_dataset.apply_model(model26, label_field="yolov26l")

    #coco_classes = [c for c in dataset.default_classes if not c.isnumeric()]
    #print(coco_classes)

    #session = fo.launch_app(dataset)
    #session.wait()