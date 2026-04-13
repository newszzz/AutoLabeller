from __future__ import annotations

from ultralytics import YOLO

from .config import ObjectClassConfig, YoloConfig
from .schemas import AnnotationResult, BoundingBox, ImageRecord


class YoloPreAnnotator:
    def __init__(self, config: YoloConfig, classes: list[ObjectClassConfig]):
        self.config = config
        self.classes = classes
        self.class_names = [item.name for item in classes]
        self.model = YOLO(str(config.model_path), task="detect")

    def annotate(self, record: ImageRecord) -> AnnotationResult:
        prediction = self.model.predict(
            source=str(record.image_path),
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )[0]

        boxes: list[BoundingBox] = []
        if prediction.boxes is not None:
            xywhn = prediction.boxes.xywhn.cpu().tolist()
            cls = prediction.boxes.cls.cpu().tolist()
            confs = prediction.boxes.conf.cpu().tolist()
            for coords, class_id, conf in zip(xywhn, cls, confs, strict=True):
                label = self.class_names[int(class_id)]
                boxes.append(
                    BoundingBox(
                        label=label,
                        x_center=float(coords[0]),
                        y_center=float(coords[1]),
                        width=float(coords[2]),
                        height=float(coords[3]),
                        confidence=float(conf),
                    )
                )

        return AnnotationResult(
            boxes=boxes,
            summary=f"YOLO ONNX proposed {len(boxes)} objects.",
        )
