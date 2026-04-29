from __future__ import annotations

from ultralytics import YOLO

from .config import ObjectClassConfig, YoloConfig
from .schemas import AnnotationResult, BoundingBox, ImageRecord


class YoloPreAnnotator:
    def __init__(self, config: YoloConfig, classes: list[ObjectClassConfig]):
        self.config = config
        self.class_id_to_name = {int(item.id): item.name for item in classes if item.id is not None}
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
            xyxy = prediction.boxes.xyxy.cpu().tolist()
            class_ids = prediction.boxes.cls.cpu().tolist()
            confidences = prediction.boxes.conf.cpu().tolist()
            for coords, class_id, confidence in zip(xyxy, class_ids, confidences, strict=True):
                label = self._label_for_class_id(int(class_id))
                if label is None:
                    continue
                x_min = _clamp(float(coords[0]), 0.0, float(record.width - 1))
                y_min = _clamp(float(coords[1]), 0.0, float(record.height - 1))
                x_max = _clamp(float(coords[2]), 0.0, float(record.width))
                y_max = _clamp(float(coords[3]), 0.0, float(record.height))
                if x_max <= x_min or y_max <= y_min:
                    continue
                boxes.append(
                    BoundingBox(
                        label=label,
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        confidence=float(confidence),
                    )
                )

        return AnnotationResult(objects=boxes)

    def _label_for_class_id(self, class_id: int) -> str | None:
        return self.class_id_to_name.get(class_id)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
