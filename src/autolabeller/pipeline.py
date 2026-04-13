from __future__ import annotations

from .config import AppConfig
from .dataset import collect_image_records, load_class_names, load_classes, save_yolo_annotation
from .reviewer_agent import ReviewAgent
from .schemas import AnnotationResult
from .utils import ensure_dir, write_json
from .vlm_agent import MultimodalAnnotatorAgent
from .yolo_annotator import YoloPreAnnotator


class AutoLabelPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.classes = load_classes(config.dataset)
        self.class_names = load_class_names(config.dataset)
        self.records = collect_image_records(config.dataset)
        self.output_dir = ensure_dir(config.dataset.output_dir)
        self.json_dir = ensure_dir(self.output_dir / "json")
        self.yolo_label_dir = ensure_dir(self.output_dir / "yolo_labels")
        self.final_label_dir = ensure_dir(self.output_dir / "final_labels")
        self.vlm_agent = MultimodalAnnotatorAgent(config.ollama, self.classes)
        self.review_agent = ReviewAgent(config.ollama, self.classes)
        self.yolo_annotator = (
            YoloPreAnnotator(config.yolo, self.classes) if config.yolo.enabled else None
        )

    def run(self) -> dict:
        limit = self.config.pipeline.max_images or len(self.records)

        for record in self.records[:limit]:
            yolo_result = (
                self.yolo_annotator.annotate(record)
                if self.yolo_annotator
                else AnnotationResult(image_path=record.image_path, boxes=[], source="yolo")
            )
            vlm_result = self.vlm_agent.annotate(record)
            review_result = self.review_agent.review(record, yolo_result, vlm_result)
            final_result = AnnotationResult(
                image_path=record.image_path,
                boxes=review_result.final_boxes,
                source="reviewer",
                summary=review_result.summary,
                issues=review_result.suspicious_labels,
            )

            save_yolo_annotation(
                yolo_result,
                labels_dir=self.yolo_label_dir,
                images_dir=self.config.dataset.images_dir,
                class_names=self.class_names,
            )
            save_yolo_annotation(
                final_result,
                labels_dir=self.final_label_dir,
                images_dir=self.config.dataset.images_dir,
                class_names=self.class_names,
            )

            if self.config.pipeline.save_intermediate_json:
                write_json(
                    self.json_dir
                    / record.image_path.relative_to(self.config.dataset.images_dir).with_suffix(".json"),
                    {
                        "image_path": record.image_path,
                        "yolo": yolo_result,
                        "vlm": vlm_result,
                        "review": review_result,
                        "final": final_result,
                    },
                )

        summary = {
            "processed_images": min(limit, len(self.records)),
            "output_dir": str(self.output_dir),
            "saved_yolo_labels": str(self.yolo_label_dir),
            "saved_final_labels": str(self.final_label_dir),
        }
        write_json(self.output_dir / "run_summary.json", summary)
        return summary
