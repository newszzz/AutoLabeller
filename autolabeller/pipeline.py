from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .dataset import (
    collect_image_records,
    llm_result_to_annotation,
    load_class_names,
    load_classes,
    validate_annotation_result,
)
from .multimodal_agent import MultimodalAgent
from .schemas import AnnotationResult, LlmAnnotationResult, ReviewResult
from .utils import (
    build_label_color_map,
    copy_image_as,
    ensure_dir,
    render_annotation_image,
    save_annotation_json,
    write_json,
)
from .yolo_annotator import YoloPreAnnotator


class AutoLabelPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.classes = load_classes(config.dataset)
        self.class_names = load_class_names(config.dataset)
        self.color_by_label = build_label_color_map(self.class_names)
        self.records = collect_image_records(config.dataset)
        self.output_dir = ensure_dir(config.dataset.output_dir)
        self.label_dir = ensure_dir(self.output_dir / "labels")
        self.manual_review_dir = ensure_dir(self.output_dir / "manual_review")
        self.visual_dir = ensure_dir(self.output_dir / "visualizations")
        self.agent = MultimodalAgent(
            config.llm_api,
            self.classes,
            few_shot_visual_dir=ensure_dir(self.visual_dir / "few_shots"),
        )
        self.yolo_annotator = YoloPreAnnotator(config.yolo, self.classes)

    def run(self) -> dict:
        approved_count = 0
        manual_review_count = 0

        for record in self.records:
            relative_png = record.image_path.relative_to(self.config.dataset.images_dir).with_suffix(".png")
            yolo_result = validate_annotation_result(
                self.yolo_annotator.annotate(record),
                self.class_names,
                record.width,
                record.height,
            )
            yolo_image_path = render_annotation_image(
                record.image_path,
                yolo_result,
                self.visual_dir / "yolo" / relative_png,
                self.color_by_label,
            )

            llm_result = self.agent.annotate(record, yolo_result, yolo_image_path)
            if llm_result.issues:
                print(f"Image {record.image_path} has annotation issues: {llm_result.issues}")
            llm_annotation = llm_result_to_annotation(llm_result)
            llm_image_path = render_annotation_image(
                record.image_path,
                llm_annotation,
                self.visual_dir / "llm" / relative_png,
                self.color_by_label,
            )

            review_result = self.agent.review(
                record,
                yolo_result,
                yolo_image_path,
                llm_result,
                llm_image_path,
            )
            final_result = self._select_final_result(review_result, yolo_result, llm_annotation)

            if final_result is None:
                self._save_manual_review_case(
                    record.image_path,
                    yolo_result,
                    yolo_image_path,
                    llm_result,
                    llm_image_path,
                    review_result,
                )
                manual_review_count += 1
            else:
                save_annotation_json(
                    record.image_path,
                    final_result,
                    self.label_dir,
                    self.config.dataset.images_dir,
                )
                render_annotation_image(
                    record.image_path,
                    final_result,
                    self.visual_dir / "final" / relative_png,
                    self.color_by_label,
                )
                approved_count += 1

        summary = {
            "processed_images": len(self.records),
            "approved_images": approved_count,
            "manual_review_images": manual_review_count,
            "saved_labels": str(self.label_dir),
            "manual_review_dir": str(self.manual_review_dir),
            "visualizations": str(self.visual_dir),
        }
        write_json(self.output_dir / "run_summary.json", summary)
        return summary

    def _select_final_result(
        self,
        review_result: ReviewResult,
        yolo_result: AnnotationResult,
        llm_annotation: AnnotationResult,
    ) -> AnnotationResult | None:
        if review_result.yolo_is_correct:
            return yolo_result
        if review_result.llm_is_correct:
            return llm_annotation
        return None

    def _save_manual_review_case(
        self,
        image_path: Path,
        yolo_result: AnnotationResult,
        yolo_image_path: Path,
        llm_result: LlmAnnotationResult,
        llm_image_path: Path,
        review_result: ReviewResult,
    ) -> None:
        case_dir = ensure_dir(
            self.manual_review_dir
            / image_path.relative_to(self.config.dataset.images_dir).with_suffix("")
        )
        copy_image_as(image_path, case_dir / f"images{image_path.suffix}")
        write_json(case_dir / "yolo.json", yolo_result)
        write_json(case_dir / "llm.json", llm_result)
        write_json(case_dir / "review.json", review_result)
        copy_image_as(yolo_image_path, case_dir / f"yolo{yolo_image_path.suffix}")
        copy_image_as(llm_image_path, case_dir / f"llm{llm_image_path.suffix}")
