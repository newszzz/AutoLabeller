# AutoLabeller

基于 `LangChain + llama-factory(OpenAI-style API) + YOLO(ONNX)` 的自动标注项目。

流程：
1. 使用 ONNX YOLO 模型对图片做预标注，只保留 `dataset.classes` 中声明的类别。
2. 将原图、YOLO 标注 JSON、YOLO 标注图和 few-shot 示例输入多模态标注 agent，由 agent 删除多余框、补充漏框。
3. 将原图、YOLO 结果、LLM 结果以及两者的标注图输入 review agent，由 reviewer 判断两份标注是否正确；最终保存时固定按 YOLO 优先、LLM 其次、人工复核兜底。
4. 正确标注保存到 `outputs/labels`，两者都不正确的样本保存到 `outputs/manual_review` 供人工处理。

所有 agent 输入输出中的框坐标都使用原图像素绝对值：`x_min, y_min, x_max, y_max`。

## 运行

```bash
python -m autolabeller.main config/example.yaml
```

## 配置说明

- `dataset.classes` 使用 `id + name + description` 定义类别；`id` 对应 YOLO 模型输出的 class id，`name` 是最终保存和 LLM 输出使用的标签名。
- `dataset.few_shots` 中的 `annotation_path` 推荐使用 JSON；现有 YOLO `.txt` 也会先转换为像素坐标 JSON 再输入模型。
- `yolo.model_path` 必须是 `.onnx` 文件。
- `llm_api.backend` 支持 `vllm` 和 `ollama`，两者都通过 OpenAI-compatible `/v1` 接口调用。
- `llm_api.model` 同时用于 annotator 和 reviewer。
- `llm_api.base_url` 可省略：`vllm` 默认 `http://localhost:8000/v1`，`ollama` 默认 `http://localhost:11434/v1`。
