# AutoLabeller

基于 `LangChain + llama-factory(OpenAI-style API) + YOLO(ONNX)` 的自动标注项目。

流程：
1. 使用 ONNX 格式的 YOLO 模型做预标注。
2. 使用 llama-factory 部署的多模态模型独立标注。
3. 使用复核 agent 汇总两份结果并输出最终标签。
4. YOLO 数据集导出由独立的 llama-factory 数据集工具负责，可同时生成标注与审核任务样本。

## 运行

```bash
python main.py annotate --config config/example.yaml
python export_llamafactory.py --config config/export_example.yaml
```

## 配置说明

- `dataset.classes` 使用 `name + description` 定义类别。
- `yolo.model_path` 必须是 `.onnx` 文件。
- `llama_factory.base_url` 指向 llama-factory 的 OpenAI 风格接口，默认会规范到 `/v1`。
- `autolabeller` 通过 `langchain_openai.ChatOpenAI` 调用 llama-factory 部署模型。
- `config/export_example.yaml` 会导出 llama-factory 兼容的混合任务多模态数据集，包含 `annotate` 与 `review` 两类样本和 `dataset_info.json`。
