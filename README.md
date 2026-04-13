# AutoLabeller

基于 `LangChain + Ollama + YOLO(ONNX)` 的自动标注项目。

流程：
1. 使用 ONNX 格式的 YOLO 模型做预标注。
2. 使用本地 Ollama 多模态模型独立标注。
3. 使用复核 agent 汇总两份结果并输出最终标签。
4. 可选地把 YOLO 标注数据集导出成 SFT 训练样本。

## 运行

```bash
python main.py annotate --config config/example.yaml
python main.py export-sft --config config/example.yaml
```

## 配置说明

- `dataset.classes` 使用 `name + description` 定义类别。
- `yolo.model_path` 必须是 `.onnx` 文件。
- 自动标注流程只依赖图片，不再读取 ground-truth 标签，也不做 evaluate。
- `finetune.dataset` 仅在导出已有标注数据集时使用。
