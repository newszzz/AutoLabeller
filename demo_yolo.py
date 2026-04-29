from ultralytics import YOLO

model = YOLO("val_JZ_v8s320.onnx", task="detect")

prediction = model.predict(
    source="demo2.jpg",
    conf=0.25,
    iou=0.5,
    verbose=False
)

print(prediction[0])
