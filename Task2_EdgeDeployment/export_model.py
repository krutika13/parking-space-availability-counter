from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx')
model.export(format='tflite')
