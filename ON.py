from ultralytics import YOLO
model = YOLO('box.pt') 
model.export(format='onnx', imgsz=640, opset=12)