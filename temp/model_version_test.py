import onnxruntime as ort

session = ort.InferenceSession("best_opset21.onnx")
print("Model loaded successfully")
