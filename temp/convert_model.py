import onnx

# Load model
model = onnx.load("best.onnx")

# Convert opset to 21
onnx_model = onnx.version_converter.convert_version(model, 21)

# Save downgraded model
onnx.save(onnx_model, "best_opset21.onnx")
print("Saved: best_opset21.onnx")
