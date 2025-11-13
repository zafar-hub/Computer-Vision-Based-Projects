import os
import subprocess
from ultralytics import YOLO

# === CONFIG ===
model_path = r"A:Pathpath\runs\classify\train4\weights\best.pt"
onnx_path = os.path.splitext(model_path)[0] + ".onnx"
export_dir = os.path.dirname(model_path)
input_shape = "1,3,320,320"  # B,C,H,W for YOLOv8-classify
quantize = False  # change to True for fp16 quantization

# === 1. Export to ONNX ===
print("🚀 Exporting YOLO model to ONNX...")
model = YOLO(model_path)
model.export(format="onnx")
print(f"✅ Exported ONNX model: {onnx_path}")

# === 2. Convert ONNX → TensorFlow Lite using onnx2tf ===
print("⚙️ Converting ONNX → TensorFlow Lite using onnx2tf...")

cmd = [
    "onnx2tf",
    "-i", onnx_path,
    "-b", "1",
    "-ois", input_shape
]

if quantize:
    cmd += ["-qt", "fp16"]

try:
    subprocess.run(cmd, check=True)
    print("✅ Conversion complete! Check output folder for .tflite files.")
except subprocess.CalledProcessError as e:
    print("❌ Conversion failed.")
    print(e)

