# MedVision — Jetson edge inference (ONNX Runtime, no PyTorch)
#
# Base image MUST match your JetPack / L4T version:
#   • Jetson Nano / TX2  → JetPack 4.6  → L4T r32.7.1  (image below)
#   • Jetson Orin Nano/NX → JetPack 5.x → nvcr.io/nvidia/l4t-ml:r35.x
#   • JetPack 6.x         → r36.x
# The l4t-ml image already bundles onnxruntime-gpu, numpy, and pillow.
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

WORKDIR /app

# only the API layer is missing from the base image
RUN pip3 install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY artifacts/model.onnx ./artifacts/model.onnx

EXPOSE 8000

# ONNX Runtime auto-selects the CUDA/TensorRT provider on the Jetson GPU
CMD ["uvicorn", "src.api.edge_main:app", "--host", "0.0.0.0", "--port", "8000"]
