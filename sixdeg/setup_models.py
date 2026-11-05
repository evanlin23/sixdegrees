"""Download the local models (face detection/recognition + TTS). Idempotent."""
from .common import MODELS, download

MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx":
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    "kokoro-v1.0.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def main():
    for name, url in MODEL_FILES.items():
        p = download(url, MODELS / name)
        print(f"ok {name} ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
