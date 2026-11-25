from pathlib import Path
import argparse
import cv2
import numpy as np

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESNET = BASE_DIR / "models" / "garbage_resnet50.tflite"
DEFAULT_EFFNET = BASE_DIR / "models" / "garbage_efficientnet.tflite"
DEFAULT_IMAGES_DIR = BASE_DIR / "test_images"

CLASS_NAMES_MODEL1 = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_NAMES_MODEL2 = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]


def fuse_predictions(class1: str, conf1: float, class2: str, conf2: float):
    if class2 == "biological" and conf2 >= 85.0:
        return "BIO", conf2
    if class1 == "metal" and conf1 >= 70.0:
        return "METAL", conf1
    if class1 == "plastic" and conf1 >= 85.0:
        return "PLASTIC", conf1
    if class2 == "plastic" and conf2 >= 85.0:
        return "PLASTIC", conf2
    if class2 == "metal" and conf2 >= 70.0:
        return "METAL", conf2
    return None, 0.0


def run_evaluation(images_dir: Path, resnet_path: Path, effnet_path: Path):
    interp1 = tflite.Interpreter(model_path=str(resnet_path))
    interp1.allocate_tensors()
    in1, out1 = interp1.get_input_details()[0], interp1.get_output_details()[0]

    interp2 = tflite.Interpreter(model_path=str(effnet_path))
    interp2.allocate_tensors()
    in2, out2 = interp2.get_input_details()[0], interp2.get_output_details()[0]

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"{img_path.name:20s} | unreadable image")
            continue

        resized = cv2.resize(frame, (224, 224))

        img1 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        img1[:, :, 0] -= 103.939
        img1[:, :, 1] -= 116.779
        img1[:, :, 2] -= 123.68
        interp1.set_tensor(in1["index"], np.expand_dims(img1, axis=0))
        interp1.invoke()
        p1 = interp1.get_tensor(out1["index"]).flatten()
        idx1 = int(np.argmax(p1))
        c1, conf1 = CLASS_NAMES_MODEL1[idx1], float(100.0 * p1[idx1])

        img2 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        interp2.set_tensor(in2["index"], np.expand_dims(img2, axis=0))
        interp2.invoke()
        p2 = interp2.get_tensor(out2["index"]).flatten()
        idx2 = int(np.argmax(p2))
        c2, conf2 = CLASS_NAMES_MODEL2[idx2], float(100.0 * p2[idx2])

        final_cls, final_conf = fuse_predictions(c1, conf1, c2, conf2)
        print(
            f"{img_path.name:18s} | "
            f"ResNet: {c1:9s} ({conf1:5.1f}%) | "
            f"EffNet: {c2:11s} ({conf2:5.1f}%) | "
            f"Fusion: {str(final_cls):7s} ({final_conf:5.1f}%)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dual TFLite fusion on test images.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--resnet", type=Path, default=DEFAULT_RESNET)
    parser.add_argument("--effnet", type=Path, default=DEFAULT_EFFNET)
    args = parser.parse_args()
    run_evaluation(args.images_dir, args.resnet, args.effnet)
