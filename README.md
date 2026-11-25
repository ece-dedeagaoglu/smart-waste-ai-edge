# Smart Waste AI Edge

An edge-AI waste classification and mechanical sorting system built for **Raspberry Pi 5**. The system combines motion-based foreground segmentation (`OpenCV MOG2`), dual **TensorFlow Lite** models (`ResNet50` and `EfficientNetB0` at `224x224` input resolution), rule-based class fusion with multi-frame stability filtering, and dual-servo control (`gpiozero` with `LGPIOFactory`).

## System Architecture

1. **Motion Detection & ROI Extraction:** Captures frames via OpenCV, applies `BackgroundSubtractorMOG2` with `5x5` elliptical morphological closing, and isolates moving objects (`contourArea > 8000`) to crop a square region of interest.
2. **Dual-Model Edge Inference:** Resizes the candidate crop to `224x224` and runs inference on two TFLite models:
   * **ResNet50 (`garbage_resnet50.tflite`):** 6-class classifier (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) with ImageNet RGB mean subtraction (`103.939`, `116.779`, `123.68`).
   * **EfficientNetB0 (`garbage_efficientnet.tflite`):** 12-class classifier (`battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`).
3. **Fusion & Stability Gating:** Maps model predictions into target sorting classes (`BIO`, `METAL`, `PLASTIC`) based on confidence thresholds and requires **4 consecutive matching frames** (`STABILITY_THRESHOLD = 4`) before triggering actuation.
4. **Servo Actuation (Raspberry Pi 5):** Uses `LGPIOFactory` to control two servos:
   * **GPIO 19:** Rotates the sorting mechanism to the target compartment (`PLASTIC`, `METAL`, `BIO`).
   * **GPIO 18:** Tilts the lid to release the item and returns the mechanism to neutral standby.

## Repository Structure

* `src/main.py` — Full Raspberry Pi 5 sorting pipeline (camera motion segmentation, dual TFLite inference, stability filter, and servo control).
* `src/camera_fusion_test.py` — Camera and dual-model fusion test script without servo actuation.
* `docs/smart_waste_ai_flowchart.svg` — System workflow diagram.
* `models/` — Directory for placing `.tflite` weight files (`garbage_resnet50.tflite`, `garbage_efficientnet.tflite`).
* `requirements.txt` — Python dependencies.

## Setup & Usage

Place the `.tflite` model weights inside `models/` before running inference (large weight files, virtual environments, and intermediate experiment files are excluded from Git).

```bash
pip install -r requirements.txt