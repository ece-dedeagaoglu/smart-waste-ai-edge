# File Guide: Autonomous Waste Sorting Edge AI

This guide documents every active file in the repository, earlier prototype/experiment scripts, fusion rules, and offline evaluation results on `test_images/`.

## 1. Active Repository Files

- `src/main.py`: Primary Raspberry Pi 5 runtime script. Resolves models from `models/garbage_resnet50.tflite` and `models/garbage_efficientnet.tflite`, captures frames via OpenCV (`VideoCapture(0)`), segments moving objects using `BackgroundSubtractorMOG2` and morphological closing (`contourArea > 8000`), runs dual-model inference at `224x224`, applies class fusion rules (`BIO`, `METAL`, `PLASTIC`), enforces a 4-frame consecutive stability filter (`STABILITY_THRESHOLD = 4`), and drives two servos via `gpiozero` (`LGPIOFactory`) on `GPIO 18` (lid tilt) and `GPIO 19` (compartment rotation).
- `src/godfather.py`: Turkish-logged full hardware + dual-TFLite fusion script aligned with `models/garbage_resnet50.tflite` and `models/garbage_efficientnet.tflite`. Implements the same MOG2 tracking, confidence thresholds, 4-frame stability logic (`KARARLILIK_ESIGI = 4`), and `GPIO 18` / `GPIO 19` actuation sequence.
- `src/evaluate_images.py`: Hardware-free offline evaluator for testing the dual TFLite fusion pipeline on static images without a camera or Raspberry Pi GPIO hardware. Loads both models via LiteRT (`ai_edge_litert.interpreter` with `tensorflow.lite` fallback), applies `ResNet50` ImageNet mean subtraction (`103.939`, `116.779`, `123.68`) and `EfficientNetB0` `float32` RGB input, and prints per-image predictions and fusion decisions.
- `archive/prototypes/camera_fusion_test.py`: Earlier LiteRT camera + dual-model fusion test script without servo actuation; originally written with `cv2.VideoCapture(0, cv2.CAP_DSHOW)` and root-level model paths.
- `docs/smart_waste_ai_flowchart.svg`: System workflow diagram illustrating detection, inference, fusion, stability filtering, and sorting actuation.
- `test_images/`: Seven sample images used for offline model and fusion verification.
- `README.md`: Project overview, architecture summary, fusion thresholds, repository layout, and evaluation usage.
- `requirements.txt`: Python dependencies for inference, computer vision, and Raspberry Pi 5 GPIO control.
- `.gitignore`: Excludes virtual environments (`.venv/`), cache files, and large `.tflite` / `.h5` weight files from Git tracking.

## 2. Earlier Prototypes & Exploratory Scripts (Outside Active `src/` Runtime)

During earlier experimentation stages, several standalone/prototype scripts were created for single-model classification, alternative label sets, angle mapping, or external detectors:

- `ai_inference.py`: Defines `WasteClassifier` using a single `waste_classifier.tflite` model (`224x224`, normalized `[0, 1]` via `/ 255.0`) and `CLASS_NAMES = ["metal", "organic", "plastic"]`.
- `batch_test_tflite.py`: Batch-tests images in `test_images/` against `waste_classifier.tflite` (`["metal", "organic", "plastic"]`, `[0, 1]` normalization) and prints top label and confidence.
- `test_tflite.py`: CLI script that runs a single image through `waste_classifier.tflite` and prints sorted class probabilities (`metal`, `organic`, `plastic`).
- `label_map.py`: Holds `CLASS_NAMES = ["metal", "organic", "plastic"]` for the single-model prototype flow.
- `servo_logic.py`: Maps `plastic: -40`, `organic: 0`, `metal: 40` via `get_servo_angle()`, distinct from the `gpiozero` (`GPIO 18` / `GPIO 19`) sequence in `src/main.py` and `src/godfather.py`.
- `train_model.py`: Trains a 3-class (`metal`, `organic`, `plastic`) `MobileNetV2` model on a `dataset/` directory (`ImageDataGenerator` with `rescale=1./255`, `validation_split=0.2`, 5 epochs) and saves `waste_classifier.h5`.
- `test_model.py`: CLI script that loads `waste_classifier.h5` via Keras and predicts on a single image (`metal`, `organic`, `plastic`).
- `dad_test_camera.py`: Near-identical predecessor/duplicate of the LiteRT dual-model camera fusion test script.
- `test_best.py`: Real-time webcam script using `ultralytics.YOLO` with an external path (`/Users/air/Downloads/ece`) and `0.75` confidence threshold.

## 3. Dual-Model Fusion Rules & Target Compartments

- **Model 1 (`garbage_resnet50.tflite`):** 6 classes (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`), `224x224` RGB `float32` with ImageNet mean subtraction (`B/G/R` channel offsets `103.939`, `116.779`, `123.68` applied to RGB array channels).
- **Model 2 (`garbage_efficientnet.tflite`):** 12 classes (`battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`), `224x224` RGB `float32` without mean subtraction.
- **Fusion Thresholds:**
  - `BIO`: `class2 == "biological"` and `conf2 >= 85.0%`
  - `METAL`: (`class1 == "metal"` and `conf1 >= 70.0%`) or (`class2 == "metal"` and `conf2 >= 70.0%`)
  - `PLASTIC`: (`class1 == "plastic"` and `conf1 >= 85.0%`) or (`class2 == "plastic"` and `conf2 >= 85.0%`)
  - Any other prediction returns `None` (`0.0%`) and does not trigger sorting.

## 4. Offline Evaluation Results on `test_images/`

| Image File | ResNet50 Top Class | EfficientNetB0 Top Class | Fusion Decision | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `Unknown.jpeg` | `trash` (`47.8%`) | `shoes` (`90.1%`) | `None` (`0.0%`) | Non-target classes; safely ignored |
| `metcancop.jpeg` | `metal` (`100.0%`) | `metal` (`92.5%`) | `METAL` (`100.0%`) | Strong agreement across both models |
| `metcancopp.jpeg` | `glass` (`46.8%`) | `white-glass` (`89.9%`) | `None` (`0.0%`) | Glass is outside `BIO` / `METAL` / `PLASTIC` rules |
| `placop.jpeg` | `metal` (`53.6%`) | `plastic` (`49.8%`) | `None` (`0.0%`) | Below threshold (`metal < 70%`, `plastic < 85%`) |
| `placopp.jpeg` | `cardboard` (`84.5%`) | `metal` (`67.0%`) | `None` (`0.0%`) | Below threshold (`metal < 70%`, `cardboard` non-target) |
| `porcop.jpeg` | `plastic` (`53.7%`) | `biological` (`57.1%`) | `None` (`0.0%`) | Below threshold (`plastic < 85%`, `biological < 85%`) |
| `porcopp.jpeg` | `metal` (`99.8%`) | `biological` (`91.9%`) | `BIO` (`91.9%`) | `EfficientNetB0` `biological >= 85%` triggers `BIO`; highlights model disagreement and why multi-frame stability matters |
