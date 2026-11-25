from pathlib import Path
import time
import cv2
import numpy as np
import tensorflow as tf
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory

# =================================================================
# 1. HARDWARE & SERVO MOTOR CONFIGURATION (GPIO 18 & GPIO 19)
# =================================================================
try:
    pi_5_factory = LGPIOFactory()
    tilt_servo = Servo(18, pin_factory=pi_5_factory)     # Lid tilt mechanism
    rotate_servo = Servo(19, pin_factory=pi_5_factory)   # Bin compartment rotation
    print("[INFO] Servo motors initialized with Pi 5 LGPIOFactory.")
except Exception as e:
    print(f"[WARN] Servo motors could not be initialized: {e}")
    pi_5_factory = None


def reset_servos():
    if pi_5_factory:
        tilt_servo.mid()       # Neutral / flat position
        rotate_servo.min()     # Default starting bin position
        time.sleep(0.5)


reset_servos()


# =================================================================
# 2. MECHANICAL SORTING & DUMPING SEQUENCE
# =================================================================
def sort_and_dump(category: str):
    if not pi_5_factory:
        print(f"[SKIP] Servo hardware unavailable; skipping actuation for {category}.")
        return

    print(f"\n>>> Actuation triggered! Target compartment: {category}")

    if category == "PLASTIC":
        rotate_servo.min()
    elif category == "METAL":
        rotate_servo.mid()
    elif category == "BIO":
        rotate_servo.max()

    time.sleep(1.5)

    # Tilt lid to release waste into target compartment
    tilt_servo.min()
    time.sleep(2.0)

    # Return lid to flat position
    tilt_servo.mid()
    time.sleep(1.0)

    # Reset servos back to standby state
    reset_servos()
    print("[INFO] System reset and ready for next item.\n")


# =================================================================
# 3. MODEL DEFINITIONS & TFLITE INTERPRETERS
# =================================================================
CLASS_NAMES_MODEL1 = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_NAMES_MODEL2 = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL1_PATH = BASE_DIR / "models" / "garbage_resnet50.tflite"
MODEL2_PATH = BASE_DIR / "models" / "garbage_efficientnet.tflite"

print("Loading TensorFlow Lite models...")
try:
    interpreter1 = tf.lite.Interpreter(model_path=str(MODEL1_PATH))
    interpreter1.allocate_tensors()
    input_details1 = interpreter1.get_input_details()
    output_details1 = interpreter1.get_output_details()

    interpreter2 = tf.lite.Interpreter(model_path=str(MODEL2_PATH))
    interpreter2.allocate_tensors()
    input_details2 = interpreter2.get_input_details()
    output_details2 = interpreter2.get_output_details()

    print("[INFO] Both TFLite models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load TFLite models: {e}")
    raise SystemExit(1)


# =================================================================
# 4. CAMERA & MOTION SEGMENTATION LOOP
# =================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    raise SystemExit(1)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)

last_printed_class = ""
consecutive_detections = 0
last_detection = None
STABILITY_THRESHOLD = 4  # Trigger mechanism after 4 consecutive matching frames

print("\n--- Smart Waste AI Edge System Active ---")
print("Press 'q' in the video window to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    default_size = int(min(h, w) * 0.7)
    x, y, width, height = (w - default_size) // 2, (h - default_size) // 2, default_size, default_size

    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 8000]

    if large_contours:
        best_contour = max(large_contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(best_contour)
        box_size = max(width, height)
        x = max(0, min(x, w - box_size))
        y = max(0, min(y, h - box_size))
        width = height = box_size

    crop_frame = frame[y:y + height, x:x + width]

    if crop_frame.size > 0:
        input_image = cv2.resize(crop_frame, (224, 224))

        # Model 1 (ResNet50) inference with ImageNet channel mean subtraction
        img_rgb1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_rgb1[:, :, 0] -= 103.939
        img_rgb1[:, :, 1] -= 116.779
        img_rgb1[:, :, 2] -= 123.68
        input_data1 = np.expand_dims(img_rgb1, axis=0)

        interpreter1.set_tensor(input_details1[0]["index"], input_data1)
        interpreter1.invoke()
        output1 = interpreter1.get_tensor(output_details1[0]["index"]).flatten()

        idx1 = int(np.argmax(output1))
        class1 = CLASS_NAMES_MODEL1[idx1]
        conf1 = float(100.0 * output1[idx1])

        # Model 2 (EfficientNetB0) inference
        img_rgb2 = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_data2 = np.expand_dims(img_rgb2, axis=0)

        interpreter2.set_tensor(input_details2[0]["index"], input_data2)
        interpreter2.invoke()
        output2 = interpreter2.get_tensor(output_details2[0]["index"]).flatten()

        idx2 = int(np.argmax(output2))
        class2 = CLASS_NAMES_MODEL2[idx2]
        conf2 = float(100.0 * output2[idx2])

        # Fusion decision rules for target categories
        final_class = None
        final_conf = 0.0

        if class2 == "biological" and conf2 >= 85.0:
            final_class = "BIO"
            final_conf = conf2
        elif class1 == "metal" and conf1 >= 70.0:
            final_class = "METAL"
            final_conf = conf1
        elif class1 == "plastic" and conf1 >= 85.0:
            final_class = "PLASTIC"
            final_conf = conf1
        elif class2 == "plastic" and conf2 >= 85.0:
            final_class = "PLASTIC"
            final_conf = conf2
        elif class2 == "metal" and conf2 >= 70.0:
            final_class = "METAL"
            final_conf = conf2

        if final_class is not None:
            status_text = f"CLASS: {final_class} ({final_conf:.1f}%)"
            box_color = (0, 255, 0)

            if final_class == last_detection:
                consecutive_detections += 1
            else:
                last_detection = final_class
                consecutive_detections = 1

            if final_class != last_printed_class:
                print(f"[FUSION] -> {final_class} ({final_conf:.1f}%)")
                last_printed_class = final_class
        else:
            status_text = "SEARCHING... (METAL / PLASTIC / BIO)"
            box_color = (0, 165, 255)
            last_printed_class = ""
            consecutive_detections = 0
            last_detection = None
    else:
        status_text = "NO OBJECT DETECTED"
        box_color = (0, 0, 255)
        consecutive_detections = 0
        last_detection = None

    if consecutive_detections >= STABILITY_THRESHOLD and last_detection is not None:
        cv2.putText(
            frame, "SORTING...", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )
        cv2.imshow("Smart Waste AI Edge System", frame)
        cv2.waitKey(1)

        sort_and_dump(last_detection)

        consecutive_detections = 0
        last_detection = None
        for _ in range(15):
            cap.read()

    cv2.rectangle(frame, (x, y), (x + width, y + height), box_color, 2)
    cv2.rectangle(frame, (10, 10), (550, 45), (0, 0, 0), -1)
    cv2.putText(
        frame, status_text, (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
    )

    cv2.imshow("Smart Waste AI Edge System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if pi_5_factory:
    tilt_servo.detach()
    rotate_servo.detach()
print("\nSystem stopped.")
