import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import csv
import os
from datetime import datetime


MODEL_PATH = r"C:\Users\Sudarshan\PycharmProjects\WasteClassification\.venv\MobileNetV2_model.keras"   # your trained model
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CONF_THRESHOLD = 0.90
CONSEC_FRAMES_REQUIRED = 3
RELEASE_FRAMES = 10
CSV_FOLDER = "counts_csv"


os.makedirs(CSV_FOLDER, exist_ok=True)

# Load model
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded!")

# Get model input size
try:
    ishape = model.input_shape
    IMG_H, IMG_W = ishape[1], ishape[2]
    if IMG_H is None or IMG_W is None:
        IMG_H, IMG_W = 224, 224
except Exception:
    IMG_H, IMG_W = 224, 224
print("[INFO] Model input size:", (IMG_W, IMG_H))

# Open webcam
print("[INFO] Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Could not open webcam!")
    exit()

cv2.namedWindow("LIVE", cv2.WINDOW_NORMAL)

# counting state
counts = {c: 0 for c in CLASS_NAMES}

# per-class state for consecutive frames and release logic
consec_high = {c: 0 for c in CLASS_NAMES}
consec_low = {c: 0 for c in CLASS_NAMES}
locked = {c: False for c in CLASS_NAMES}

last_count_time = {c: 0.0 for c in CLASS_NAMES}

# CSV helper
def append_to_csv(row, filename=None):
    """Append row (list) to CSV. If filename None, create one with timestamp."""
    if filename is None:
        fname = datetime.now().strftime("counts_%Y%m%d_%H%M%S.csv")
        filename = os.path.join(CSV_FOLDER, fname)
        write_header = True
    else:
        write_header = not os.path.exists(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp_iso", "label", "confidence", "class_count_total"])
        writer.writerow(row)
    return filename

current_csv_file = None

print("[INFO] Controls: 's' = save CSV snapshot, 'r' = reset counts, 'q' = quit")
print(f"[INFO] Counting only when confidence >= {CONF_THRESHOLD*100:.0f}%")
print(f"[INFO] Requires {CONSEC_FRAMES_REQUIRED} consecutive frames to count; release after {RELEASE_FRAMES} low-confidence frames.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Preprocess frame for model
        img = cv2.resize(frame, (IMG_W, IMG_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
        img = preprocess_input(img)
        img = np.expand_dims(img, 0)

        preds = model.predict(img, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx]
        conf = float(preds[idx])

        incremented_label = None

        # Update per-class consecutive counters
        for cls in CLASS_NAMES:
            cls_idx = CLASS_NAMES.index(cls)
            cls_conf = float(preds[cls_idx])

            if cls_conf >= CONF_THRESHOLD:
                consec_high[cls] += 1
                consec_low[cls] = 0
            else:
                consec_low[cls] += 1
                consec_high[cls] = 0

            # If class is locked (recently counted), wait for release frames of low-confidence to unlock
            if locked[cls]:
                if consec_low[cls] >= RELEASE_FRAMES:
                    locked[cls] = False
                    consec_high[cls] = 0
                    consec_low[cls] = 0


        # Count logic: only consider the top predicted label for counting
        if conf >= CONF_THRESHOLD and not locked[label]:
            if consec_high[label] >= CONSEC_FRAMES_REQUIRED:
                counts[label] += 1
                last_count_time[label] = time.time()
                locked[label] = True
                consec_high[label] = 0
                consec_low[label] = 0
                incremented_label = label

                if current_csv_file is None:
                    current_csv_file = append_to_csv([datetime.now().isoformat(), label, f"{conf:.4f}", counts[label]])
                else:
                    append_to_csv([datetime.now().isoformat(), label, f"{conf:.4f}", counts[label]], current_csv_file)

        # Display overlay text
        if conf >= CONF_THRESHOLD:
            text = f"{label} ({conf*100:.1f}%)"
            text_color = (0, 255, 0)
        else:
            text = f"{label} ({conf*100:.1f}%) LOW"
            text_color = (0, 140, 255)

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        # Draw counts
        start_x = frame.shape[1] - 320
        start_y = 30
        line_h = 26
        cv2.rectangle(frame, (start_x-10, 0), (frame.shape[1], 220), (0,0,0), -1)
        i = 0
        for cls in CLASS_NAMES:
            lock_mark = " (L)" if locked[cls] else ""
            count_text = f"{cls}: {counts[cls]}{lock_mark}"
            cv2.putText(frame, count_text, (start_x, start_y + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            i += 1

        # indicate if incremented this frame
        if incremented_label is not None:
            cv2.putText(frame, f"COUNTED: {incremented_label}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

        # show threshold / settings
        cv2.putText(frame, f"Threshold: {int(CONF_THRESHOLD*100)}%  ReqFrames: {CONSEC_FRAMES_REQUIRED}  Release: {RELEASE_FRAMES}",
                    (10, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("LIVE", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            summary_file = os.path.join(CSV_FOLDER, datetime.now().strftime("summary_%Y%m%d_%H%M%S.csv"))
            with open(summary_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["class", "total_count"])
                for cls in CLASS_NAMES:
                    writer.writerow([cls, counts[cls]])
            print(f"[INFO] Summary saved to {summary_file}")
            break
        elif key == ord('s'):
            if current_csv_file is None:
                current_csv_file = os.path.join(CSV_FOLDER, datetime.now().strftime("counts_%Y%m%d_%H%M%S.csv"))
            for cls in CLASS_NAMES:
                append_to_csv([datetime.now().isoformat(), cls, "", counts[cls]], current_csv_file)
            print(f"[INFO] Saved snapshot to {current_csv_file}")
        elif key == ord('r'):
            counts = {c: 0 for c in CLASS_NAMES}
            consec_high = {c: 0 for c in CLASS_NAMES}
            consec_low = {c: 0 for c in CLASS_NAMES}
            locked = {c: False for c in CLASS_NAMES}
            current_csv_file = None
            print("[INFO] Counts and state reset.")

finally:
    cap.release()
    cv2.destroyAllWindows()
