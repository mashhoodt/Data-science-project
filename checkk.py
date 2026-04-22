import cv2
import time
from ultralytics import YOLO

# ==============================
# Load YOLO Model
# ==============================
model = YOLO(r"D:\Mashhood\training\best.pt")

# Detection threshold
threshold = 0.5

# ==============================
# Start Webcam
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot open webcam")
    exit()

# FPS calculation
prev_time = 0

print("✅ Press 'q' to exit")

# ==============================
# Main Loop
# ==============================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    # ==============================
    # YOLO Detection
    # ==============================
    results = model(frame)[0]

    object_count = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            object_count += 1
            label = results.names[int(class_id)]

            # Draw bounding box
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0),
                          3)

            # Draw label
            cv2.putText(frame,
                        f"{label} {score:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

            print(label)

    # ==============================
    # FPS Calculation
    # ==============================
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # ==============================
    # Display Info on Screen
    # ==============================
    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    cv2.putText(frame,
                f"Objects: {object_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    # ==============================
    # Show Frame
    # ==============================
    cv2.imshow("Live Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()