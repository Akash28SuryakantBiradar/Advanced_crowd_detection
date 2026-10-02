import cv2
import time
from ultralytics import YOLO
from analytics.movement import MovementAnalyzer
from safety.risk_engine import RiskEngine


# ==============================
# INITIALIZATION
# ==============================

model = YOLO("yolo11n.pt")

movement_analyzer = MovementAnalyzer()
risk_engine = RiskEngine()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

prev_time = time.time()


print("====================================")
print(" ADVANCED CROWD SAFETY SYSTEM")
print(" YOLO11 + BYTETRACK")
print(" MOVEMENT + RISK ANALYSIS")
print("====================================")
print("Press Q to quit.")


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame read failed.")
        break


    # ==============================
    # YOLO + BYTETRACK
    # ==============================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.30,
        verbose=False
    )

    result = results[0]


    # ==============================
    # PERSON COUNT
    # ==============================

    if result.boxes is not None:
        people_count = len(result.boxes)
    else:
        people_count = 0


    # ==============================
    # CROWD DENSITY
    # ==============================

    if people_count <= 10:
        density = "LOW"

    elif people_count <= 25:
        density = "MEDIUM"

    elif people_count <= 50:
        density = "HIGH"

    else:
        density = "EXTREME"


    # ==============================
    # MOVEMENT ANALYSIS
    # ==============================

    movement_data = []

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            speed, direction = movement_analyzer.analyze(
                track_id,
                center_x,
                center_y
            )

            movement_data.append(
                (track_id, speed, direction)
            )


    # ==============================
    # RISK ANALYSIS
    # ==============================

    risk_level, alerts = risk_engine.analyze(
        people_count,
        density,
        movement_data
    )

    alert_text = alerts[0]


    # ==============================
    # DRAW DETECTIONS
    # ==============================

    output = result.plot()


    # ==============================
    # FPS
    # ==============================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time


    # ==============================
    # INFORMATION DISPLAY
    # ==============================

    cv2.putText(
        output,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Density: {density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        output,
        "YOLO11 + ByteTrack",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ==============================
    # MOVEMENT DISPLAY
    # ==============================

    if movement_data:

        track_id, speed, direction = movement_data[0]

        cv2.putText(
            output,
            f"ID {track_id}: {direction} | Speed: {speed:.1f}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )


    # ==============================
    # RISK DISPLAY
    # ==============================

    cv2.putText(
        output,
        f"Risk: {risk_level}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    # ==============================
    # ALERT DISPLAY
    # ==============================

    cv2.putText(
        output,
        f"Alert: {alert_text}",
        (20, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )


    # ==============================
    # SHOW WINDOW
    # ==============================

    cv2.imshow(
        "Advanced Crowd Safety System",
        output
    )


    # ==============================
    # QUIT
    # ==============================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()