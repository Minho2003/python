import cv2
import numpy as np
from mss import mss
from datetime import datetime
import csv
import os
from ultralytics import YOLO

# 컴퓨터 화면 감지

# YOLO 모델 불러오기
model = YOLO("yolov8n.pt")

# 캡처할 화면 영역 (전체 화면 또는 일부)
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
sct = mss()

# CSV 저장 경로 설정
output_dir = "/Users/minho/Desktop/python/outputs"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "test_3_output.csv")

# CSV 파일 열기 및 세션 헤더 작성
is_new_file = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as log_file:
    writer = csv.writer(log_file)
    if is_new_file or os.stat(csv_path).st_size == 0:
        writer.writerow(["Time", "Class", "Confidence", "x1", "y1", "x2", "y2"])
        writer.writerow(["--- Session started at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " ---"])

    while True:
        # 화면 캡처 및 프레임 변환
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        results = model.predict(source=frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{now}] Class: {r.names[cls]}, Confidence: {conf:.2f}, "
                      f"Location: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
                writer.writerow([now, r.names[cls], round(conf, 2), int(x1), int(y1), int(x2), int(y2)])

            # 시각화
            annotated_frame = r.plot()
            cv2.imshow("Screen Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()