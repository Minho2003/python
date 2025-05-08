from datetime import datetime
from ultralytics import YOLO
import cv2
import os
import csv

# 웹캠으로 실시간 감지 및 csv에 결과 저장

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

# CSV 저장 경로 설정
output_dir = "/Users/minho/Desktop/python/outputs"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "test_2_output.csv")

# CSV 파일 열기 및 추가 기록
is_new_file = not os.path.exists(csv_path)

with open(csv_path, "a", newline="") as log_file:
    writer = csv.writer(log_file)
    if is_new_file or os.stat(csv_path).st_size == 0:
        writer.writerow(["Time", "Class", "Confidence", "x1", "y1", "x2", "y2"])
        writer.writerow(["--- Session started at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " ---"])

    # 웹캠에서 사람만 실시간 감지
    results = model.predict(source=0, stream=True, classes=[0])  # 클래스 0 = person

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now}] Class: {r.names[cls]}, Confidence: {conf:.2f}, "
                  f"Location: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
            writer.writerow([now, r.names[cls], round(conf, 2), int(x1), int(y1), int(x2), int(y2)])

            # 시각화 표시
            frame = r.plot()
            cv2.imshow("YOLOv8 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()