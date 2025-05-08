import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ultralytics as ul
from ultralytics import YOLO

# 웹캠으로 감지


# 모델 불러오기 (YOLOv8n: nano, YOLOv8s: small, YOLOv8m, YOLOv8l, YOLOv8x: extra large)
model = YOLO("../yolov8n.pt")  # 사전학습된 모델 사용

# 이미지에 대해 예측 수행
results = model.predict(
    source=0,
    classes=[0],
    show=True,
    save=True,
    project='/Users/minho/Desktop/python/outputs/',
    name='webcam-person-detect'
)  # source 0= 내 웹캠 , classes 탐지할 id (person=0), show 실시간으로 보여줌, save 결과값 저장, project와 name으로 결과 디렉토리 지정
