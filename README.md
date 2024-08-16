<h1>KEB_TEAM_PROJECT (2024.08)</h1>
<div align="center">
  <img width="550" alt="logo" src="https://github.com/user-attachments/assets/ce0ec293-d067-48e1-96bf-866d22f40edc">
</div>

---
## 한국 차량 번호판 인식 및 번호 인식 모델 
---
## 사용 모델
[YOLOv10: Real-Time End-to-End Object Detection](https://github.com/THU-MIG/yolov10)
![333755285-f9b1bec0-928e-41ce-a205-e12db3c4929a](https://github.com/user-attachments/assets/2699db02-6fe9-405a-9fb4-163de80efbd9)
[PaddleOCR PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR)
![image](https://github.com/user-attachments/assets/f4a892c0-3b18-4db8-b21d-62079cc7fc08)

---
## 프로젝트 구성
- 데이터 수집: 다양한 환경에서 차량 번호판 이미지를 수집.
- 데이터 전처리: 이미지 정제, 증강 및 라벨링.
- 모델 개발: YOLOv10s 모델을 사용하여 번호판과 번호를 검출 후 PP-OCRv3로 인식 후 텍스트 변환
- 서비스 구축: 번호판 인식 API 개발 및 사용자 인터페이스 구현.
---
### 데이터 수집 예시)

<img width="401" alt="데이터 수집 예시" src="https://github.com/user-attachments/assets/b8f81a3b-2449-4412-86c8-99c17bb06cc6">

### 데이터 라벨링 예시)

<img width="405" alt="데이터 라벨링 예시" src="https://github.com/user-attachments/assets/071bcfbe-252c-4ff6-a412-004b3880d363">

### 모델 학습 후 예측 예시)

<img width="500" alt="모델 학습 후 예측 예시" src="https://github.com/user-attachments/assets/f8a021f8-75e1-4d55-8195-bd0fa9b925bd">

---

<div align=Left><h2>📚STACKS</h2></div>
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
<img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> 
<img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"> 
<img src="https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white"> 


---
## 
## dataset 출처
[AI-hub CCTV 기반 차량정보 및 교통정보 계측 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71573)
![image](https://github.com/user-attachments/assets/71ad06bd-8bdf-411e-b145-e73d850657c6)

[RoboFlow](https://universe.roboflow.com/university-of-toronto-xho85/numberdetection-eppfj)
![example](https://github.com/user-attachments/assets/9181b5d5-5885-47ef-ac89-17074040f903)
