import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from paddleocr import PaddleOCR

model_dir = "PaddleOCR/inference/rec_license_plate"

ocr = PaddleOCR(lang="korean", use_angle_cls=True, rec_model_dir=model_dir)

model = YOLO("최종_v10s.pt")
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "License Plate"]

allowed_korean_chars = set(
    "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주바사아자배하허호육공해국합"
)

# 문자 매핑
char_mapping = {
    # "에": "어",
}


def enhance_contrast(image):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image


def adaptive_threshold(image):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary_image


def predict_license_plate(image):
    image_np = np.array(image)
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    results = model.predict(image_np, conf=0.4)

    annotations = []
    for result in results:
        for detection in result.boxes:
            bbox = detection.xyxy[0].tolist()
            confidence = detection.conf[0].item()
            class_id = int(detection.cls[0].item())
            class_name = class_names[class_id]

            annotations.append(
                {"bbox": bbox, "confidence": confidence, "class_name": class_name}
            )

    return annotations


def extract_korean_char(ocr_text):
    korean_chars = [char for char in ocr_text if char in allowed_korean_chars]
    if korean_chars:
        return korean_chars[0]

    for char in ocr_text:
        if char in char_mapping:
            return char_mapping[char]

    return ""  # 적절한 문자를 찾지 못한 경우


def classify_vehicle_type(first_part):
    # Convert the first part of the license plate number to an integer
    first_num = int(first_part)

    # Classify vehicle type based on the first part
    if len(first_part) == 2:
        # 6-digit license plates
        if 1 <= first_num <= 69:
            return "승용차"
        elif 70 <= first_num <= 79:
            return "승합차"
        elif 80 <= first_num <= 97:
            return "화물차"
        elif 98 <= first_num <= 99:
            return "특수차"
    elif len(first_part) == 3:
        # 7-digit license plates
        if 100 <= first_num <= 699:
            return "승용차"
        elif 700 <= first_num <= 799:
            return "승합차"
        elif 800 <= first_num <= 979:
            return "화물차"
        elif 980 <= first_num <= 997:
            return "특수차"
        elif 998 <= first_num <= 999:
            return "긴급차"

    return "알 수 없는 종류"


def draw_annotations(image, annotations):
    draw = ImageDraw.Draw(image)
    license_plate_bbox = None
    detected_numbers = []

    for ann in annotations:
        bbox = ann["bbox"]
        confidence = ann["confidence"]
        class_name = ann["class_name"]

        if class_name == "License Plate":
            license_plate_bbox = bbox
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="blue", width=3
            )
        elif class_name.isdigit():
            detected_numbers.append((int(class_name), bbox[0], confidence))
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=3
            )
            label = f"{class_name} {confidence:.2f}"
            font = ImageFont.load_default()
            bbox_text = draw.textbbox((bbox[0], bbox[1]), label, font=font)
            w, h = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
            draw.rectangle((bbox[0], bbox[1] - h, bbox[0] + w, bbox[1]), fill="red")
            draw.text((bbox[0], bbox[1] - h), label, fill="white", font=font)

    if license_plate_bbox and detected_numbers:
        detected_numbers.sort(key=lambda x: x[1])
        yolo_number = "".join([str(num[0]) for num in detected_numbers])

        cropped_plate = image.crop(
            (
                license_plate_bbox[0],
                license_plate_bbox[1],
                license_plate_bbox[2],
                license_plate_bbox[3],
            )
        )
        cropped_plate_arr = np.array(cropped_plate)
        enhanced_image = enhance_contrast(cropped_plate_arr)
        binary_image = adaptive_threshold(enhanced_image)

        results = ocr.ocr(binary_image, cls=False, det=False)

        if results:
            ocr_text = results[0][0][0]
            ocr_confidence = results[0][0][1]

            korean_char = extract_korean_char(ocr_text)

            front_numbers = (
                yolo_number[:2] if len(yolo_number) <= 6 else yolo_number[:3]
            )
            back_numbers = yolo_number[-4:]
            final_number = f"{front_numbers}{back_numbers}"  # {korean_char}
            vehicle_type = classify_vehicle_type(front_numbers)

            # print("YOLO 탐지 번호:", yolo_number)
            # print("OCR 인식 텍스트:", ocr_text)
            # print("추출된 한글 문자:", korean_char)
            print("차량 번호:", final_number)
            print("차량 종류:", vehicle_type)

            label = final_number
            font = ImageFont.load_default()
            bbox_text = draw.textbbox(
                (license_plate_bbox[0], license_plate_bbox[1]), label, font=font
            )
            w, h = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
            draw.rectangle(
                (
                    license_plate_bbox[0],
                    license_plate_bbox[1] - h,
                    license_plate_bbox[0] + w,
                    license_plate_bbox[1],
                ),
                fill="blue",
            )
            draw.text(
                (license_plate_bbox[0], license_plate_bbox[1] - h),
                label,
                fill="white",
                font=font,
            )

    return image


def process_image(image_path):
    image = Image.open(image_path)
    annotations = predict_license_plate(image)
    annotated_image = draw_annotations(image.copy(), annotations)
    annotated_image.show()


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotations = predict_license_plate(image)
        annotated_image = draw_annotations(image.copy(), annotations)
        frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main(input_path):
    if not os.path.exists(input_path):
        print(f"File {input_path} does not exist.")
        return

    if input_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        process_image(input_path)
    elif input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
        process_video(input_path)
    else:
        print("Unsupported file type.")


if __name__ == "__main__":
    input_path = "realtest/carnum4.jpg"
    main(input_path)
