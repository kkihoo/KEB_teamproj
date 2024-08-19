import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ocr = PaddleOCR(use_angle_cls=True, lang="korean")

model = YOLO("최종_v10s.pt")

class_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "License_Plate",
]


def enhance_image(image):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced


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


def classify_vehicle_type(first_part):
    first_num = int(first_part)

    if len(first_part) == 2:
        if 1 <= first_num <= 69:
            return "승용차"
        elif 70 <= first_num <= 79:
            return "승합차"
        elif 80 <= first_num <= 97:
            return "화물차"
        elif 98 <= first_num <= 99:
            return "특수차"
    elif len(first_part) == 3:
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


def draw_annotations(image, annotations, license_plate_number):
    draw = ImageDraw.Draw(image)
    license_plate_bbox = None

    for ann in annotations:
        bbox = ann["bbox"]
        confidence = ann["confidence"]
        class_name = ann["class_name"]

        if class_name == "License_Plate":
            license_plate_bbox = bbox
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="blue", width=3
            )

    if license_plate_bbox and license_plate_number:
        font = ImageFont.load_default()
        bbox_text = draw.textbbox(
            (license_plate_bbox[0], license_plate_bbox[1]),
            license_plate_number,
            font=font,
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
            license_plate_number,
            fill="white",
            font=font,
        )

    return image


def merge_results_korean_format(yolo_result, ocr_result):
    def extract_numbers(s):
        return "".join(filter(str.isdigit, s))

    def extract_hangul(s):
        return "".join(
            filter(
                lambda x: x
                in "보부다더도두어가거고구하호저조주라러버로루마머모무나너노누오서소수우",
                s,
            )
        )

    yolo_numbers = extract_numbers(yolo_result)
    yolo_hangul = extract_hangul(yolo_result)
    ocr_numbers = extract_numbers(ocr_result)
    ocr_hangul = extract_hangul(ocr_result)

    if len(yolo_numbers) == 6:
        prefix = yolo_numbers[:2]
        hangul = (
            yolo_hangul[0] if yolo_hangul else (ocr_hangul[0] if ocr_hangul else "")
        )
        suffix = yolo_numbers[-4:]

    elif len(yolo_numbers) == 7:
        prefix = yolo_numbers[:3]
        hangul = (
            yolo_hangul[0] if yolo_hangul else (ocr_hangul[0] if ocr_hangul else "")
        )
        suffix = yolo_numbers[-4:]

    else:
        return yolo_result if len(yolo_result) > len(ocr_result) else ocr_result

    final_result = f"{prefix}{hangul}{suffix}"

    if len(final_result) < 7:
        return yolo_result if len(yolo_result) > len(ocr_result) else ocr_result

    return final_result


def process_image(image_path):
    image = Image.open(image_path)
    annotations = predict_license_plate(image)

    license_plate_bbox = None
    detected_chars = []

    for ann in annotations:
        if ann["class_name"] == "License_Plate":
            license_plate_bbox = ann["bbox"]
        elif ann["class_name"] in class_names:
            detected_chars.append(
                (ann["class_name"], ann["bbox"][0], ann["confidence"])
            )

    if license_plate_bbox and detected_chars:
        detected_chars.sort(key=lambda x: x[1])
        yolo_result = "".join([char[0] for char in detected_chars])

        cropped_plate = image.crop(
            (
                license_plate_bbox[0],
                license_plate_bbox[1],
                license_plate_bbox[2],
                license_plate_bbox[3],
            )
        )
        cropped_plate_arr = np.array(cropped_plate)
        enhanced_image = enhance_image(cropped_plate_arr)

        ocr_results = ocr.ocr(enhanced_image, cls=True)
        ocr_result = ""
        if ocr_results and isinstance(ocr_results, list) and len(ocr_results) > 0:
            if isinstance(ocr_results[0], list) and len(ocr_results[0]) > 0:
                ocr_text = " ".join(
                    [
                        result[1][0]
                        for result in ocr_results[0]
                        if isinstance(result, (list, tuple)) and len(result) > 1
                    ]
                )
                ocr_result = "".join([char for char in ocr_text if char.isalnum()])

        print("YOLO 인식 결과:", yolo_result)
        print("OCR 인식 결과:", ocr_result)

        if len(yolo_result) >= 6:
            final_result = yolo_result
        elif ocr_result:
            final_result = merge_results_korean_format(yolo_result, ocr_result)
        else:
            final_result = yolo_result

        if len(final_result) >= 6:
            vehicle_type = classify_vehicle_type(
                final_result[:3] if len(final_result) > 7 else final_result[:2]
            )

            print("차량 번호:", final_result)
            print("차량 종류:", vehicle_type)

            annotated_image = draw_annotations(image.copy(), annotations, final_result)
        else:
            print("유효한 차량 번호를 추출하지 못했습니다.")
            annotated_image = image.copy()
    else:
        print("번호판 또는 문자를 탐지하지 못했습니다.")
        annotated_image = image.copy()

    # 결과 저장 코드는 그대로 유지
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image_path = os.path.join(output_dir, f"result_{timestamp}.png")

    if annotated_image.mode == "RGBA":
        annotated_image = annotated_image.convert("RGB")

    annotated_image.save(output_image_path)
    print(f"결과 이미지가 저장되었습니다: {output_image_path}")

    return annotated_image


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"results_{timestamp}.txt")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(output_dir, f"output_{timestamp}.mp4"),
        fourcc,
        fps,
        (width, height),
    )

    with open(output_file, "w", encoding="utf-8") as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0:
                out.write(frame)
                continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotations = predict_license_plate(image)

            license_plate_bbox = None
            detected_chars = []

            for ann in annotations:
                if ann["class_name"] == "License_Plate":
                    license_plate_bbox = ann["bbox"]
                elif ann["class_name"] in class_names:
                    detected_chars.append(
                        (ann["class_name"], ann["bbox"][0], ann["confidence"])
                    )

            if license_plate_bbox and detected_chars:
                detected_chars.sort(key=lambda x: x[1])
                yolo_result = "".join([char[0] for char in detected_chars])

                cropped_plate = image.crop(
                    (
                        license_plate_bbox[0],
                        license_plate_bbox[1],
                        license_plate_bbox[2],
                        license_plate_bbox[3],
                    )
                )
                cropped_plate_arr = np.array(cropped_plate)
                enhanced_image = enhance_image(cropped_plate_arr)

                ocr_results = ocr.ocr(enhanced_image, cls=True)

                if ocr_results:
                    ocr_text = " ".join([result[1][0] for result in ocr_results[0]])
                    ocr_result = "".join([char for char in ocr_text if char.isalnum()])

                    result_text = f"Frame {frame_count}:\n"
                    result_text += f"YOLO 인식 결과: {yolo_result}\n"
                    result_text += f"OCR 인식 결과: {ocr_result}\n"

                    final_result = merge_results_korean_format(yolo_result, ocr_result)

                    if len(final_result) >= 7:
                        vehicle_type = classify_vehicle_type(
                            final_result[:3]
                            if len(final_result) > 7
                            else final_result[:2]
                        )

                        result_text += f"차량 번호: {final_result}\n"
                        result_text += f"차량 종류: {vehicle_type}\n"

                        cv2.rectangle(
                            frame,
                            (int(license_plate_bbox[0]), int(license_plate_bbox[1])),
                            (int(license_plate_bbox[2]), int(license_plate_bbox[3])),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"{final_result} ({vehicle_type})",
                            (
                                int(license_plate_bbox[0]),
                                int(license_plate_bbox[1]) - 10,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        result_text += "유효한 차량 번호를 추출하지 못했습니다.\n"
                else:
                    result_text = (
                        f"Frame {frame_count}: OCR이 번호판을 인식하지 못했습니다.\n"
                    )
                    result_text += f"YOLO 인식 결과: {yolo_result}\n"
            else:
                result_text = (
                    f"Frame {frame_count}: 번호판 또는 문자를 탐지하지 못했습니다.\n"
                )

            result_text += "---\n"
            print(result_text)
            f.write(result_text)

            out.write(frame)

    cap.release()
    out.release()
    print(f"Results saved to {output_file}")
    print(
        f"Output video saved to {os.path.join(output_dir, f'output_{timestamp}.mp4')}"
    )


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
    input_path = "realtest/carnum9.jpg"
    main(input_path)
