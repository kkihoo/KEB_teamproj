import json
import os
from pathlib import Path


def prepare_data(json_folder, output_dir, image_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_list = []

    # JSON 폴더 내의 모든 JSON 파일 처리
    for json_file in Path(json_folder).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 각 JSON 파일이 리스트가 아니라 딕셔너리인 경우 리스트로 변환
        if isinstance(data, dict):
            data = [data]

        for item in data:
            image_path = os.path.join(image_dir, item["imagePath"])
            label = item["value"]

            # PaddleOCR 형식으로 변환
            line = f"{image_path}\t{label}\n"
            train_list.append(line)

    # 학습 데이터 파일 생성
    with open(output_dir / "list.txt", "w", encoding="utf-8") as f:
        f.writelines(train_list)

    print(f"데이터 준비 완료. 파일 저장 위치: {output_dir / 'list.txt'}")


# 사용 예
json_folder = "C:/Users/kimki/Desktop/KEB/train_data/rec/valid/labels"
output_dir = "C:/Users/kimki/Desktop/KEB/train_data/rec/valid/"
image_dir = "C:/Users/kimki/Desktop/KEB/train_data/rec/valid/images"
prepare_data(json_folder, output_dir, image_dir)
