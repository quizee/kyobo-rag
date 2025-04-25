import os
import json
import fitz  # PyMuPDF
from PIL import Image
import numpy as np


class ImageExtractor:
    def __init__(self):
        self.output_dir = "data/extracted/images"
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_images_from_pdf(self, pdf_path: str, json_path: str):
        """PDF 파일에서 JSON 파일의 좌표에 맞춰 이미지를 추출"""
        try:
            # JSON 파일 읽기
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # PDF 파일 열기
            pdf_document = fitz.open(pdf_path)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]

            # 각 페이지의 이미지 정보 처리
            for page_data in json_data.get("pages", []):
                page_num = page_data.get("page", 1) - 1  # 0-based index로 변환
                page = pdf_document[page_num]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # 페이지의 모든 이미지 처리
                for image_info in page_data.get("images", []):
                    try:
                        # 원본 이미지 크기와 PDF 페이지 크기의 비율 계산
                        scale_x = pix.width / page_data.get(
                            "width", 595.276
                        )  # PDF 표준 너비
                        scale_y = pix.height / page_data.get(
                            "height", 841.89
                        )  # PDF 표준 높이

                        # 좌표 변환
                        x = int(image_info.get("x", 0) * scale_x)
                        y = int(image_info.get("y", 0) * scale_y)
                        width = int(image_info.get("width", 0) * scale_x)
                        height = int(image_info.get("height", 0) * scale_y)

                        # 이미지 크롭
                        cropped_img = img.crop((x, y, x + width, y + height))

                        # 파일명 생성 (이미지 타입 포함)
                        image_name = image_info.get("name", "").replace(".jpg", "")
                        image_type = image_info.get("type", "unknown")
                        output_filename = f"{base_name}_{image_name}_{image_type}.png"
                        output_path = os.path.join(self.output_dir, output_filename)

                        # 이미지 저장
                        cropped_img.save(output_path, "PNG")
                        print(f"Saved: {output_path}")

                    except Exception as crop_error:
                        print(
                            f"Error cropping image: {str(crop_error)}, Coordinates: x={x}, y={y}, w={width}, h={height}"
                        )

            pdf_document.close()

        except Exception as e:
            print(f"Error extracting images: {str(e)}")


def main():
    extractor = ImageExtractor()

    # 특정 PDF 파일만 처리
    pdf_name = "1.교보마이플랜건강보험 [2409](무배당).pdf"
    pdf_path = os.path.join("data/상품설명", pdf_name)
    json_path = os.path.join(
        "data/상품설명", f"{os.path.splitext(pdf_name)[0]}.pdf.json"
    )

    if os.path.exists(json_path):
        print(f"Processing: {pdf_name}")
        extractor.extract_images_from_pdf(pdf_path, json_path)
    else:
        print(f"JSON file not found for: {pdf_name}")


if __name__ == "__main__":
    main()
