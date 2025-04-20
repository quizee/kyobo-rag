import os
import logging
import json
import pdfplumber
import base64
import io
from dataclasses import dataclass
from typing import List, Tuple, Optional
from openai import OpenAI
from PIL import Image
import fitz
import traceback
import shutil

logging.basicConfig(level=logging.INFO)


@dataclass
class VisualSection:
    title: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    section_type: str
    description: Optional[str] = None


class PDFProcessor:
    def __init__(self, test_mode=False, test_page_limit=3):
        self.client = OpenAI()
        self.test_mode = test_mode
        self.test_page_limit = test_page_limit
        self.model = "gpt-4.1-mini"
        self.product_info = None
        self.logger = logging.getLogger(__name__)
        self.pdf_path = None

        # 출력 디렉토리 설정
        self.base_output_dir = "data/extracted"
        self.image_dir = os.path.join(self.base_output_dir, "images")
        self.text_dir = os.path.join(self.base_output_dir, "text")
        self.image_output_dir = self.image_dir
        self.text_output_dir = self.text_dir

        # 출력 디렉토리 생성
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)

        self.vision_system_prompt = """당신은 PDF 문서의 시각적 구조를 분석하는 전문가입니다.
페이지의 각 섹션을 분석하여 다음 단계로 처리해주세요:

1. 먼저 섹션의 모든 텍스트를 있는 그대로 추출합니다.
2. 추출된 텍스트의 형식(text/list/table/box/image)을 파악합니다.
3. 추출된 텍스트를 바탕으로 섹션의 내용을 잘 설명하는 제목을 생성합니다.

다음과 같은 JSON 형식으로만 응답하세요:

{
    "sections": [
        {
            "description": "섹션의 실제 텍스트를 먼저 있는 그대로 추출",
            "section_type": "섹션 타입 (text/list/table/box/image 중 하나)",
            "bbox": [x1, y1, x2, y2],
            "title": "description을 바탕으로 생성한 명확한 제목"
        }
    ]
}

섹션 분석 규칙:

1. description 작성 규칙 (가장 먼저 수행)
   - 섹션에 있는 모든 텍스트를 있는 그대로 복사
   - 텍스트의 순서와 구조를 원본과 동일하게 유지
   - 어떠한 요약이나 수정도 하지 않음
   - 모든 특수문자, 공백, 줄바꿈을 그대로 유지

2. section_type 결정 규칙
   - text: 일반 텍스트 형식의 내용
   - list: 번호나 기호로 구분된 목록
   - table: 행과 열로 구성된 표 형식
   - box: 박스나 강조 표시된 내용
   - image: 도표, 그래프, 아이콘이 포함된 시각적 요소

3. title 작성 규칙 (description 추출 후 수행)
   - description의 내용을 정확하게 이해하고 요약
   - 단순히 '보장안내', '주의사항' 같은 일반적인 제목 사용 금지
   - 구체적인 내용을 포함 (예: '임플란트 및 치아보철 보장내용 안내')
   - 섹션의 핵심 내용이나 목적이 명확히 드러나도록 작성

4. 섹션 구분 규칙
   - 의미적으로 연관된 내용은 하나의 섹션으로 처리
   - 표의 제목과 내용은 하나의 섹션으로 통합
   - 이미지와 관련 설명은 하나의 섹션으로 통합
   - 목록의 제목과 항목들은 하나의 섹션으로 통합

예시:
- 잘못된 title: "보장안내", "주의사항", "상품설명"
- 좋은 title: "연령별 가입조건 및 보험기간 안내", "임플란트 치료비 보장금액 및 한도", "보험료 납입면제 조건 설명"

주의사항:
1. 반드시 description을 먼저 추출한 후 title 생성
2. description은 원문 그대로 복사
3. title은 description의 내용을 바탕으로 구체적으로 요약
4. 이미지에 없는 내용 임의 생성 금지
5. 모든 텍스트와 구조는 원본 유지"""

        self.text_extraction_prompt = """
이 이미지에서 보이는 모든 텍스트를 추출해주세요. 다음 사항을 준수해주세요:

1. 텍스트 구조 유지
   - 표의 경우 행과 열 구조 유지
   - 들여쓰기와 단락 구분 유지
   - 번호 매기기와 글머리 기호 유지

2. 특수 기호 처리
   - ※, †, * 등의 특수 기호 보존
   - 괄호, 따옴표 등의 구두점 정확히 표기

3. 순서 준수
   - 왼쪽에서 오른쪽으로
   - 위에서 아래로
   - 표의 경우 행 단위로 처리

4. 수치 정보 정확성
   - 금액, 날짜, 기간 등의 수치 정확히 추출
   - 단위 표기 유지 (원, 세, 년 등)

응답은 다음 JSON 형식으로 제공해주세요:
{
    "text": "추출된 전체 텍스트",
    "structure": "텍스트의 구조적 특징 설명"
}"""

    def analyze_first_page(self, page) -> dict:
        """첫 페이지에서 상품 정보 추출"""
        try:
            # 첫 페이지 이미지 생성
            img = self.get_page_image(page)
            img_base64 = self.encode_image_to_base64(img)

            # 첫 페이지 분석을 위한 프롬프트
            first_page_prompt = """이 보험 상품 설명서의 첫 페이지를 분석하여 다음 정보를 추출해주세요:
1. 상품명: 정확한 상품명 (예: (무)교보치아보장보험(갱신형))
2. 상품특징: 주요 보장 내용과 특징
3. 가입연령: 연령대별 가입 유형

JSON 형식으로 응답해주세요. 실제 텍스트만 포함하고 임의 해석은 하지 마세요."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": first_page_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "이 보험 상품의 정보를 알려주세요.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=1000,
            )

            product_info = json.loads(response.choices[0].message.content)
            self.product_info = product_info
            self.update_system_prompt(product_info)
            return product_info

        except Exception as e:
            self.logger.error(f"첫 페이지 분석 중 오류 발생: {e}")
            return None

    def update_system_prompt(self, product_info: dict):
        """상품 정보를 바탕으로 시스템 프롬프트 업데이트"""
        self.vision_system_prompt = f"""당신은 {product_info['상품명']} 상품 설명서를 분석하는 전문가입니다.
이 상품의 주요 특징은 다음과 같습니다:
{product_info['상품특징']}

가입 가능 연령:
{product_info['가입연령']}

각 페이지의 섹션을 분석하여 JSON 형식으로 응답해주세요:
{{
    "sections": [
        {{
            "description": "섹션의 실제 텍스트를 먼저 있는 그대로 추출",
            "section_type": "섹션 타입 (text/list/table/box/image 중 하나)",
            "bbox": [x1, y1, x2, y2],
            "title": "description을 바탕으로 생성한 명확한 제목"
        }}
    ]
}}

주의사항:
1. description은 반드시 원문 그대로 복사
2. 텍스트를 재해석하거나 다른 말로 바꾸지 마세요
3. 보이는 텍스트를 있는 그대로 복사하여 입력하세요
4. 이미지에 없는 내용을 임의로 생성하지 마세요"""

    def process_pdf(self, pdf_path):
        """PDF 파일을 처리하고 섹션별로 분석"""
        try:
            self.pdf_path = pdf_path
            self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            pdf_document = fitz.open(pdf_path)

            # Process first page differently
            first_page = pdf_document[0]
            self.save_first_page(first_page, 0)

            # Process remaining pages
            num_pages = (
                min(6, len(pdf_document)) if self.test_mode else len(pdf_document)
            )
            for page_num in range(1, num_pages):
                try:
                    page = pdf_document[page_num]
                    self.analyze_page_visually(page, page_num)
                except Exception as e:
                    logging.error(f"페이지 {page_num} 처리 중 오류 발생: {str(e)}")
                    traceback.print_exc()

            pdf_document.close()

            # Save the processed PDF to a different location only if source and destination are different
            dest_path = os.path.join("data", "상품설명", os.path.basename(pdf_path))
            if os.path.abspath(pdf_path) != os.path.abspath(dest_path):
                shutil.copy2(pdf_path, dest_path)

        except Exception as e:
            logging.error(f"PDF 처리 중 오류 발생: {str(e)}")
            traceback.print_exc()

    def analyze_page_visually(self, page, page_num):
        """GPT-4V를 사용하여 페이지의 시각적 구조를 분석"""
        try:
            # 페이지를 이미지로 변환
            img = self.get_page_image(page)
            img_base64 = self.encode_image_to_base64(img)

            # GPT-4V에 이미지 분석 요청
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.vision_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "이 페이지의 구조를 분석해주세요.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high",  # 고해상도 분석 사용
                                },
                            },
                        ],
                    },
                ],
                max_tokens=2000,
            )

            # 응답 로깅 추가
            self.logger.info(f"GPT 응답: {response.choices[0].message.content}")

            # 응답 파싱
            analysis = json.loads(response.choices[0].message.content)
            sections = []

            for section_data in analysis["sections"]:
                section = VisualSection(
                    title=section_data["title"],
                    section_type=section_data["section_type"],
                    bbox=section_data["bbox"],
                    description=section_data.get("description"),
                )
                sections.append(section)

                # 섹션 저장
                self.save_visual_section(page, section, page_num)

            return sections

        except Exception as e:
            self.logger.error(f"비주얼 분석 중 오류 발생: {str(e)}")
            return []

    def save_visual_section(self, page, section: VisualSection, page_num: int):
        """섹션을 이미지와 메타데이터로 저장"""
        try:
            # 좌표 유효성 검사 및 조정
            x1, y1, x2, y2 = section.bbox
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            # 좌표가 페이지 범위를 벗어나지 않도록 조정
            x1 = max(0, min(x1, page_width - 1))
            y1 = max(0, min(y1, page_height - 1))
            x2 = max(1, min(x2, page_width))
            y2 = max(1, min(y2, page_height))

            # 섹션 이미지 추출 및 저장
            clip = fitz.Rect(x1, y1, x2, y2)
            pix = page.get_pixmap(clip=clip)

            # 파일명 생성
            base_name = f"{self.get_base_filename(page_num)}_{section.title}"
            image_path = os.path.join("data", "extracted", "images", f"{base_name}.png")
            json_path = os.path.join("data", "extracted", "text", f"{base_name}.json")

            # 이미지 저장
            pix.save(image_path)

            # 메타데이터 저장
            metadata = {
                "title": section.title,
                "section_type": section.section_type,
                "bbox": [x1, y1, x2, y2],
                "description": section.description,
                "image_path": image_path,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"섹션 저장 중 오류 발생: {str(e)}")

    def save_first_page(self, page, page_num):
        """
        PDF의 첫 페이지를 처리합니다.
        전체 페이지를 하나의 섹션으로 처리하고 PDF 파일명을 제목으로 사용합니다.
        """
        # 페이지의 전체 텍스트 추출
        text = page.get_text()

        # 이미지 저장을 위한 기본 이름 생성
        base_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        image_name = f"{base_name}_p{page_num+1}_full_page.png"
        json_name = f"{base_name}_p{page_num+1}_full_page.json"

        # 이미지 저장 경로
        image_path = os.path.join(self.image_dir, image_name)
        json_path = os.path.join(self.text_dir, json_name)

        # 페이지를 이미지로 저장
        pix = page.get_pixmap()
        pix.save(image_path)

        # 메타데이터 생성
        metadata = {
            "title": base_name,
            "text": text,
            "type": "text",
            "page": page_num + 1,
            "image_path": image_path,
        }

        # JSON 파일로 저장
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def extract_text_from_section(self, img_base64: str) -> dict:
        """GPT-4V를 사용하여 섹션의 텍스트를 추출"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.text_extraction_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "이 이미지에서 모든 텍스트를 추출해주세요.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high",  # 텍스트 추출은 high detail 필요
                                },
                            },
                        ],
                    },
                ],
                max_tokens=2000,
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"텍스트 추출 중 오류 발생: {str(e)}")
            return {"text": "", "structure": ""}

    def get_base_filename(self, page_num: int) -> str:
        """페이지 번호를 기반으로 기본 파일명 생성"""
        return f"{self.pdf_name}_p{page_num + 1}"

    def extract_section_image(self, page, bbox, scale=1):
        """섹션 이미지 추출"""
        try:
            # bbox 좌표 조정 - 여백 추가
            x1, y1, x2, y2 = bbox
            padding = 20

            # 특히 y1 좌표를 위쪽으로 더 확장 (전체 높이의 약 20%부터 시작)
            y1 = min(y1, page.height * 0.2)

            adjusted_bbox = (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(page.width, x2 + padding),
                min(page.height, y2 + padding),
            )

            cropped = page.crop(adjusted_bbox)
            return cropped
        except Exception as e:
            self.logger.error(f"이미지 추출 중 오류 발생: {e}")
            return None

    def get_page_image(self, page) -> Image.Image:
        """페이지를 PIL Image로 변환"""
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))

    def encode_image_to_base64(self, img: Image.Image) -> str:
        """PIL Image를 base64로 인코딩"""
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG", optimize=True)
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    def process_first_page(self, page, page_num):
        """
        PDF의 첫 페이지를 처리합니다.
        전체 페이지를 하나의 섹션으로 처리하고 PDF 파일명을 제목으로 사용합니다.
        """
        # 페이지의 전체 텍스트 추출
        text = page.get_text()

        # 이미지 저장을 위한 기본 이름 생성
        base_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        image_name = f"{base_name}_p{page_num+1}_full_page.png"
        json_name = f"{base_name}_p{page_num+1}_full_page.json"

        # 이미지 저장 경로
        image_path = os.path.join(self.image_dir, image_name)
        json_path = os.path.join(self.text_dir, json_name)

        # 페이지를 이미지로 저장
        pix = page.get_pixmap()
        pix.save(image_path)

        # 메타데이터 생성
        metadata = {
            "title": base_name,
            "text": text,
            "type": "text",
            "page": page_num + 1,
            "image_path": image_path,
        }

        # JSON 파일로 저장
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def process_all_pdfs():
    """모든 PDF 파일 처리"""
    processor = PDFProcessor(
        test_mode=True, test_page_limit=6
    )  # test_mode를 True로 설정
    pdf_dir = "data/상품설명"

    # 테스트 모드에서는 첫 번째 PDF만 처리
    if processor.test_mode:
        test_file = "(무)교보치아보장보험(갱신형).pdf"
        test_file_path = os.path.join(pdf_dir, test_file)
        processor.logger.info(f"테스트 모드: {test_file} 처리 중")
        processor.process_pdf(test_file_path)
        processor.logger.info(f"테스트 모드: {test_file} 처리 완료")
    else:
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                processor.process_pdf(pdf_path)


if __name__ == "__main__":
    process_all_pdfs()
