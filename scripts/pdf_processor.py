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
    description: Optional[str] = None
    continues_to_next_page: bool = False
    continues_from_previous_page: bool = False


class PDFProcessor:
    def __init__(self, test_mode=False, test_page_limit=3):
        self.client = OpenAI()
        self.test_mode = test_mode
        self.test_page_limit = test_page_limit
        self.model = "gpt-4.1-mini"
        self.product_info = None
        self.logger = logging.getLogger(__name__)
        self.pdf_path = None
        self.processed_regions = {}  # 페이지별 처리된 영역 추적
        self.pending_sections = {}  # 페이지 간 연속되는 섹션을 추적하기 위한 딕셔너리

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
페이지를 위에서 아래로 분석하여 각 섹션을 식별하고 다음 형식으로 응답하세요.
코드 블록 표시(```json, ```)는 사용하지 말고 JSON 객체만 직접 응답하세요:

{
    "sections": [
        {
            "description": "섹션의 실제 텍스트를 먼저 있는 그대로 추출",
            "bbox": [x1, y1, x2, y2],
            "title": "description을 바탕으로 생성한 명확한 제목",
            "continues_to_next_page": false,
            "continues_from_previous_page": false
        }
    ]
}

핵심 규칙:

1. 논리적 정보 단위 (최우선)
   - 동일한 주제나 목적을 가진 연속된 내용은 하나의 섹션으로 처리
   - 제목과 그에 대한 상세 설명이 함께 있는 경우 반드시 하나의 섹션으로 처리
   - 강조된 텍스트(굵은 글씨, 색상, 큰 폰트 등)와 그 뒤의 일반 텍스트가 하나의 내용을 설명할 때는 분리하지 않음

2. 중복 방지 (필수)
   - 모든 내용은 정확히 한 번만 처리
   - 이미 포함된 내용을 다른 섹션으로 만들지 않음
   - 하나의 섹션에 포함된 텍스트나 이미지를 다른 섹션에서 중복해서 사용하지 않음

3. 시각적 그룹화
   - 동일한 배경색이나 디자인을 공유하는 요소들은 하나의 섹션으로 처리
   - 연속된 박스나 아이콘이 하나의 주제를 설명하는 경우 분리하지 않음
   - 같은 디자인의 연속된 요소(아이콘 등)는 하나의 섹션으로 처리

4. 페이지 연속성 처리
   - 섹션이 다음 페이지로 이어지는 경우 continues_to_next_page를 true로 설정
   - 섹션이 이전 페이지에서 이어지는 경우 continues_from_previous_page를 true로 설정
   - 표나 설명이 페이지 경계에서 잘리는 경우 연속성 표시

5. bbox 설정 규칙
   - x1은 항상 0으로 설정 (페이지 왼쪽 끝)
   - x2는 항상 페이지 너비로 설정 (페이지 오른쪽 끝)
   - y1과 y2는 섹션의 모든 내용을 완전히 포함하도록 설정

6. description과 title
   - description: 원문 텍스트를 그대로 복사
   - title: 내용을 명확하게 요약

올바른 섹션 구분의 예:

1. 제목과 설명이 함께 있는 경우
   [올바름] "비용이 부담스러운 보철치료는 든든하게!"와 그 아래의 "비용이 많이 들어가는 임플란트, 브릿지, 틀니 보철치료시 부담이 덜하도록 보장" 설명을 하나의 섹션으로 처리
   [잘못됨] 제목과 설명을 별도의 섹션으로 분리

2. 아이콘과 설명이 함께 있는 경우
   [올바름] 치아 아이콘과 "자주 발생하는 보존치료는 꼼꼼하게!"와 그 아래의 "온라게 발생하는 복합레진, 크라운, 아말감, 글래스아이노머 보존치료도 부담없고 치료" 설명을 하나의 섹션으로 처리
   [잘못됨] 아이콘, 제목, 설명을 각각 다른 섹션으로 분리

3. 동일한 배경색/디자인의 연속된 박스들
   [올바름] 같은 파란색 배경의 연속된 3개의 특징 설명 박스를 하나의 섹션으로 처리
   [잘못됨] 각각의 파란색 박스를 개별 섹션으로 분리

4. 표나 도표가 있는 경우
   [올바름] "치아보장 보험료 예시"라는 제목과 그 아래의 연령별/성별 보험료 표를 하나의 섹션으로 처리
   [잘못됨] 표의 제목과 내용을 별도의 섹션으로 분리

5. 강조된 텍스트와 일반 텍스트
   [올바름] "보험료 갱신에 관한 사항"이라는 강조 텍스트와 그 아래의 상세 설명을 하나의 섹션으로 처리
   [잘못됨] 강조된 제목과 일반 텍스트 설명을 별도의 섹션으로 분리

중요: 응답은 JSON 객체만 반환하고, 코드 블록 표시(```json, ```)나 다른 마크다운 형식을 사용하지 마세요."""

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

응답은 다음 형식으로 제공해주세요 (코드 블록 표시 없이 JSON 객체만 직접 응답):
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

            # 페이지의 전체 텍스트 추출하여 프롬프트에 포함
            page_text = page.get_text()

            analysis_prompt = f"""이 페이지의 구조를 분석해주세요.
페이지의 전체 텍스트는 다음과 같습니다:

{page_text}

위 텍스트를 참고하여 페이지의 섹션을 정확하게 구분하고, 각 섹션의 전체 텍스트가 누락되지 않도록 해주세요.
반드시 올바른 JSON 형식으로 응답해주세요."""

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
                                "text": analysis_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high",
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
            analysis = self.parse_gpt_response(response.choices[0].message.content)
            sections = []

            # 페이지의 구조 분석
            structure = self.analyze_page_structure(page)

            for section_data in analysis.get("sections", []):
                # bbox 좌표가 전체 텍스트를 포함하는지 확인
                bbox = section_data.get("bbox", [0, 0, 0, 0])
                section_rect = fitz.Rect(bbox)

                # 섹션 영역 내의 모든 텍스트 블록 확인
                text_in_section = page.get_text("text", clip=section_rect)
                if not text_in_section and section_data.get("description"):
                    # 텍스트가 누락된 경우 bbox 확장
                    expanded_rect = self.expand_bbox_to_include_text(
                        page, section_rect, section_data["description"]
                    )
                    if expanded_rect:
                        section_data["bbox"] = [
                            expanded_rect.x0,
                            expanded_rect.y0,
                            expanded_rect.x1,
                            expanded_rect.y1,
                        ]

                section = VisualSection(
                    title=section_data.get("title", ""),
                    bbox=section_data.get("bbox", [0, 0, 0, 0]),
                    description=section_data.get("description", ""),
                    continues_to_next_page=section_data.get(
                        "continues_to_next_page", False
                    ),
                    continues_from_previous_page=section_data.get(
                        "continues_from_previous_page", False
                    ),
                )
                sections.append(section)

                # 섹션 저장
                self.save_visual_section(page, section, page_num)

            return sections

        except Exception as e:
            self.logger.error(f"비주얼 분석 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return []

    def expand_bbox_to_include_text(self, page, initial_rect, target_text):
        """주어진 텍스트를 포함하도록 bbox를 확장"""
        try:
            # 페이지의 모든 텍스트 블록 가져오기
            page_dict = page.get_text("dict")
            # Rect 복사본 생성
            expanded_rect = fitz.Rect(
                initial_rect.x0, initial_rect.y0, initial_rect.x1, initial_rect.y1
            )

            for block in page_dict["blocks"]:
                if block.get("type") == 0:  # 텍스트 블록
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]

                    # 이 블록이 target_text의 일부를 포함하는지 확인
                    if any(part in block_text for part in target_text.split()):
                        block_rect = fitz.Rect(block["bbox"])
                        expanded_rect.include_rect(block_rect)

            return expanded_rect
        except Exception as e:
            self.logger.error(f"bbox 확장 중 오류 발생: {str(e)}")
            return None

    def analyze_page_structure(self, page):
        """페이지의 구조를 분석하여 텍스트 스타일과 레이아웃 정보를 반환"""
        structure = {
            "title_spans": [],  # 제목으로 보이는 텍스트
            "body_spans": [],  # 본문 텍스트
            "image_blocks": [],  # 이미지 블록
            "avg_font_size": 0,  # 평균 폰트 크기
        }

        # 페이지의 상세 정보 가져오기
        page_dict = page.get_text("dict")
        total_font_size = 0
        span_count = 0

        # 모든 블록 분석
        for block in page_dict["blocks"]:
            # 텍스트 블록 분석
            if block.get("type") == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        total_font_size += span["size"]
                        span_count += 1

                        # 폰트 크기가 크거나 볼드체인 경우 제목으로 간주
                        if span["size"] > 10 or "Bold" in span["font"]:
                            structure["title_spans"].append(span)
                        else:
                            structure["body_spans"].append(span)

            # 이미지 블록 저장
            elif block.get("type") == 1:
                structure["image_blocks"].append(block)

        # 평균 폰트 크기 계산
        if span_count > 0:
            structure["avg_font_size"] = total_font_size / span_count

        return structure

    def is_overlapping(self, rect1, rect2, threshold=0.5):
        """두 영역이 지정된 임계값 이상 겹치는지 확인"""
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])

        if x1 >= x2 or y1 >= y2:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        smaller_area = min(area1, area2)

        return intersection / smaller_area > threshold

    def check_region_overlap(self, page_num, bbox):
        """이미 처리된 영역과의 중복 여부 확인"""
        if page_num not in self.processed_regions:
            self.processed_regions[page_num] = []
            return False, None

        for idx, (region, section_info) in enumerate(self.processed_regions[page_num]):
            if self.is_overlapping(bbox, region):
                return True, section_info
        return False, None

    def save_visual_section(self, page, section: VisualSection, page_num: int):
        """섹션을 이미지와 메타데이터로 저장"""
        try:
            # 페이지 크기 가져오기
            page_rect = page.rect

            # 섹션의 텍스트 내용을 기반으로 실제 영역 찾기
            text_instances = []
            page_dict = page.get_text("dict")
            target_text = section.description.split("\n")  # 설명을 줄바꿈으로 분리

            # 페이지의 모든 텍스트 블록을 검사하여 섹션 내용이 포함된 영역 찾기
            for block in page_dict["blocks"]:
                if block.get("type") == 0:  # 텍스트 블록
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]

                    # 섹션 내용이 이 블록에 포함되어 있는지 확인
                    if any(part in block_text for part in target_text):
                        text_instances.append(fitz.Rect(block["bbox"]))

            if text_instances:
                # 모든 텍스트 인스턴스를 포함하는 최소 영역 계산
                combined_rect = text_instances[0]
                for rect in text_instances[1:]:
                    combined_rect.include_rect(rect)

                # 여백 추가
                padding = 10
                combined_rect.x0 = max(0, combined_rect.x0 - padding)
                combined_rect.y0 = max(0, combined_rect.y0 - padding)
                combined_rect.x1 = min(page_rect.width, combined_rect.x1 + padding)
                combined_rect.y1 = min(page_rect.height, combined_rect.y1 + padding)

                # 업데이트된 bbox 저장
                section.bbox = [
                    combined_rect.x0,
                    combined_rect.y0,
                    combined_rect.x1,
                    combined_rect.y1,
                ]
            else:
                # 텍스트를 찾지 못한 경우 원본 bbox 사용
                self.logger.warning(
                    f"섹션 '{section.title}'의 텍스트를 페이지에서 찾을 수 없습니다."
                )

            # 이미지 저장
            base_name = f"{self.get_base_filename(page_num)}_{section.title}"
            image_path = os.path.join(self.image_output_dir, f"{base_name}.png")
            json_path = os.path.join(self.text_output_dir, f"{base_name}.json")

            # 섹션을 이미지로 저장
            clip_rect = fitz.Rect(section.bbox)
            pix = page.get_pixmap(clip=clip_rect)
            pix.save(image_path)

            # 메타데이터 저장
            metadata = {
                "title": section.title,
                "bbox": list(section.bbox),  # bbox를 리스트로 변환
                "description": section.description,
                "page_number": page_num,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"섹션 저장 중 오류 발생: {str(e)}")
            traceback.print_exc()

    def remove_existing_section(self, page_num, section):
        """기존 섹션 제거"""
        try:
            # 이미지 파일 삭제
            base_name = f"{self.get_base_filename(page_num)}_{section.title}"
            image_path = os.path.join("data", "extracted", "images", f"{base_name}.png")
            json_path = os.path.join("data", "extracted", "text", f"{base_name}.json")

            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(json_path):
                os.remove(json_path)

            # processed_regions에서 제거
            for idx, (region, sec) in enumerate(self.processed_regions[page_num]):
                if sec == section:
                    self.processed_regions[page_num].pop(idx)
                    break

        except Exception as e:
            self.logger.error(f"섹션 제거 중 오류 발생: {str(e)}")

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

    def clean_json_string(self, json_str: str) -> str:
        """JSON 문자열을 정리하여 파싱 가능한 형태로 변환"""
        try:
            # ```json 형식의 코드 블록 제거
            json_str = json_str.replace("```json", "").replace("```", "").strip()

            # 제어 문자 제거
            json_str = "".join(
                char for char in json_str if ord(char) >= 32 or char in "\n\r\t"
            )

            # 줄바꿈과 탭을 적절한 이스케이프로 변환
            json_str = (
                json_str.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
            )

            # 따옴표 처리
            json_str = json_str.replace('\\"', '"')  # 이미 이스케이프된 따옴표 복원
            json_str = json_str.replace('"', '\\"')  # 모든 따옴표 이스케이프

            # 전체 문자열을 JSON 형식으로 감싸기
            json_str = f'{{"content": "{json_str}"}}'

            # 파싱 테스트
            parsed = json.loads(json_str)
            return parsed["content"]
        except Exception as e:
            self.logger.error(f"JSON 문자열 정리 중 오류 발생: {str(e)}")
            return json_str

    def parse_gpt_response(self, response_content: str) -> dict:
        """GPT 응답을 파싱하여 딕셔너리로 변환"""
        try:
            # 먼저 직접 파싱 시도
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                self.logger.warning("직접 JSON 파싱 실패, 정리 후 재시도")

            # JSON 부분만 추출 (코드 블록 처리 포함)
            if "```json" in response_content:
                start = response_content.find("```json") + 7
                end = response_content.find("```", start)
                if end == -1:  # 닫는 블록이 없는 경우
                    json_str = response_content[start:].strip()
                else:
                    json_str = response_content[start:end].strip()
            else:
                # 일반적인 JSON 찾기
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                else:
                    raise ValueError("JSON 형식을 찾을 수 없습니다")

            # 직접 파싱 시도
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 실패하면 문자열 정리 후 다시 시도
                cleaned_json = self.clean_json_string(json_str)
                return json.loads(cleaned_json)

        except Exception as e:
            self.logger.error(f"GPT 응답 파싱 중 오류 발생: {str(e)}")
            self.logger.error(f"원본 응답: {response_content}")
            # 기본 구조 반환
            return {"sections": []}


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
