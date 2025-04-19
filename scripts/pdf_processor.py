import os
import json
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import pdfplumber
import logging
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import io


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(
        self,
        input_dir: str = "data/상품설명",
        output_dir: str = "data/extracted",
        test_mode: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.text_output_dir = self.output_dir / "text"
        self.image_output_dir = self.output_dir / "images"
        self.test_mode = test_mode
        self.test_file = "(무)교보치아보장보험(갱신형).pdf"

        # 출력 디렉토리 생성
        self.text_output_dir.mkdir(parents=True, exist_ok=True)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        # .env 파일에서 환경 변수 로드
        load_dotenv()

        # OpenAI 클라이언트 초기화
        self.client = OpenAI()

        # 로깅 설정 추가
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def get_page_structure(self, page) -> List[Dict]:
        """페이지의 텍스트 구조를 분석합니다."""
        lines = []
        current_y = None
        line_texts = []

        # 페이지의 모든 텍스트 요소를 y좌표 순으로 정렬
        text_elements = sorted(page.extract_words(), key=lambda x: (x["top"], x["x0"]))

        for element in text_elements:
            # 새로운 줄 시작
            if (
                current_y is None or abs(element["top"] - current_y) > 5
            ):  # 5픽셀 이상 차이나면 새로운 줄
                if line_texts:
                    lines.append(
                        {
                            "text": " ".join(line_texts),
                            "top": current_y,
                            "bottom": current_y + element["bottom"] - element["top"],
                            "line_number": len(lines) + 1,
                        }
                    )
                line_texts = [element["text"]]
                current_y = element["top"]
            else:
                line_texts.append(element["text"])

        # 마지막 줄 추가
        if line_texts:
            lines.append(
                {
                    "text": " ".join(line_texts),
                    "top": current_y,
                    "bottom": current_y + element["bottom"] - element["top"],
                    "line_number": len(lines) + 1,
                }
            )

        return lines

    def analyze_page_content(self, page, page_num: int) -> List[Dict]:
        """GPT를 사용하여 페이지의 주제 영역들을 분석합니다."""
        # 페이지 구조 분석
        lines = self.get_page_structure(page)

        # 줄 번호와 함께 텍스트 구성
        text_with_lines = "\n".join(
            [f"[{line['line_number']}] {line['text']}" for line in lines]
        )

        prompt = f"""다음은 PDF의 {page_num}페이지 내용입니다. 각 줄 앞에는 줄 번호가 표시되어 있습니다.
이 페이지에서 각각의 독립적인 주제 영역을 찾아주세요.

가장 중요한 규칙:
1. 제목과 내용 통합: 제목과 그에 해당하는 내용(이미지, 표, 설명 등)은 반드시 하나의 섹션으로 처리해야 합니다.
2. 연속성 우선: 비슷한 내용이나 형식이 이어지는 경우 반드시 하나의 큰 섹션으로 처리해야 합니다.
3. 분할 금지: 하나의 주제나 내용을 여러 섹션으로 나누지 마세요.
4. 시각적 요소 무시: 여백이나 구분선으로 분리되어 보이더라도, 내용이 연속되면 하나로 처리하세요.
5. 번호 항목 통합: "1." 또는 "1)" 등으로 시작하는 번호 항목이 있다면, 반드시 1번부터 마지막 번호까지 모두 하나의 섹션으로 처리해야 합니다.

섹션 유형별 처리 방법:

1. 표(table):
   - 표의 제목이나 설명도 반드시 표의 일부로 포함
   - 표 형식이 유사하거나 같은 주제를 다루는 표는 하나로 처리
   - 중간에 설명이나 주석이 있어도 연속되는 표는 하나의 섹션으로 처리
   - 여러 페이지에 걸쳐있는 표도 각 페이지에서는 하나의 섹션으로 처리
   - 만기지급금과 같은 특별 행도 표의 일부로 처리
   - 시각적으로 구분되어 있더라도 같은 표의 일부인 행은 모두 포함
   - 표 하단의 주석이나 설명도 표의 일부로 포함

2. 다이어그램(diagram):
   - 제목/설명과 관련 이미지는 반드시 하나의 diagram 섹션으로 처리
   - 연속된 이미지나 도표는 무조건 하나의 섹션으로 처리
   - 단계별/순서별로 구성된 이미지들은 하나의 시리즈로 처리
   - 이미지 사이에 설명 텍스트가 있어도 모두 하나의 diagram으로 처리
   - 시각적으로 분리되어 있더라도 같은 주제를 설명하는 이미지들은 하나로 통합

3. 나열 요소(list):
   - 제목과 목록 항목들은 하나의 list 섹션으로 처리
   - 번호나 글머리 기호로 시작하는 모든 연속된 항목을 하나로 처리
   - 하위 항목이 있어도 모두 하나의 list로 처리
   - 설명이나 부연 내용이 중간에 있어도 연속된 목록은 하나로 처리

4. 박스(box):
   - 제목을 포함한 박스 전체 내용을 하나의 섹션으로 처리
   - 테두리나 배경으로 구분된 영역도 주제가 같으면 하나로 처리
   - 연속된 박스들은 하나의 섹션으로 통합
   - 디자인이 다르더라도 같은 주제의 박스들은 하나로 처리

텍스트:
{text_with_lines}

예시 응답:
{{
    "sections": [
        {{
            "title": "보험금 지급 관련 안내사항",
            "start_line": 1,
            "end_line": 50
        }}
    ]
}}

주의사항:
1. 제목 포함: 섹션의 제목이나 설명은 반드시 해당 섹션에 포함시켜야 함
2. 최대한 큰 단위로 묶기: 비슷한 내용은 가능한 한 큰 섹션으로 통합
3. 문맥 우선: 시각적 구분보다 내용의 연관성을 우선적으로 고려
4. 번호 목록 처리 시 주의사항:
   - 1번(또는 1.)으로 시작하는 목록은 반드시 마지막 번호까지 하나의 섹션
   - 번호 사이의 모든 설명과 부연 내용도 포함
   - 번호 목록 중간의 들여쓰기나 하위 항목도 모두 포함
   - 번호가 끊기더라도 내용이 연속되면 하나의 섹션으로 처리
5. 표 처리 시 주의사항:
   - 만기지급금과 같은 특별 행도 반드시 표의 일부로 포함
   - 표 하단의 주석이나 설명도 표의 일부로 포함
   - 시각적 구분선이나 여백이 있어도 같은 표의 내용이면 분리하지 않음

위 텍스트를 분석하여 각각의 독립적인 주제 영역을 찾아 예시와 같은 형식으로 응답해주세요."""

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            sections = json.loads(completion.choices[0].message.content)["sections"]
            logger.info(
                f"GPT가 찾은 섹션들: {json.dumps(sections, ensure_ascii=False, indent=2)}"
            )
            return sections
        except Exception as e:
            logger.error(f"GPT 분석 중 오류 발생: {e}")
            return []

    def normalize_text(self, text: str) -> str:
        """텍스트를 정규화합니다."""
        return " ".join(text.split())

    def find_text_in_words(self, words: List[Dict], target_text: str) -> Optional[Dict]:
        """단어 목록에서 특정 텍스트가 포함된 단어들을 찾습니다."""
        normalized_target = self.normalize_text(target_text)

        # 단일 단어 검색
        for word in words:
            if normalized_target in self.normalize_text(word["text"]):
                return word

        # 여러 단어 검색
        text_buffer = ""
        start_word = None

        for word in words:
            text_buffer += word["text"] + " "
            if not start_word:
                start_word = word

            if normalized_target in self.normalize_text(text_buffer):
                # 여러 단어에 걸친 경우, 첫 단어의 x0, top과 마지막 단어의 x1, bottom 사용
                return {
                    "x0": start_word["x0"],
                    "x1": word["x1"],
                    "top": start_word["top"],
                    "bottom": word["bottom"],
                    "text": text_buffer.strip(),
                }

            # 버퍼가 너무 길어지면 초기화
            if len(text_buffer) > len(target_text) * 2:
                text_buffer = ""
                start_word = None

        return None

    def find_region_bbox(
        self, page, start_line: int, end_line: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """페이지에서 지정된 줄 범위의 영역을 찾습니다."""
        lines = self.get_page_structure(page)

        if start_line < 1 or end_line > len(lines):
            logger.warning(
                f"유효하지 않은 줄 범위입니다: {start_line} -> {end_line} (전체 줄 수: {len(lines)})"
            )
            return None

        start_line_data = lines[start_line - 1]
        end_line_data = lines[end_line - 1]

        # bbox 계산 (여백 20픽셀 추가)
        padding = 20
        x0 = 0  # 페이지 전체 너비 사용
        y0 = max(0, float(start_line_data["top"]) - padding)
        x1 = float(page.width)  # 페이지 전체 너비 사용
        y1 = min(float(page.height), float(end_line_data["bottom"]) + padding)

        return (x0, y0, x1, y1)

    def capture_region(
        self, page, bbox: Tuple[float, float, float, float]
    ) -> Optional[Image.Image]:
        """지정된 영역을 이미지로 캡처합니다."""
        try:
            zoom = 2.0  # 고해상도를 위한 2배 확대
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=bbox)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as e:
            logger.error(f"이미지 캡처 중 오류 발생: {e}")
            return None

    def process_pdf(self, pdf_path: str) -> None:
        """PDF 파일을 처리합니다."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"총 {total_pages}페이지를 처리합니다.")

                # 테스트 모드일 경우 처음 6장만 처리
                pages_to_process = 6 if self.test_mode else total_pages

                for page_num in range(pages_to_process):
                    logger.info(f"페이지 {page_num + 1} / {pages_to_process} 처리 중")
                    page = pdf.pages[page_num]

                    # 첫 번째 페이지는 전체를 하나의 섹션으로 처리
                    if page_num == 0:
                        lines = self.get_page_structure(page)
                        section = {
                            "title": "표지",
                            "start_line": 1,
                            "end_line": len(lines),
                        }
                        self.save_section(page, section, page_num + 1, 1, pdf_path)
                        logger.info(f"Saved first page as a single section")
                    else:
                        # 나머지 페이지는 GPT로 분석
                        sections = self.analyze_page_content(page, page_num + 1)
                        for i, section in enumerate(sections, 1):
                            self.save_section(page, section, page_num + 1, i, pdf_path)
                            logger.info(f"Saved section {i} from page {page_num + 1}")

                    logger.info(f"{page_num + 1}")

        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {e}")
            raise

    def process_all_pdfs(self):
        """모든 PDF 파일을 처리합니다."""
        if self.test_mode:
            test_file_path = self.input_dir / self.test_file
            if test_file_path.exists():
                logger.info(f"테스트 모드: {self.test_file} 처리 중")
                self.process_pdf(str(test_file_path))
                logger.info(f"테스트 모드: {self.test_file} 처리 완료")
            else:
                logger.error(f"테스트 모드: {self.test_file} 파일을 찾을 수 없습니다")
            return

        pdf_files = list(self.input_dir.glob("**/*.pdf"))
        total_files = len(pdf_files)

        for idx, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"파일 처리 중 {idx}/{total_files}: {pdf_path}")
            self.process_pdf(str(pdf_path))
            logger.info(f"파일 처리 완료: {pdf_path}")

    def save_section(
        self, page, section: Dict, page_num: int, section_num: int, pdf_path: str
    ) -> None:
        """섹션을 저장합니다."""
        try:
            # PyMuPDF 페이지 객체로 변환
            doc = fitz.open()
            doc.insert_pdf(
                fitz.open(pdf_path), from_page=page_num - 1, to_page=page_num - 1
            )
            mupdf_page = doc[0]

            # 섹션의 bbox 찾기
            bbox = self.find_region_bbox(
                page, section["start_line"], section["end_line"]
            )
            if bbox:
                # 이미지 캡처 및 저장
                img = self.capture_region(mupdf_page, bbox)
                if img:
                    # 파일명 생성
                    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    section_name = f"{base_name}_p{page_num}_section_{section_num}"

                    # 이미지 저장
                    img_path = self.image_output_dir / f"{section_name}.png"
                    img.save(str(img_path))

                    # 메타데이터 저장
                    section["image_path"] = str(img_path)
                    section["bbox"] = bbox
                    metadata_path = self.text_output_dir / f"{section_name}.json"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(section, f, ensure_ascii=False, indent=2)

            doc.close()
        except Exception as e:
            logger.error(f"섹션 저장 중 오류 발생: {e}")
            raise


if __name__ == "__main__":
    processor = PDFProcessor(test_mode=True)
    processor.process_all_pdfs()
