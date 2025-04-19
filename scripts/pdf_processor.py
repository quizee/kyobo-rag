import os
import json
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import pdfplumber
import logging
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

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

텍스트:
{text_with_lines}

예시 응답:
{{
    "sections": [
        {{
            "title": "치아의 기본 구조와 기능",
            "start_line": 1,
            "end_line": 3
        }},
        {{
            "title": "임플란트 시술의 단계별 과정",
            "start_line": 4,
            "end_line": 7
        }}
    ]
}}

주의사항:
1. 각 섹션은 논리적으로 완결된 하나의 주제를 다뤄야 합니다
2. start_line과 end_line은 해당 섹션의 시작과 끝 줄 번호입니다
3. title은 해당 섹션의 내용을 잘 설명하는 구체적인 제목을 지정해주세요

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

    def process_pdf(self, pdf_path: str, test_mode: bool = False):
        """PDF 파일을 처리하여 텍스트와 이미지를 추출합니다."""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return

        logger.info(f"Processing {os.path.basename(pdf_path)}")
        total_pages = 0

        # PyMuPDF로 PDF 열기 (이미지 추출용)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"총 페이지 수: {total_pages}")

        # pdfplumber로 PDF 열기 (텍스트 추출용)
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(len(pdf.pages)):
                logger.info(f"Processing page {page_num + 1} of {total_pages}")

                try:
                    # pdfplumber 페이지 객체
                    plumber_page = pdf.pages[page_num]

                    # 페이지 구조 출력
                    lines = self.get_page_structure(plumber_page)
                    logger.info(f"페이지 {page_num + 1}의 구조:")
                    for line in lines:
                        logger.info(f"[{line['line_number']}] {line['text']}")

                    # PyMuPDF 페이지 객체
                    mupdf_page = doc[page_num]

                    # GPT로 페이지 내용 분석
                    sections = self.analyze_page_content(plumber_page, page_num + 1)

                    # 각 섹션 처리
                    for i, section in enumerate(sections):
                        try:
                            # 섹션의 bbox 찾기
                            bbox = self.find_region_bbox(
                                plumber_page, section["start_line"], section["end_line"]
                            )
                            if bbox:
                                # 이미지 캡처 및 저장
                                img = self.capture_region(mupdf_page, bbox)
                                if img:
                                    # 파일명 생성
                                    base_name = os.path.splitext(
                                        os.path.basename(pdf_path)
                                    )[0]
                                    section_name = (
                                        f"{base_name}_p{page_num+1}_section_{i+1}"
                                    )

                                    # 이미지 저장
                                    img_path = (
                                        self.image_output_dir / f"{section_name}.png"
                                    )
                                    img.save(str(img_path))

                                    # 메타데이터 저장
                                    section["image_path"] = str(img_path)
                                    section["bbox"] = bbox
                                    metadata_path = (
                                        self.text_output_dir / f"{section_name}.json"
                                    )
                                    with open(
                                        metadata_path, "w", encoding="utf-8"
                                    ) as f:
                                        json.dump(
                                            section, f, ensure_ascii=False, indent=2
                                        )

                                    logger.info(
                                        f"Saved section {i+1} from page {page_num+1}"
                                    )

                        except Exception as e:
                            logger.error(f"섹션 처리 중 오류 발생: {e}")
                            continue

                except Exception as e:
                    logger.error(f"페이지 처리 중 오류 발생: {e}")
                    continue

        doc.close()
        logger.info(f"PDF 처리 완료: {os.path.basename(pdf_path)}")

    def process_all_pdfs(self):
        """모든 PDF 파일을 처리합니다."""
        if self.test_mode:
            test_file_path = self.input_dir / self.test_file
            if test_file_path.exists():
                logger.info(f"테스트 모드: {self.test_file} 처리 중")
                self.process_pdf(str(test_file_path), test_mode=True)
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


if __name__ == "__main__":
    processor = PDFProcessor(test_mode=True)
    processor.process_all_pdfs()
