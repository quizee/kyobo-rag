import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MarkdownStructureEnhancer:
    # x 좌표 차이 임계값 (이 값보다 크면 들여쓰기로 간주)
    X_THRESHOLD = 20

    def __init__(self, json_path: str, md_path: str):
        self.json_path = json_path
        self.md_path = md_path
        self.output_path = md_path.replace(".md", "_enhanced.md")
        self.header_levels = {}  # 텍스트 내용을 키로 하는 헤더 레벨 매핑
        self.page_first_headers = set()  # 각 페이지의 첫 번째 헤더를 저장
        self.page_text_map = {}  # 페이지별 텍스트 매핑
        self.initialize_structure()

    def load_json_data(self) -> Dict:
        """JSON 파일을 로드합니다."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_md_content(self) -> str:
        """Markdown 파일을 로드합니다."""
        with open(self.md_path, "r", encoding="utf-8") as f:
            return f.read()

    def initialize_structure(self):
        """
        Initialize document structure based on JSON data.
        - First header on each page is always level 1
        - Subsequent headers must follow proper level progression:
          - After level 1, next header must be level 2 regardless of indentation
          - After level 2, header can be level 3 if indented more
          - Level 3 can decrease to level 2 if indentation becomes shallower
        - Headers are processed in the order they appear in the JSON
        """
        structure = []
        current_page_number = 1

        for page in self.load_json_data()["pages"]:
            # Reset for new page
            current_page_headers = []
            has_level_1 = False  # 현재 페이지에서 level 1이 나왔는지 추적
            base_x = None

            # Find headers in the order they appear in JSON
            for item in page["items"]:
                if item["type"] == "heading":
                    x = item["bBox"]["x"]
                    if base_x is None:
                        base_x = x
                    current_page_headers.append((x, item["value"]))

            # Process headers with proper leveling
            prev_x = None
            prev_level = None

            for i, (x, text) in enumerate(current_page_headers):
                if not has_level_1:  # 현재 페이지에서 아직 level 1이 없으면
                    level = 1
                    has_level_1 = True
                else:
                    # Calculate x-coordinate difference
                    x_diff = x - prev_x

                    # 이전 레벨에 따라 현재 레벨 결정
                    if prev_level == 1:
                        # level 1 다음에는 무조건 level 2 (들여쓰기와 상관없이)
                        level = 2
                    elif prev_level == 2:
                        # level 2 다음에만 level 3으로 갈 수 있음
                        if x_diff >= self.X_THRESHOLD:
                            level = 3
                        else:
                            level = 2
                    elif prev_level == 3:
                        # level 3은 들여쓰기가 얕아지면 level 2로
                        if x_diff <= -self.X_THRESHOLD:
                            level = 2
                        else:
                            level = 3
                    else:
                        # 기본값은 level 2
                        level = 2

                structure.append(
                    {
                        "type": "header",
                        "level": level,
                        "text": text,
                        "page": current_page_number,
                    }
                )

                # Update previous values for next iteration
                prev_x = x
                prev_level = level

            # Add page text
            current_page_text = page["text"]
            if current_page_text.strip():
                structure.append(
                    {
                        "type": "text",
                        "text": current_page_text,
                        "page": current_page_number,
                    }
                )

            # Add page separator
            structure.append({"type": "separator", "page": current_page_number})
            current_page_number += 1

        self.header_levels = {
            item["text"]: item["level"]
            for item in structure
            if item["type"] == "header"
        }
        self.page_text_map = {
            item["page"]: item["text"] for item in structure if item["type"] == "text"
        }
        self.page_first_headers = {
            item["page"] for item in structure if item["type"] == "header"
        }

    def enhance_line(self, line: str) -> str:
        """
        각 라인을 분석하여 필요한 경우 구조를 개선합니다.
        헤더의 경우 저장된 레벨 정보를 사용하여 적절한 레벨로 변경합니다.
        """
        line = line.strip()
        if not line:
            return line

        # 헤더 패턴 매칭
        header_match = re.match(r"^(#+)\s+(.+)$", line)
        if header_match:
            header_text = header_match.group(2).strip()
            level = self.header_levels.get(header_text, 1)
            return f"{'#' * level} {header_text}"

        return line

    def generate_enhanced_markdown(self) -> str:
        """
        개선된 Markdown을 생성합니다.
        JSON의 페이지 정보를 기반으로 구조화된 문서를 생성합니다.
        """
        content = self.load_md_content()
        lines = content.split("\n")
        enhanced_lines = []

        for line in lines:
            enhanced_line = self.enhance_line(line)
            enhanced_lines.append(enhanced_line)

        return "\n".join(enhanced_lines)

    def save_enhanced_markdown(self) -> None:
        """개선된 Markdown을 파일로 저장합니다."""
        enhanced_content = self.generate_enhanced_markdown()
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(enhanced_content)


def main():
    # 파일 경로 설정
    json_path = "data/상품설명/1.교보마이플랜건강보험 [2409](무배당) (1).pdf (2).json"
    md_path = "data/상품설명/1.교보마이플랜건강보험 [2409](무배당) (1).pdf.md"

    # Markdown 구조 개선
    enhancer = MarkdownStructureEnhancer(json_path, md_path)
    enhancer.save_enhanced_markdown()
    print(f"Enhanced Markdown saved to: {enhancer.output_path}")


if __name__ == "__main__":
    main()
