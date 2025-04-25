import os
import re
import json
import fitz
from PIL import Image
import numpy as np
from typing import List, Dict, Any


class SectionExtractor:
    def __init__(self, md_path: str):
        self.md_path = md_path
        self.sections = []

        # 출력 디렉토리 설정
        self.output_dir = "data/extracted/sections"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 섹션 JSON 저장 디렉토리
        self.sections_json_dir = os.path.join(self.output_dir, "sections_json")
        if not os.path.exists(self.sections_json_dir):
            os.makedirs(self.sections_json_dir)

    def parse_md_sections(self) -> None:
        """마크다운 파일을 파싱하여 섹션 정보를 추출합니다."""
        print("\nParsing MD file...")

        with open(self.md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_section = None
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            header_match = header_pattern.match(line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)

                if current_section is not None:
                    self.sections.append(current_section)

                current_section = {
                    "title": title,
                    "content": [],
                    "level": level,
                    "start_line": i + 1,
                    "section_number": len(self.sections) + 1,
                }
            elif current_section is not None:
                current_section["content"].append(line)

        if current_section is not None:
            self.sections.append(current_section)

    def save_sections_as_json(self) -> None:
        """파싱된 섹션들을 JSON 파일로 저장합니다."""
        # 전체 섹션 정보를 하나의 JSON 파일로 저장
        all_sections_file = os.path.join(self.sections_json_dir, "all_sections.json")
        with open(all_sections_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_sections": len(self.sections),
                    "source_file": self.md_path,
                    "sections": self.sections,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n전체 섹션 정보가 저장되었습니다: {all_sections_file}")

        # 개별 섹션을 각각의 JSON 파일로 저장
        for section in self.sections:
            safe_title = section["title"].replace("/", "_").replace(" ", "_")
            section_file = os.path.join(
                self.sections_json_dir,
                f"section_{section['section_number']:02d}_{safe_title}.json",
            )
            with open(section_file, "w", encoding="utf-8") as f:
                json.dump(section, f, ensure_ascii=False, indent=2)
            print(f"섹션이 저장되었습니다: {section_file}")

    def print_sections(self) -> None:
        """파싱된 섹션을 보기 좋게 출력합니다."""
        print(f"\n총 {len(self.sections)}개의 섹션을 찾았습니다.\n")

        for section in self.sections:
            print(f"섹션 {section['section_number']}:")
            print(f"  제목: {section['title']}")
            print(f"  레벨: {section['level']}")
            print(f"  시작 라인: {section['start_line']}")
            print(f"  내용 ({len(section['content'])}줄):")
            for line in section["content"][:3]:  # 처음 3줄만 출력
                print(f"    {line}")
            if len(section["content"]) > 3:
                print("    ...")
            print()

    def find_section_coordinates(
        self, section_title: str, page_data: Dict
    ) -> List[Dict]:
        """페이지 데이터에서 섹션 제목에 해당하는 좌표와 관련 항목들을 찾습니다."""
        related_items = []

        print(f"\nSearching for section: {section_title}")

        # 먼저 헤더에서 찾기
        header_found = False
        for i, item in enumerate(page_data.get("items", [])):
            if item.get("type") == "heading" and section_title == item.get("value", ""):
                print(f"Found header item: {item.get('value')} at index {i}")
                header_found = True
                related_items.append(item)

                # 헤더 다음에 나오는 항목들도 포함
                for j, next_item in enumerate(
                    page_data.get("items", [])[i + 1 :], start=i + 1
                ):
                    # 다음 헤더를 만나면 중단
                    if next_item.get("type") == "heading" and j != i:
                        print(f"Found next header at index {j}, stopping")
                        break

                    if next_item.get("type") in ["text", "table", "list"]:
                        print(
                            f"Adding related item: {next_item.get('type')} at index {j}"
                        )
                        related_items.append(next_item)

        # 헤더에서 찾지 못한 경우 일반 텍스트에서 찾기
        if not header_found:
            print("Header not found, searching in text items")
            for i, item in enumerate(page_data.get("items", [])):
                if section_title == item.get("value", ""):
                    print(f"Found text item: {item.get('value')} at index {i}")
                    related_items.append(item)

        print(f"Total related items found: {len(related_items)}")
        return related_items

    def extract_section_images(self) -> None:
        """PDF에서 각 섹션의 이미지를 추출하고 관련 JSON 데이터를 저장합니다."""
        if not self.sections:
            print("\nNo sections found to extract")
            return

        print(f"\nFound {len(self.sections)} sections")
        print("\nExtracting section images...")

        pdf_doc = fitz.open(self.pdf_path)
        print(f"\nProcessing PDF with {len(pdf_doc)} pages")

        for page_num in range(len(pdf_doc)):
            print(f"\nProcessing page {page_num + 1}")
            page = pdf_doc[page_num]
            page_data = self.json_data["pages"][page_num]

            # 페이지 데이터 디버깅
            print(f"Page items count: {len(page_data.get('items', []))}")

            for section in self.sections:
                # 섹션 제목에 해당하는 JSON 데이터 찾기
                items = self.find_section_coordinates(section["title"], page_data)
                if items:
                    print(f"Found {len(items)} items for section: {section['title']}")

                    # 섹션의 JSON 데이터 저장
                    section_json = {"page": page_num + 1, "items": items}
                    section["json_data"].append(section_json)

                    # JSON 파일로 저장
                    json_filename = f"{os.path.splitext(os.path.basename(self.pdf_path))[0]}_p{page_num + 1}_{section['title'].replace('/', '_').replace(' ', '_')}.json"
                    json_path = os.path.join(self.sections_json_dir, json_filename)

                    try:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(section_json, f, ensure_ascii=False, indent=2)
                        print(f"Saved JSON: {json_path}")
                    except Exception as e:
                        print(f"Error saving JSON: {str(e)}")

                    # 이미지 추출 및 저장
                    # 모든 관련 항목의 bBox 좌표를 포함하는 영역 계산
                    x0 = min(item["bBox"][0] for item in items)
                    y0 = min(item["bBox"][1] for item in items)
                    x1 = max(item["bBox"][2] for item in items)
                    y1 = max(item["bBox"][3] for item in items)

                    # 여백 추가
                    padding = 20
                    x0 = max(0, x0 - padding)
                    y0 = max(0, y0 - padding)
                    x1 = min(page.rect.width, x1 + padding)
                    y1 = min(page.rect.height, y1 + padding)

                    # 이미지 추출
                    clip_rect = fitz.Rect(x0, y0, x1, y1)
                    pix = page.get_pixmap(clip=clip_rect)

                    # 이미지 저장
                    image_filename = f"{os.path.splitext(os.path.basename(self.pdf_path))[0]}_p{page_num + 1}_{section['title'].replace('/', '_')}.png"
                    image_path = os.path.join(self.output_dir, image_filename)
                    pix.save(image_path)
                    print(f"Saved section: {section['title']} -> {image_path}")

        pdf_doc.close()


def main():
    # 파일 경로 설정
    md_path = "data/상품설명/1.교보마이플랜건강보험 [2409](무배당) (1).pdf.md"

    # 섹션 추출기 생성 및 실행
    extractor = SectionExtractor(md_path)
    extractor.parse_md_sections()
    extractor.print_sections()
    extractor.save_sections_as_json()


if __name__ == "__main__":
    main()
