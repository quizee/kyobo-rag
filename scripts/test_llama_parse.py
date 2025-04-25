from pathlib import Path
from llama_cloud_services import LlamaParse
import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
import re
from pdf2image import convert_from_path
from PIL import Image
import tempfile

# .env 파일에서 API 키 로드
load_dotenv()


def serialize_layout(layout):
    """레이아웃 객체를 JSON 직렬화 가능한 형태로 변환"""
    if layout is None:
        return None

    if isinstance(layout, (int, float, str, bool)):
        return layout

    if isinstance(layout, (list, tuple)):
        return [serialize_layout(item) for item in layout]

    if isinstance(layout, dict):
        return {k: serialize_layout(v) for k, v in layout.items()}

    # 객체를 딕셔너리로 변환
    try:
        return {
            k: serialize_layout(v)
            for k, v in layout.__dict__.items()
            if not k.startswith("_")
        }
    except:
        return str(layout)


def extract_headers_from_markdown(markdown_text: str) -> List[Dict[str, str]]:
    """Markdown 텍스트에서 헤더를 추출"""
    # Markdown 헤더 패턴 (ATX 스타일: #, ##, ### 등)
    header_pattern = r"^(#{1,6})\s+(.+?)(?:\s*#*\s*)?$"

    headers = []
    for line in markdown_text.split("\n"):
        match = re.match(header_pattern, line.strip())
        if match:
            level = len(match.group(1))  # '#' 개수로 헤더 레벨 판단
            text = match.group(2).strip()
            headers.append({"level": level, "text": text})

    return headers


def normalize_text(text: str) -> str:
    """텍스트 정규화: 공백 제거 및 소문자 변환"""
    return re.sub(r"\s+", "", text.lower())


def texts_are_similar(text1: str, text2: str) -> bool:
    """두 텍스트가 유사한지 확인"""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    # 정확히 일치
    if norm1 == norm2:
        return True

    # 한쪽이 다른쪽을 포함
    if norm1 in norm2 or norm2 in norm1:
        return True

    return False


def find_matching_layout_element(page: Any, header_text: str) -> Dict[str, Any]:
    """페이지 내에서 헤더 텍스트와 일치하는 레이아웃 요소 찾기"""
    if not hasattr(page, "images"):
        return None

    # 레이아웃 요소들을 y 좌표 순으로 정렬
    layout_elements = sorted(
        [img for img in page.images if hasattr(img, "text") and img.text.strip()],
        key=lambda x: x.y,
    )

    # 각 요소와 비교
    for elem in layout_elements:
        if texts_are_similar(elem.text, header_text):
            return {
                "y": elem.y,
                "x": elem.x,
                "width": elem.width,
                "height": elem.height,
                "type": getattr(elem, "type", "unknown"),
                "text": elem.text,
            }

    return None


def extract_section_coordinates(pages: List[Any]) -> List[Dict[str, Any]]:
    """각 페이지의 섹션 정보를 추출"""
    sections = []

    for page_num, page in enumerate(pages, 1):
        if not hasattr(page, "items"):
            continue

        # 현재 페이지의 모든 항목을 y좌표 순으로 정렬
        items = sorted(page.items, key=lambda x: x.bBox.y)

        # 헤더 항목들만 추출
        headers = [item for item in items if item.type == "heading"]

        # 각 헤더에 대해 섹션 정보 추출
        for header in headers:
            # 섹션의 끝은 헤더의 y좌표 + 높이
            section_end = header.bBox.y + header.bBox.h

            # 현재 헤더와 끝점 사이의 모든 항목 찾기
            section_items = [
                item for item in items if header.bBox.y <= item.bBox.y < section_end
            ]

            sections.append(
                {
                    "page": page_num,
                    "header_level": header.lvl,
                    "header_text": header.value,
                    "y": header.bBox.y,
                    "x": header.bBox.x,
                    "width": header.bBox.w,
                    "height": header.bBox.h,
                    "next_y": section_end,  # 헤더의 y + height
                    "items": [
                        {
                            "type": item.type,
                            "text": item.value,
                            "y": item.bBox.y,
                            "height": item.bBox.h,
                        }
                        for item in section_items
                    ],
                }
            )

    return sections


def crop_pdf_sections(result, sections, output_dir):
    """PDF 페이지 스크린샷을 저장하고 섹션별로 자릅니다."""
    print("\n페이지 스크린샷 저장 중...")

    # PIL의 이미지 크기 제한 늘리기
    Image.MAX_IMAGE_PIXELS = None

    # 페이지별 스크린샷 이미지와 높이 정보 저장
    page_images = {}
    page_heights = {}
    for page_num, page in enumerate(result.pages, 1):
        # 전체 페이지 스크린샷 찾기
        full_page_screenshots = [
            img
            for img in page.images
            if getattr(img, "type", None) == "full_page_screenshot"
        ]

        if not full_page_screenshots:
            print(f"페이지 {page_num}의 스크린샷을 찾을 수 없습니다.")
            continue

        # 스크린샷 저장
        screenshot = full_page_screenshots[0]
        try:
            image_path = result.save_image(screenshot.name, str(output_dir))
            page_images[page_num] = Image.open(image_path)
            page_heights[page_num] = screenshot.height  # PDF 좌표계의 높이 저장
            print(f"페이지 {page_num} 스크린샷 저장됨: {image_path}")
        except Exception as e:
            print(f"페이지 {page_num} 스크린샷 저장 중 오류 발생: {str(e)}")
            continue

    print("\n섹션별 이미지 추출 중...")
    # 페이지별로 섹션 그룹화
    page_sections = {}
    for section in sections:
        page_num = section["page"]
        if page_num not in page_sections:
            page_sections[page_num] = []
        page_sections[page_num].append(section)

    # 각 페이지의 섹션을 y 좌표 순으로 정렬하고 처리
    for page_num, page_sections_list in page_sections.items():
        if page_num not in page_images:
            continue

        # y 좌표 순으로 정렬
        page_sections_list.sort(key=lambda x: x["y"])
        page_image = page_images[page_num]
        img_width, img_height = page_image.size
        pdf_height = page_heights[page_num]  # PDF 좌표계의 높이

        # 원본 이미지의 높이와 PDF 좌표 사이의 비율 계산
        scale_factor = img_height / pdf_height

        # 각 섹션 처리
        for section in page_sections_list:
            # PDF 좌표를 이미지 픽셀 좌표로 변환
            y_start = int(section["y"] * scale_factor)
            section_height = int(section["height"] * scale_factor)
            y_end = y_start + section_height

            # 좌표 유효성 검사
            if y_start < 0:
                y_start = 0
            if y_end > img_height:
                y_end = img_height
            if y_end <= y_start:
                print(
                    f"섹션 좌표 오류 - 시작: {y_start}, 끝: {y_end}, 높이: {img_height}"
                )
                continue

            try:
                # 섹션 이미지 추출
                section_image = page_image.crop((0, y_start, img_width, y_end))

                # 파일명 생성
                title = (
                    section["header_text"]
                    .replace(" ", "-")
                    .replace("?", "")
                    .replace("/", "-")
                )
                image_path = os.path.join(output_dir, f"page{page_num}_{title}.png")

                # 이미지 저장
                section_image.save(image_path, "PNG")
                print(f"섹션 이미지 저장됨: {image_path} (y: {y_start}-{y_end})")
            except Exception as e:
                print(f"섹션 이미지 저장 중 오류 발생: {str(e)}")
                continue


def test_parse():
    """PDF 파싱 테스트"""
    pdf_path = Path(
        "/Users/jeeyoonlee/Desktop/kyobo-project/data/상품설명/1.교보마이플랜건강보험 [2409](무배당).pdf"
    )
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    print(f"파일: {pdf_path}")

    try:
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            num_workers=1,
            verbose=True,
            language="ko",
            extract_layout=True,
            take_screenshot=True,
            premium_mode=True,
            continuous_mode=False,
            extract_charts=True,
            save_images=True,
            output_tables_as_HTML=False,
            max_pages=6,  # 테스트를 위해 6페이지로 제한
        )

        print("PDF 파싱 시작...")
        result = parser.parse(str(pdf_path))
        print("파싱 완료!")

        # 3페이지 분석
        print("\n=== 3페이지 분석 ===")
        page3 = result.pages[2]

        if hasattr(page3, "items"):
            items = page3.items
            print(f"\n총 {len(items)}개의 항목 발견")

            # 타입별 항목 수 집계
            type_counts = {}
            for item in items:
                if item.type not in type_counts:
                    type_counts[item.type] = 0
                type_counts[item.type] += 1

            print("\n타입별 항목 수:")
            for type_name, count in type_counts.items():
                print(f"- {type_name}: {count}개")

            # 헤더 분석
            headers = [item for item in items if item.type == "heading"]
            print(f"\n헤더 수: {len(headers)}")
            for header in headers:
                print(f"\n레벨 {header.lvl} 헤더: {header.value}")
                print(f"  위치: y={header.bBox.y:.2f}, h={header.bBox.h:.2f}")

        # 섹션 좌표 추출
        sections = extract_section_coordinates(result.pages)

        # 섹션 정보 저장
        coords_path = output_dir / f"{pdf_path.stem}_sections.json"
        with open(coords_path, "w", encoding="utf-8") as f:
            json.dump(
                {"pdf_path": str(pdf_path), "sections": sections},
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n섹션 좌표 파일 저장됨: {coords_path}")

        # 섹션별 이미지 추출
        crop_pdf_sections(result, sections, output_dir)

        # 결과 요약
        print("\n=== 처리 결과 ===")
        print(f"처리된 페이지 수: {len(result.pages)}")
        print(f"추출된 섹션 수: {len(sections)}")

        # 섹션 좌표 예시 출력
        print("\n=== 섹션 좌표 예시 ===")
        for section in sections[:3]:  # 처음 3개 섹션만 출력
            print(f"\n페이지 {section['page']}:")
            print(f"  헤더 레벨: {section['header_level']}")
            print(f"  헤더 텍스트: {section['header_text']}")
            print(f"  시작 y: {section['y']:.2f}")
            print(f"  끝 y: {section['next_y']:.2f}")
            print(f"  항목 수: {len(section['items'])}")

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_parse()
