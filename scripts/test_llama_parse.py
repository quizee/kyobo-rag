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
import pytesseract

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
    """텍스트 정규화: 공백 제거"""
    # 한글, 영문, 숫자만 남기고 모두 제거
    normalized = "".join(char for char in text if char.isalnum() or char.isspace())
    # 연속된 공백을 하나로 치환
    normalized = " ".join(normalized.split())
    # 모든 공백 제거
    normalized = "".join(normalized.split())
    return normalized


def calculate_similarity(text1, text2):
    # 1. 연속 문자열 매칭 (70% 비중)
    longest_substring = find_longest_common_substring(text1, text2)
    substring_ratio = len(longest_substring) / max(len(text1), len(text2))

    # 2. 문자 기반 Jaccard 유사도 (30% 비중)
    chars1 = set(text1)
    chars2 = set(text2)
    jaccard = len(chars1 & chars2) / len(chars1 | chars2)

    # 가중치 적용 (연속 문자열 매칭에 더 높은 가중치)
    similarity = (0.7 * substring_ratio) + (0.3 * jaccard)

    return similarity, longest_substring


def find_longest_common_substring(s1, s2):
    # 두 문자열에서 가장 긴 공통 부분 문자열 찾기
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0

    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x

    return s1[x_longest - longest : x_longest]


def group_texts_by_line(texts, y_threshold=5):
    """같은 y좌표(±threshold)에 있는 텍스트들을 하나의 라인으로 그룹화"""
    if not texts:
        return []

    # y좌표를 기준으로 정렬
    sorted_texts = sorted(texts, key=lambda x: x["y"])

    lines = []
    current_line = [sorted_texts[0]]
    current_y = sorted_texts[0]["y"]

    for text in sorted_texts[1:]:
        if abs(text["y"] - current_y) <= y_threshold:
            current_line.append(text)
        else:
            # x좌표로 정렬하여 라인의 텍스트들을 순서대로 결합
            sorted_line = sorted(current_line, key=lambda x: x["x"])
            line_text = " ".join(t["text"] for t in sorted_line)
            line_y = sum(t["y"] for t in current_line) / len(current_line)
            lines.append(
                {
                    "text": line_text,
                    "y": line_y,
                    "confidence": sum(t["confidence"] for t in current_line)
                    / len(current_line),
                }
            )
            current_line = [text]
            current_y = text["y"]

    # 마지막 라인 처리
    if current_line:
        sorted_line = sorted(current_line, key=lambda x: x["x"])
        line_text = " ".join(t["text"] for t in sorted_line)
        line_y = sum(t["y"] for t in current_line) / len(current_line)
        lines.append(
            {
                "text": line_text,
                "y": line_y,
                "confidence": sum(t["confidence"] for t in current_line)
                / len(current_line),
            }
        )

    return lines


def find_text_positions(image):
    """이미지에서 텍스트 위치 찾기"""
    # OCR 설정 변경
    custom_config = r"--oem 3 --psm 3 -l kor+eng --dpi 300"

    # OCR 실행
    data = pytesseract.image_to_data(
        image, config=custom_config, output_type=pytesseract.Output.DICT
    )

    # 텍스트 정보 추출 (신뢰도 임계값 조정)
    texts = []
    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 60:  # 신뢰도 임계값 하향
            text = data["text"][i].strip()
            if text:  # 빈 문자열이 아닌 경우만 포함
                # print(f"텍스트: {text}")
                # print(f"  위치: x={data['left'][i]}, y={data['top'][i]}, w={data['width'][i]}, h={data['height'][i]}")
                # print(f"  신뢰도: {int(data['conf'][i])}")
                texts.append(
                    {
                        "text": text,
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                        "confidence": int(data["conf"][i]),
                    }
                )

    # 텍스트를 라인별로 그룹화 (y좌표 기준 병합 범위 확대)
    lines = group_texts_by_line(texts, y_threshold=15)  # 임계값 증가

    # print("\n그룹화된 라인:")
    # for line in lines:
    #     print(f"라인: {line['text']}")
    #     print(f"  y좌표: {line['y']}")
    #     print(f"  평균 신뢰도: {line['confidence']:.1f}")

    return lines


def find_matching_position(header_text, text_areas):
    normalized_header = normalize_text(header_text)
    best_match = None
    best_similarity = 0
    best_substring = ""

    for text_area in text_areas:
        normalized_text = normalize_text(text_area["text"])
        similarity, longest_substring = calculate_similarity(
            normalized_header, normalized_text
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = text_area
            best_substring = longest_substring

    # 매칭 대상 헤더와 best match만 출력
    if best_match:
        print(
            f"[매칭] 헤더: '{header_text}' | OCR: '{best_match['text']}' | 유사도: {best_similarity:.3f} | y: {best_match['y']}"
        )
    else:
        print(f"[매칭] 헤더: '{header_text}' | OCR: (매칭 실패)")

    return best_match, best_similarity


def correct_header_positions_with_fixed_first(headers, image, image_height, pdf_height):
    """헤더 위치 보정 (첫 번째 헤더는 y=0으로 고정)"""
    # 이미지에서 텍스트 위치 찾기
    lines = find_text_positions(image)

    corrected_positions = []

    # 첫 번째 헤더는 y=0으로 고정
    first_header = headers[0]
    corrected_positions.append(
        {
            "page": first_header["page"],
            "header_level": first_header["header_level"],
            "header_text": first_header["header_text"],
            "y": 0,
            "x": first_header["x"],
            "width": first_header["width"],
            "height": first_header["height"],
            "items": first_header["items"],
        }
    )

    # 나머지 헤더들의 위치 보정
    for header in headers[1:]:
        # 헤더 텍스트와 가장 잘 매칭되는 위치 찾기
        best_match, best_similarity = find_matching_position(
            header["header_text"], lines
        )

        if best_match is not None:
            # 이미지 좌표를 PDF 좌표로 변환
            pdf_y = (best_match["y"] / image_height) * pdf_height
            corrected_positions.append(
                {
                    "page": header["page"],
                    "header_level": header["header_level"],
                    "header_text": header["header_text"],
                    "y": pdf_y,
                    "x": header["x"],
                    "width": header["width"],
                    "height": header["height"],
                    "items": header["items"],
                }
            )

    return corrected_positions


def crop_pdf_sections(page_image, headers, output_dir, page_num, pdf_height):
    """한 페이지의 섹션별 이미지를 OCR 기반으로 추출"""
    img_width, img_height = page_image.size

    # 헤더 y좌표 보정 (OCR)
    corrected_sections = correct_header_positions_with_fixed_first(
        headers, page_image, img_height, pdf_height
    )

    # 시작 y ~ 다음 헤더 y, 마지막은 페이지 끝까지
    for i, section in enumerate(corrected_sections):
        scale_factor = img_height / pdf_height
        y_start = int(section["y"] * scale_factor)
        if i < len(corrected_sections) - 1:
            y_end = int(corrected_sections[i + 1]["y"] * scale_factor)
        else:
            y_end = img_height

        if y_start < 0:
            y_start = 0
        if y_end > img_height:
            y_end = img_height
        if y_end <= y_start:
            print(f"섹션 좌표 오류 - 시작: {y_start}, 끝: {y_end}, 높이: {img_height}")
            continue

        # 파일명 생성
        title = (
            section["header_text"].replace(" ", "-").replace("?", "").replace("/", "-")
        )
        image_path = os.path.join(output_dir, f"page{page_num}_{title}.png")

        # 이미지 저장
        section_image = page_image.crop((0, y_start, img_width, y_end))
        section_image.save(image_path, "PNG")
        print(f"섹션 이미지 저장됨: {image_path} (y: {y_start}-{y_end})")


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


def test_parse():
    pdf_path = Path(
        "/Users/jeeyoonlee/Desktop/kyobo-project/data/상품설명/1.교보마이플랜건강보험 [2409](무배당).pdf"
    )
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        num_workers=1,
        verbose=True,
        language="ko",
        extract_layout=True,
        premium_mode=True,
        continuous_mode=False,
        extract_charts=True,
        save_images=True,
        output_tables_as_HTML=False,
        max_pages=6,
    )

    print("PDF 파싱 시작...")
    result = parser.parse(str(pdf_path))
    print("파싱 완료!")

    # 1. llama_parse result를 serialize_layout로 JSON 저장
    raw_json_path = output_dir / "llama_parse_raw.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(serialize_layout(result), f, ensure_ascii=False, indent=2)
    print(f"llama_parse 원본 JSON 저장: {raw_json_path}")

    # 2. 페이지별 보정된 y좌표를 저장할 dict
    page_corrections = {}

    for page_num, page in enumerate(result.pages, 1):
        if not hasattr(page, "items") or not page.items:
            continue

        print(f"\n=== {page_num}페이지 분석 ===")
        items = page.items
        headers = [
            {
                "page": page_num,
                "header_level": item.lvl,
                "header_text": item.value,
                "y": item.bBox.y,
                "x": item.bBox.x,
                "width": item.bBox.w,
                "height": item.bBox.h,
                "items": [],
            }
            for item in items
            if item.type == "heading"
        ]
        if not headers:
            print("헤더 없음")
            continue

        # 페이지 전체 이미지
        full_page_screenshots = [
            img
            for img in page.images
            if getattr(img, "type", None) == "full_page_screenshot"
        ]
        if not full_page_screenshots:
            print("페이지 스크린샷 없음")
            continue
        page_image = Image.open(
            result.save_image(full_page_screenshots[0].name, str(output_dir))
        )
        pdf_height = full_page_screenshots[0].height

        # 보정된 섹션 리스트 저장
        corrected_sections = correct_header_positions_with_fixed_first(
            headers, page_image, page_image.height, pdf_height
        )
        page_corrections[page_num] = corrected_sections

        crop_pdf_sections(page_image, headers, output_dir, page_num, pdf_height)

    # 3. 보정된 y좌표를 llama_parse JSON에 반영하여 새 파일로 저장
    corrected_json_path = output_dir / "llama_parse_corrected.json"
    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for page_num, corrected_sections in page_corrections.items():
        page = data["pages"][page_num - 1]
        for corr in corrected_sections:
            for item in page["items"]:
                if item["type"] == "heading" and item["value"] == corr["header_text"]:
                    item["bBox"]["y"] = corr["y"]

    with open(corrected_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"보정된 y좌표 반영 JSON 저장: {corrected_json_path}")


if __name__ == "__main__":
    test_parse()
