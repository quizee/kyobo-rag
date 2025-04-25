import pytesseract
from typing import List, Dict
from statistics import mean, stdev
import os
from pdf2image import convert_from_path
from pathlib import Path


def analyze_headers(headers: List[Dict]) -> Dict:
    """헤더들의 통계적 특성 분석"""
    # 레벨별 그룹화
    level_groups = {}
    for header in headers:
        level = header["lvl"]
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(header)

    # 레벨별 통계 계산
    stats = {}
    for level, group in level_groups.items():
        heights = [h["bBox"]["h"] for h in group]
        y_values = [h["bBox"]["y"] for h in group]
        y_diffs = [y2 - y1 for y1, y2 in zip(y_values[:-1], y_values[1:])]

        stats[level] = {
            "avg_height": mean(heights) if heights else 0,
            "std_height": stdev(heights) if len(heights) > 1 else 0,
            "avg_y_diff": mean(y_diffs) if y_diffs else 0,
            "std_y_diff": stdev(y_diffs) if len(y_diffs) > 1 else 0,
        }

    return stats


def convert_pdf_to_images(pdf_path, output_dir):
    """PDF를 이미지로 변환하고 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path)
    image_paths = []

    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        image.save(image_path, "JPEG")
        image_paths.append(image_path)

    return image_paths


def normalize_text(text):
    """텍스트 정규화: 공백 제거, 특수문자 처리"""
    # 한글, 영문, 숫자만 남기고 모두 제거
    normalized = "".join(char for char in text if char.isalnum() or char.isspace())
    # 연속된 공백을 하나로 치환
    normalized = " ".join(normalized.split())
    # 모든 공백 제거
    normalized = "".join(normalized.split())
    return normalized


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
                print(f"텍스트: {text}")
                print(
                    f"  위치: x={data['left'][i]}, y={data['top'][i]}, w={data['width'][i]}, h={data['height'][i]}"
                )
                print(f"  신뢰도: {int(data['conf'][i])}")

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

    print("\n그룹화된 라인:")
    for line in lines:
        print(f"라인: {line['text']}")
        print(f"  y좌표: {line['y']}")
        print(f"  평균 신뢰도: {line['confidence']:.1f}")

    return lines


def find_matching_position(header_text, text_areas):
    normalized_header = normalize_text(header_text)
    best_match = None
    best_similarity = 0
    best_substring = ""

    print(f"\n[매칭 시도] 찾으려는 헤더: '{header_text}'")
    print(f"정규화된 헤더: '{normalized_header}'")

    for text_area in text_areas:
        normalized_text = normalize_text(text_area["text"])
        print(f"\n  비교 대상: '{text_area['text']}'")
        print(f"  정규화된 텍스트: '{normalized_text}'")

        # 유사도와 가장 긴 공통 부분 문자열 계산
        similarity, longest_substring = calculate_similarity(
            normalized_header, normalized_text
        )
        print(f"  유사도: {similarity:.3f}")
        print(
            f"  가장 긴 연속 일치 문자열: '{longest_substring}' (길이: {len(longest_substring)})"
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = text_area
            best_substring = longest_substring
            print("  => 현재까지 가장 좋은 매칭!")

    print(f"\n최종 매칭된 텍스트: '{best_match['text'] if best_match else 'None'}'")
    print(f"최종 유사도: {best_similarity:.3f}")

    return best_match, best_similarity


def correct_header_positions_with_fixed_first(headers, image, image_height, pdf_height):
    """헤더 위치 보정 (첫 번째 헤더는 y=0으로 고정)"""
    # 이미지에서 텍스트 위치 찾기
    lines = find_text_positions(image)

    corrected_positions = []

    # 첫 번째 헤더는 y=0으로 고정
    first_header = headers[0]
    corrected_positions.append(
        {"text": first_header["text"], "level": first_header["level"], "y": 0}
    )

    # 나머지 헤더들의 위치 보정
    for header in headers[1:]:
        # 헤더 텍스트와 가장 잘 매칭되는 위치 찾기
        best_match, best_similarity = find_matching_position(header["text"], lines)

        if best_match is not None:
            # 이미지 좌표를 PDF 좌표로 변환
            pdf_y = (best_match["y"] / image_height) * pdf_height
            corrected_positions.append(
                {"text": header["text"], "level": header["level"], "y": pdf_y}
            )

    return corrected_positions


def extract_sections(headers, page_height):
    """유효한 섹션 추출"""
    if not headers:
        return []

    # y좌표로 정렬
    sorted_headers = sorted(headers, key=lambda x: x["y"])

    sections = []
    for i, header in enumerate(sorted_headers):
        # 섹션의 시작 위치는 현재 헤더의 y좌표
        start_y = header["y"]

        # 섹션의 끝 위치는 다음 헤더의 y좌표 또는 페이지 끝
        end_y = (
            sorted_headers[i + 1]["y"] if i < len(sorted_headers) - 1 else page_height
        )

        sections.append(
            {
                "text": header["text"],
                "level": header["level"],
                "start_y": start_y,
                "end_y": end_y,
            }
        )

    return sections


def convert_pdf_to_image(pdf_path, page_number):
    """PDF 페이지를 이미지로 변환"""
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    return images[0]


def crop_pdf_sections(pdf_path, page_number, headers, output_dir):
    """PDF 페이지를 섹션별로 자르기"""
    # PDF 페이지를 이미지로 변환
    image = convert_pdf_to_image(pdf_path, page_number)
    image_height = image.height

    # 원본 PDF 높이 (A4 기준 약 842pt)
    pdf_height = 842

    print("\n원본 헤더 정보:")
    for header in headers:
        print(f"- [{header['level']}] {header['text']}: y={header['y']:.2f}")

    # 헤더 위치 보정 (첫 번째 헤더는 y=0으로 고정)
    corrected_headers = correct_header_positions_with_fixed_first(
        headers, image, image_height, pdf_height
    )

    if not corrected_headers:
        print("Warning: 매칭된 헤더가 없습니다!")
        return

    # 섹션 추출
    sections = extract_sections(corrected_headers, pdf_height)

    print("\n추출된 섹션:")
    for section in sections:
        print(f"- [{section['level']}] {section['text']}")
        print(f"  시작: {section['start_y']:.2f}, 끝: {section['end_y']:.2f}")
        print(f"  높이: {section['end_y'] - section['start_y']:.2f}")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 각 섹션을 이미지로 저장
    for section in sections:
        print(f"\n섹션 이미지 저장: {section['text']}")
        print(f"PDF 좌표 - 시작: {section['start_y']:.2f}, 끝: {section['end_y']:.2f}")

        # PDF 좌표를 이미지 좌표로 변환
        start_y = int((section["start_y"] / pdf_height) * image_height)
        end_y = int((section["end_y"] / pdf_height) * image_height)

        print(f"이미지 좌표 - 시작: {start_y}, 끝: {end_y}")

        # 이미지 자르기
        section_image = image.crop((0, start_y, image.width, end_y))

        # 파일 이름 생성
        filename = (
            section["text"]
            .replace(" ", "-")
            .replace("/", "_")
            .replace("(", "_")
            .replace(")", "_")
        )
        filepath = os.path.join(output_dir, f"page{page_number}_{filename}.png")

        # 이미지 저장
        section_image.save(filepath)
        print(f"저장 완료: {filepath}")


def test_crop():
    """테스트 실행"""
    pdf_path = "/Users/jeeyoonlee/Desktop/kyobo-project/data/상품설명/1.교보마이플랜건강보험 [2409](무배당).pdf"
    page_number = 3

    # 테스트용 헤더 정보
    headers = [
        {"text": "미리 체크해 보는 교보마이플랜건강보험 2411", "level": 1, "y": 0},
        {"text": "CHECK 01 무해약환급금형이란?", "level": 2, "y": 127.89},
        {
            "text": "CHECK 02 무해약환급금형 여부에 따른 상품 비교",
            "level": 2,
            "y": 128.44,
        },
        {"text": "해약환급금 도해 비교(순수보장형)", "level": 3, "y": 353.89},
        {"text": "보험료 및 해약환급금 비교", "level": 3, "y": 560.89},
    ]

    # 섹션 추출 및 저장
    crop_pdf_sections(pdf_path, page_number, headers, "test_output/sections")


if __name__ == "__main__":
    test_crop()
