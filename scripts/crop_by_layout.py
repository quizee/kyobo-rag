import json
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from scripts.upstage_parser import UpstageParser
from dotenv import load_dotenv
import base64
from io import BytesIO
from openai import OpenAI


def serialize_layout(layout):
    if layout is None:
        return None
    if isinstance(layout, (int, float, str, bool)):
        return layout
    if isinstance(layout, (list, tuple)):
        return [serialize_layout(item) for item in layout]
    if isinstance(layout, dict):
        return {k: serialize_layout(v) for k, v in layout.items()}
    try:
        return {
            k: serialize_layout(v)
            for k, v in layout.__dict__.items()
            if not k.startswith("_")
        }
    except:
        return str(layout)


def crop_pdf_by_layout(llama_json_path, output_dir):
    llama_json_path = Path(llama_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # llama_parse 결과 로드
    with open(llama_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for page_idx, page in enumerate(data["pages"]):
        page_num = page_idx + 1
        image_path = output_dir / f"page_{page_num}.jpg"
        if not image_path.exists():
            print(f"페이지 이미지가 없습니다: {image_path}")
            continue
        page_image = Image.open(image_path)
        img_width, img_height = page_image.size
        # PDF 좌표계 높이 (llama_parse 기준)
        pdf_height = page_image.height  # 이미지와 PDF 비율이 다를 수 있음
        # heading 추출 및 y로 정렬
        headers = [
            {"text": item["value"], "y": item["bBox"]["y"], "height": item["bBox"]["h"]}
            for item in page["items"]
            if item["type"] == "heading"
        ]
        if not headers:
            continue
        headers.sort(key=lambda x: x["y"])
        # 섹션별 crop
        for i, header in enumerate(headers):
            y_start = int(header["y"] / pdf_height * img_height)
            if i < len(headers) - 1:
                y_end = int(headers[i + 1]["y"] / pdf_height * img_height)
            else:
                y_end = img_height
            if y_end <= y_start:
                continue
            section_image = page_image.crop((0, y_start, img_width, y_end))
            title = header["text"].replace(" ", "-").replace("/", "-").replace("?", "")
            image_path = output_dir / f"page{page_num}_{title}.png"
            section_image.save(image_path)
            print(
                f"[레이아웃] page{page_num} '{header['text']}' → {image_path} (y: {y_start}-{y_end})"
            )


def correct_heading_y_with_openai(
    llama_json_path, openai_client, output_json_path, output_dir
):
    """
    OpenAI(GPT-4V)로 llama_parse가 저장한 페이지별 전체 이미지 파일과 heading 리스트(JSON)를 주고,
    각 heading의 bbox.y 값을 이미지에서 시각적으로 찾아 보정하는 함수
    """
    import json
    import base64
    from pathlib import Path

    output_dir = Path(output_dir)
    with open(llama_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corrected_data = data.copy()

    for page_idx, page in enumerate(data["pages"]):
        page_num = page_idx + 1
        # llama_parse가 저장한 전체 이미지 파일 경로
        image_path = output_dir / f"page_{page_num}.jpg"
        if not image_path.exists():
            print(f"[OpenAI] page{page_num} 이미지 없음, 건너뜀")
            continue
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        # heading만 추출
        headings = [
            {"type": item["type"], "value": item["value"], "bBox": item["bBox"]}
            for item in page["items"]
            if item["type"] == "heading"
        ]
        if not headings:
            continue
        # 프롬프트 구성
        prompt = (
            "아래는 PDF 한 페이지의 전체 이미지와, 해당 페이지에서 추출한 heading 리스트(JSON)입니다.\n"
            "각 heading의 'value'가 이미지에서 실제로 보이는 위치(상단 y좌표)를 찾아, bBox의 y 값을 실제 위치로 보정해 주세요.\n"
            "JSON의 나머지 필드는 그대로 두고, y 값만 보정해서 반환해 주세요. 반드시 JSON만 반환해 주세요.\n"
            '코드블록("```json" 등)은 절대 사용하지 마세요. 오직 JSON만 반환하세요.\n'
            f"[이미지: base64 PNG]\n[JSON: {json.dumps({'items': headings}, ensure_ascii=False)}]"
        )
        # OpenAI API 호출
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 PDF 레이아웃 분석 전문가입니다.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
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
                max_tokens=1500,
            )
            content = response.choices[0].message.content
            print("OpenAI 응답:", content)  # 디버깅용
            corrected_json = json.loads(content)
            # 보정된 heading을 value로 매핑
            corrected_headings = {
                item["value"]: item
                for item in corrected_json["items"]
                if item["type"] == "heading"
            }
            # 원본 JSON의 각 heading에 대해 value로 매칭해서 y값만 덮어쓰기
            for orig in corrected_data["pages"][page_idx]["items"]:
                if orig["type"] == "heading":
                    value = orig["value"]
                    if value in corrected_headings:
                        orig["bBox"]["y"] = corrected_headings[value]["bBox"]["y"]
            print(f"[OpenAI] page{page_num} heading y 보정 완료")
        except Exception as e:
            print(f"[OpenAI] page{page_num} 보정 실패: {e}")
            print("실패한 content:", content if "content" in locals() else "없음")
            continue
    # 보정된 전체 JSON 저장
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=2)
    print(f"OpenAI 보정 결과 저장: {output_json_path}")


if __name__ == "__main__":
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="upstage parser 레이아웃 기반 PDF 섹션 crop"
    )
    parser.add_argument("--pdf", required=True, help="PDF 파일 경로")
    parser.add_argument(
        "--upstage_json",
        default=None,
        help="upstage parser 결과 JSON 경로 (없으면 자동 생성)",
    )
    parser.add_argument(
        "--output_dir", default="output_by_layout", help="이미지 저장 폴더"
    )
    args = parser.parse_args()

    upstage_json_path = args.upstage_json

    # 1. upstage parser 실행 및 이미지 저장 (upstage_json_path가 없을 때만)
    if upstage_json_path is None:
        Path(args.output_dir).mkdir(exist_ok=True)
        parser_obj = UpstageParser()
        print("upstage parser 실행 중...")
        result = parser_obj.parse(str(args.pdf))
        upstage_json_path = str(Path(args.output_dir) / "upstage_parse_result.json")
        parser_obj.save_result(result, upstage_json_path)
        print(f"upstage parser 결과 저장: {upstage_json_path}")

        # PDF를 이미지로 변환
        images = convert_from_path(args.pdf)
        for page_num, image in enumerate(images, 1):
            image_path = os.path.join(args.output_dir, f"page_{page_num}.jpg")
            image.save(image_path, "JPEG")
            print(f"페이지 {page_num} 전체 이미지 저장: {image_path}")

    # 2. 이후 단계는 upstage_json_path가 있든 없든 공통적으로 실행
    corrected_json_path = str(
        Path(args.output_dir) / "upstage_parse_corrected_by_openai.json"
    )

    # 보정된 JSON 파일이 없으면 먼저 보정 실행
    if not Path(corrected_json_path).exists():
        openai_client = OpenAI()
        correct_heading_y_with_openai(
            llama_json_path=upstage_json_path,
            openai_client=openai_client,
            output_json_path=corrected_json_path,
            output_dir=args.output_dir,
        )

    # crop 함수 실행 (보정된 JSON 사용)
    crop_pdf_by_layout(corrected_json_path, args.output_dir)
