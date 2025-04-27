import os
import fitz  # PyMuPDF
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
import json
import math
from pydantic.json import pydantic_encoder

try:
    from fastapi.encoders import jsonable_encoder
except ImportError:

    def jsonable_encoder(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        elif isinstance(obj, (list, tuple)):
            return [jsonable_encoder(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: jsonable_encoder(v) for k, v in obj.items()}
        else:
            return obj


PDF_PATH = "/Users/jeeyoonlee/Desktop/kyobo-project/data/상품설명/1.교보마이플랜건강보험 [2409](무배당).pdf"
OUTPUT_DIR = "output"
CROPPED_DIR = os.path.join(OUTPUT_DIR, "cropped_headers")
os.makedirs(CROPPED_DIR, exist_ok=True)

# OpenAI API Key 환경변수에서 읽기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

# 전체 페이지 수를 먼저 구하기 위해 fitz로 PDF 열기
doc = fitz.open(PDF_PATH)
total_pages = doc.page_count

BATCH_SIZE = 2

for batch_start in range(0, total_pages, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE - 1, total_pages - 1)
    page_range_str = f"{batch_start}-{batch_end}"
    print(f"Processing pages: {page_range_str}")

    # marker-pdf로 JSON 파싱 (LLM 사용, page_range 적용)
    config = {
        "output_format": "json",
        "use_llm": True,
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_api_key": OPENAI_API_KEY,
        "page_range": page_range_str,
    }
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(PDF_PATH)
    if isinstance(rendered, tuple):
        json_data = rendered[0]
    else:
        json_data = rendered

    # json_data를 파일로 저장 (페이지 범위별로 파일명 다르게)
    json_save_path = os.path.join(
        OUTPUT_DIR, f"parsed_marker_json_{batch_start+1}_{batch_end+1}.json"
    )
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(jsonable_encoder(json_data), f, ensure_ascii=False, indent=2)
    print(f"marker-pdf 파싱 결과를 {json_save_path}에 저장했습니다.")

    # 실제 페이지 리스트 추출
    pages = getattr(json_data, "children", None)
    if pages is None:
        raise ValueError("json_data.children이 존재하지 않습니다!")

    print("pages type:", type(pages))
    print("pages length:", len(pages))
    print("First page type:", type(pages[0]))
    print("First page content (truncated):", str(pages[0])[:1000])

    # 2. 헤더별 polygon 좌표 추출
    header_blocks = []
    for idx, page in enumerate(pages):
        if isinstance(page, dict):
            page_id = page.get("id")
            page_num = int(page_id.split("/")[2]) if page_id else idx
            children = page.get("children", [])
        elif hasattr(page, "id") and hasattr(page, "children"):
            page_id = getattr(page, "id", None)
            page_num = int(page_id.split("/")[2]) if page_id else idx
            children = getattr(page, "children", [])
        else:
            print(f"Warning: page {idx} is {type(page)}. Skipping.")
            continue

        for block in children:
            if (
                isinstance(block, dict) and block.get("block_type") == "SectionHeader"
            ) or (
                hasattr(block, "block_type")
                and getattr(block, "block_type") == "SectionHeader"
            ):
                header_blocks.append(
                    {
                        "page_num": page_num,
                        "header_id": (
                            getattr(block, "id", None)
                            if hasattr(block, "id")
                            else block.get("id")
                        ),
                        "polygon": (
                            getattr(block, "polygon", None)
                            if hasattr(block, "polygon")
                            else block.get("polygon")
                        ),
                        "html": (
                            getattr(block, "html", None)
                            if hasattr(block, "html")
                            else block.get("html")
                        ),
                    }
                )

    # 3. PDF에서 해당 polygon 영역만 이미지로 저장
    for header in header_blocks:
        page = doc[header["page_num"]]
        x_coords = [pt[0] for pt in header["polygon"]]
        y_coords = [pt[1] for pt in header["polygon"]]
        rect = fitz.Rect(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        pix = page.get_pixmap(clip=rect, dpi=300)
        import re

        header_name = re.sub(r"[^\w\-_\. ]", "_", header["html"])[:30]
        out_path = os.path.join(
            CROPPED_DIR, f"page{header['page_num']}_{header_name}.png"
        )
        pix.save(out_path)
        print(f"Saved: {out_path}")

print("헤더별 영역 이미지 저장 완료!")
