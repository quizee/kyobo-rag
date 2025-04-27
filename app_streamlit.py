import streamlit as st
import os
from dotenv import load_dotenv
import nest_asyncio
import re
from PIL import Image

nest_asyncio.apply()
from llama_cloud_services import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai
import json
from scripts.clova_ocr_utils import ocr_image
import requests
import time

# .env 환경변수 로드
load_dotenv()

st.set_page_config(page_title="PDF Chat App", layout="wide")

# 2단 컬럼 레이아웃 (왼쪽: 채팅, 오른쪽: 파일 업로드)
col_chat, col_upload = st.columns([2, 1])

# FAISS DB를 세션에 저장
if "faiss_db" not in st.session_state:
    st.session_state["faiss_db"] = None

with col_chat:
    st.header("💬 교육 자료 생성하기")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    # 채팅 내역 표시
    for msg in st.session_state["chat_history"]:
        st.markdown(f"**{msg['role']}**: {msg['content']}")
    # 사용자 입력
    user_input = st.text_input("메시지를 입력하세요", key="user_input")

    # 검색 버튼 및 결과 출력
    if st.button("검색"):
        if st.session_state["faiss_db"] is not None and user_input:
            retriever = st.session_state["faiss_db"].as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )
            results = retriever.invoke(user_input)
            st.subheader("검색 결과")
            if isinstance(results, list):
                for i, doc in enumerate(results):
                    page = doc.metadata.get("page_number", "알 수 없음")
                    st.markdown(f"**결과 {i+1} (페이지: {page})**")
                    st.code(doc.page_content)
                    st.json(doc.metadata)
            else:
                page = results.metadata.get("page_number", "알 수 없음")
                st.markdown(f"**결과 (페이지: {page})**")
                st.code(results.page_content)
                st.json(results.metadata)
        else:
            st.warning("먼저 FAISS에 임베딩을 저장하고, 검색어를 입력하세요.")

with col_upload:
    st.header("📄 파일 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
    if uploaded_file:
        st.info("파일이 업로드되었습니다. 아래 버튼을 눌러 파싱을 시작하세요.")
        output_dir = "temp_output"
        os.makedirs(output_dir, exist_ok=True)
        md_save_path = os.path.join(
            output_dir, f"temp_{os.path.splitext(uploaded_file.name)[0]}.md"
        )
        # 기존 마크다운 파일 재활용 로직 제거: 항상 새로 파싱
        if st.button("파싱 시작"):
            # 임시 파일로 저장
            temp_pdf_path = os.path.join(output_dir, f"temp_{uploaded_file.name}")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # llama parser 실행
            parser_obj = LlamaParse(
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
                max_pages=7,
            )
            st.info("llama_parse 실행 중...")
            result = parser_obj.parse(temp_pdf_path)
            # split_by_page=True로 md_docs 생성
            md_docs = result.get_markdown_documents(split_by_page=True)
            st.session_state["md_docs"] = md_docs  # 세션에 저장
            st.session_state["llama_parse_result"] = result
            st.success("마크다운 문서 생성 완료!")
            # === 모든 페이지의 full_page_screenshot 저장 ===
            for page in result.pages:
                images = getattr(page, "images", [])
                for img in images:
                    if getattr(img, "type", None) == "full_page_screenshot":
                        img_name = getattr(img, "name", None)
                        if img_name:
                            result.save_image(img_name, output_dir)
            # ===
            if md_docs:
                # 마크다운 파일로 저장
                with open(md_save_path, "w", encoding="utf-8") as f:
                    for doc in md_docs:
                        f.write(doc.text)
                st.session_state["md_save_path"] = md_save_path
                st.info(f"마크다운 파일이 저장되었습니다: {md_save_path}")
        elif "md_docs" in st.session_state:
            st.success("이전 파싱 결과:")
            # 미리보기는 제공하지 않음

    # 임베딩 및 FAISS 저장 버튼
    if "md_save_path" in st.session_state and st.button("임베딩 및 FAISS 저장"):
        md_save_path = st.session_state["md_save_path"]
        with open(md_save_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        # 1. Split (페이지별로)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        all_chunks = []
        md_docs = st.session_state.get("md_docs", [])
        for doc in md_docs:
            chunks = markdown_splitter.split_text(doc.text)
            for chunk in chunks:
                chunk.metadata["page_number"] = doc.metadata.get("page_number")
            all_chunks.extend(chunks)
        # 2. 임베딩
        embeddings = OpenAIEmbeddings()
        # 3. FAISS 벡터 DB 저장
        db = FAISS.from_documents(all_chunks, embeddings)
        st.session_state["faiss_db"] = db  # 세션에 저장
        st.success(
            f"{len(all_chunks)}개 chunk가 임베딩되어 FAISS에 저장되었습니다. 이제 검색이 가능합니다."
        )


def find_header_y_from_ocr(image_path, header_text, api_url, secret_key):
    """
    CLOVA OCR 결과에서 header_text와 가장 유사한 텍스트의 y좌표 반환
    """
    ocr_result = ocr_image(image_path, api_url, secret_key)
    best_y = None
    best_score = 0
    for field in ocr_result["images"][0]["fields"]:
        text = field["inferText"]
        bbox = field["boundingPoly"]["vertices"]
        if header_text in text:
            y_top = min(v["y"] for v in bbox)
            return y_top, text, bbox
    return None, None, None


def clova_ocr_multipart(image_path, api_url, secret_key):
    """
    CLOVA OCR General API를 multipart/form-data 방식으로 호출
    :param image_path: 업로드할 이미지 파일 경로
    :param api_url: CLOVA OCR API Gateway Invoke URL
    :param secret_key: CLOVA OCR Secret Key
    :return: OCR 결과(JSON)
    """
    headers = {"X-OCR-SECRET": secret_key}

    files = {
        "file": open(image_path, "rb"),
        "message": (
            None,
            '{"version": "V2", "requestId": "1234", "timestamp": 1722225600000, "lang": "ko", "images": [{"format": "jpg", "name": "page_7.jpg"}]}',
        ),
    }

    response = requests.post(
        api_url,
        headers=headers,
        files=files,
    )
    response.raise_for_status()
    return response.json()


def normalize_text(text):
    # 한글만 남기고 모두 제거 (숫자, 영어, 특수문자, 띄어쓰기 모두 제거)
    return re.sub(r"[^가-힣]", "", text)


def scale_ocr_y_to_image_y(ocr_y, ocr_result, image_path):
    img = Image.open(image_path)
    img_width, img_height = img.size
    st.warning(img_height)
    ocr_height = ocr_result["images"][0]["convertedImageInfo"]["height"]
    st.warning(ocr_height)
    return (ocr_y / ocr_height) * img_height


def extract_header_y_from_ocr_response(
    ocr_response, header_text, x_gap_threshold=40, image_path=None
):
    fields = ocr_response["images"][0]["fields"]
    lines = []
    current_line = []
    prev_field = None
    for field in fields:
        if not current_line:
            current_line.append(field)
        else:
            prev_right_x = max(v["x"] for v in prev_field["boundingPoly"]["vertices"])
            curr_left_x = min(v["x"] for v in field["boundingPoly"]["vertices"])
            if curr_left_x - prev_right_x > x_gap_threshold:
                lines.append(current_line)
                current_line = [field]
            else:
                current_line.append(field)
        prev_field = field
        if field.get("lineBreak", False):
            lines.append(current_line)
            current_line = []
            prev_field = None
    if current_line:
        lines.append(current_line)

    norm_header = normalize_text(header_text)
    # === 예외 처리: '암진단' -> '삼진단', '암치료' -> '참치료' ===
    ocr_compare_header = norm_header
    if norm_header == "암진단":
        ocr_compare_header = "삼진단"
    elif norm_header == "암치료":
        ocr_compare_header = "참치료"
    # =====================================================
    strict_headers = {"수술", "입원", "재해"}
    for line_fields in lines:
        line_text = "".join(f["inferText"] for f in line_fields)
        norm_line = normalize_text(line_text)
        if norm_header in strict_headers:
            # 예외: 완전 일치만 허용
            if norm_line == norm_header:
                all_y = [
                    v["y"] for f in line_fields for v in f["boundingPoly"]["vertices"]
                ]
                y_top = min(all_y)
                bbox_list = [f["boundingPoly"]["vertices"] for f in line_fields]
                if image_path is not None:
                    y_top_scaled = scale_ocr_y_to_image_y(
                        y_top, ocr_response, image_path
                    )
                    return y_top_scaled, line_text, bbox_list
                else:
                    return y_top, line_text, bbox_list
        else:
            # 일반: 부분 문자열 + 라인 길이 제한
            if norm_header in norm_line and len(norm_line) <= len(norm_header) * 1.5:
                all_y = [
                    v["y"] for f in line_fields for v in f["boundingPoly"]["vertices"]
                ]
                y_top = min(all_y)
                bbox_list = [f["boundingPoly"]["vertices"] for f in line_fields]
                if image_path is not None:
                    y_top_scaled = scale_ocr_y_to_image_y(
                        y_top, ocr_response, image_path
                    )
                    return y_top_scaled, line_text, bbox_list
                else:
                    return y_top, line_text, bbox_list
    return None, None, None


def crop_image_by_y(image_path, y_start, y_end, output_path):
    img = Image.open(image_path)
    img_width, img_height = img.size
    y_start = int(max(0, y_start - 15))
    y_end = int(min(img_height, y_end))
    x_start = max(0, 30)
    x_end = min(img_width, img_width - 30)
    if y_end <= y_start or x_end <= x_start:
        return None  # 빈 이미지 방지
    cropped = img.crop((x_start, y_start, x_end, y_end))
    cropped.save(output_path)
    return output_path


# 사용 예시 (Streamlit sidebar에서 테스트)
if st.sidebar.button("테스트: 헤더 y좌표 찾기 (CLOVA OCR, multipart)"):
    test_meta = {
        "Header 1": "미리 체크해보는 교보마이플랜건강보험[2411](무배당)",
        "Header 2": "참치료",  # 암진단 암치료 수술 입원 재해
        "page_number": 7,
        # 암치료 - 참치료
    }
    st.write(test_meta)
    page_img_path = os.path.join("temp_output", f"page_{test_meta['page_number']}.jpg")
    api_url = "https://8vb79ndbzb.apigw.ntruss.com/custom/v1/41373/3a26df9469f0c22bb024a70d5cc5e11a9fb28f6ea21993ba8eee769f4dda9216/general"
    secret_key = os.getenv("CLOVA_OCR_SECRET_KEY")
    st.write(page_img_path, api_url, secret_key)
    try:
        ocr_result = clova_ocr_multipart(page_img_path, api_url, secret_key)
        st.write(ocr_result)
        # 헤더 위치 찾기
        header_text = test_meta["Header 2"]
        y, matched_text, bbox = extract_header_y_from_ocr_response(
            ocr_result, header_text, image_path=page_img_path
        )
        if y is not None:
            st.sidebar.success(f"헤더 '{header_text}'의 y좌표: {y}")
            st.sidebar.write(f"매칭된 텍스트: {matched_text}")
            st.sidebar.write(f"bbox: {bbox}")
            # === 이미지 crop ===
            img = Image.open(page_img_path)
            img_height = img.size[1]
            y_start = y
            y_end = img_height  # 예시: 헤더부터 끝까지 crop
            cropped_path = os.path.join(
                "temp_output", f"page_{test_meta['page_number']}_cropped.jpg"
            )
            crop_image_by_y(page_img_path, y_start, y_end, cropped_path)
            st.sidebar.info(f"Crop된 이미지 저장: {cropped_path}")
            st.sidebar.image(cropped_path)
            # ===================
        else:
            st.sidebar.error("헤더를 찾지 못했습니다.")
    except Exception as e:
        st.sidebar.error(f"OCR 호출 실패: {e}")


def extract_headers_from_llamaparse_items(items, page_number):
    """LlamaParse items에서 header(page, level, text) 리스트 추출 (object 타입 대응)"""
    headers = []
    for item in items:
        if getattr(item, "type", None) == "heading":
            headers.append(
                {
                    "page": page_number,
                    "level": getattr(item, "lvl", 1),
                    "text": getattr(item, "value", ""),
                }
            )
    return headers


def build_header_dictionary(headers, ocr_results_dict, output_dir="temp_output"):
    """
    headers: [{page, level, text}, ...]
    ocr_results_dict: {page_number: ocr_result, ...}
    return: {"header_text": {page, level, text, y, crop_image_path}}
    """
    header_dict = {}
    from collections import defaultdict

    # 페이지별로 그룹화
    page_to_headers = defaultdict(list)
    for header in headers:
        if header["y"] is not None:
            page_to_headers[header["page"]].append(header)
    for page_number, page_headers in page_to_headers.items():
        page_img_path = os.path.join(output_dir, f"page_{page_number}.jpg")
        img = Image.open(page_img_path)
        img_height = img.size[1]
        # level별로 그룹화
        level_to_headers = defaultdict(list)
        for h in page_headers:
            level_to_headers[h["level"]].append(h)
        for level, headers_in_level in level_to_headers.items():
            headers_in_level = sorted(headers_in_level, key=lambda h: h["y"])
            for i, header in enumerate(headers_in_level):
                y_start = header["y"]
                if i < len(headers_in_level) - 1:
                    y_end = headers_in_level[i + 1]["y"]
                else:
                    y_end = img_height
                if y_end <= y_start:
                    continue
                out_path = os.path.join(
                    output_dir,
                    f"page_{page_number}_level{level}_header_{i+1}_{header['text'][:10]}.jpg",
                )
                crop_image_by_y(page_img_path, y_start, y_end, out_path)
                key = f"{header['text']}"
                header_dict[key] = {
                    "page": page_number,
                    "level": level,
                    "text": header["text"],
                    "y": y_start,
                    "crop_image_path": out_path,
                }
    return header_dict


# 헤더 dictionary 생성 Streamlit 버튼
if st.sidebar.button("전체 파일을 PPT 로 파싱"):
    result = st.session_state.get("llama_parse_result")
    if not result:
        st.sidebar.error("llama_parse 결과가 없습니다.")
    else:
        api_url = "https://8vb79ndbzb.apigw.ntruss.com/custom/v1/41373/3a26df9469f0c22bb024a70d5cc5e11a9fb28f6ea21993ba8eee769f4dda9216/general"
        secret_key = os.getenv("CLOVA_OCR_SECRET_KEY")
        headers = []
        ocr_results_dict = {}
        output_dir = "temp_output"
        for page_idx, page in enumerate(result.pages):
            page_number = getattr(page, "page", page_idx + 1)
            items = getattr(page, "items", [])
            page_headers = extract_headers_from_llamaparse_items(items, page_number)
            # OCR 결과 미리 저장
            page_img_path = os.path.join(output_dir, f"page_{page_number}.jpg")
            ocr_results_dict[page_number] = clova_ocr_multipart(
                page_img_path, api_url, secret_key
            )
            # 헤더별 y좌표 추출
            for header in page_headers:
                ocr_result = ocr_results_dict.get(page_number)
                y, matched_text, bbox_list = extract_header_y_from_ocr_response(
                    ocr_result, header["text"], image_path=page_img_path
                )
                header["y"] = y
            headers.extend(page_headers)
        header_dict = build_header_dictionary(
            headers, ocr_results_dict, output_dir=output_dir
        )
        st.session_state["header_dict"] = header_dict
        st.sidebar.success("헤더 dictionary 생성 및 이미지 crop 완료!")
        st.write(header_dict)
