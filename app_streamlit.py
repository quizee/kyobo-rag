import streamlit as st
import os
from dotenv import load_dotenv
import nest_asyncio
import re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

from PIL import Image as PILImage
from pptx.enum.text import PP_ALIGN
import time
import pandas as pd


nest_asyncio.apply()
from llama_cloud_services import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai
import json
from scripts.clova_ocr_utils import ocr_image
import requests

# .env 환경변수 로드
load_dotenv()

st.set_page_config(page_title="PDF Chat App", layout="wide")


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
    ocr_height = ocr_result["images"][0]["convertedImageInfo"]["height"]
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
            if (
                ocr_compare_header in norm_line
                and len(norm_line) <= len(ocr_compare_header) * 1.5
            ):
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
                    "page_number": page_number,
                }
            )
    return headers


def build_header_dictionary(headers, ocr_results_dict, output_dir="temp_output"):
    """
    headers: [{page, level, text}, ...]
    ocr_results_dict: {page_number: ocr_result, ...}
    return: {"header_text": {page, level, text, y, crop_image_path, Header 1, Header 2, Header 3}}
    """
    header_dict = {}
    from collections import defaultdict

    # 페이지별로 그룹화
    page_to_headers = defaultdict(list)
    for header in headers:
        if header["y"] is not None:
            page_to_headers[header["page"]].append(header)

    # 각 페이지별로 Header 1, 2, 3 정보 수집
    page_headers = defaultdict(lambda: {"Header 1": "", "Header 2": "", "Header 3": ""})
    for page_number, page_headers_list in page_to_headers.items():
        # y좌표 기준으로 정렬
        sorted_headers = sorted(page_headers_list, key=lambda x: x["y"])

        # 각 레벨별로 가장 위에 있는 헤더를 찾음
        for header in sorted_headers:
            level = header["level"]
            if level == 1 and not page_headers[page_number]["Header 1"]:
                page_headers[page_number]["Header 1"] = header["text"]
            elif level == 2 and not page_headers[page_number]["Header 2"]:
                page_headers[page_number]["Header 2"] = header["text"]
            elif level == 3 and not page_headers[page_number]["Header 3"]:
                page_headers[page_number]["Header 3"] = header["text"]

    # 각 페이지의 헤더별로 이미지 생성
    for page_number, page_headers_list in page_to_headers.items():
        page_img_path = os.path.join(output_dir, f"page_{page_number}.jpg")
        img = Image.open(page_img_path)
        img_height = img.size[1]

        # level별로 그룹화
        level_to_headers = defaultdict(list)
        for h in page_headers_list:
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
                    "Header 1": page_headers[page_number]["Header 1"],
                    "Header 2": page_headers[page_number]["Header 2"],
                    "Header 3": page_headers[page_number]["Header 3"],
                    "page_number": page_number,
                }

    return header_dict


def add_source_text(slide, header_info, prs, pdf_name=""):
    """하단 중앙에 출처 텍스트 추가"""
    # page_number를 header_info에서 직접 가져오기 (page 또는 page_number 키 사용)
    page_number = header_info.get("page_number") or header_info.get("page", "?")

    # 출처 텍스트 박스
    width = Inches(6)  # 고정된 너비
    left = (prs.slide_width - width) / 2  # 중앙 정렬
    top = prs.slide_height - Inches(1.2)  # AI 고지 문구 위에 위치
    height = Inches(0.5)

    ref_box = slide.shapes.add_textbox(left, top, width, height)
    ref_tf = ref_box.text_frame
    ref_tf.word_wrap = True

    p_ref = ref_tf.paragraphs[0]
    p_ref.alignment = PP_ALIGN.CENTER

    run1 = p_ref.add_run()
    run1.text = "[출처: "
    run1.font.size = Pt(12)
    run1.font.color.rgb = RGBColor(120, 120, 120)

    run2 = p_ref.add_run()
    run2.text = f'"{pdf_name}", {page_number}page]'
    run2.font.size = Pt(12)
    run2.font.color.rgb = RGBColor(120, 120, 120)


def add_header_path(slide, header_info, prs):
    """제목 위에 계층적 헤더 경로 추가"""
    # level과 text를 사용해 계층적 헤더 구성
    level = header_info.get("level", 0)
    text = header_info.get("text", "")

    # 현재 헤더의 상위 헤더들을 찾기 위해 page와 y좌표 사용
    page = header_info.get("page", 0)
    y = header_info.get("y", 0)

    # 같은 페이지의 모든 헤더를 가져와서 y좌표 기준으로 정렬
    all_headers = []
    for key, info in st.session_state.get("header_dict", {}).items():
        if info.get("page") == page:
            all_headers.append(info)

    # y좌표 기준으로 정렬
    all_headers.sort(key=lambda x: x.get("y", 0))

    # 현재 헤더보다 위에 있는 헤더들 중 직계 상위 헤더만 찾기
    header_text = ""
    for h in all_headers:
        if h.get("y", 0) < y:
            h_level = h.get("level", 0)
            # 현재 헤더의 직계 상위 헤더만 추가
            if h_level == level - 1:
                if header_text:
                    header_text += " >> "
                header_text += h.get("text", "")

    # 현재 헤더 추가
    if header_text:
        header_text += " >> " + text
    else:
        header_text = text

    # 헤더 경로 텍스트 박스
    left = Inches(0.3)
    top = Inches(0.4)  # 제목보다 더 위에 위치
    width = prs.slide_width - Inches(0.6)
    height = Inches(0.5)

    path_box = slide.shapes.add_textbox(left, top, width, height)
    path_tf = path_box.text_frame
    path_tf.word_wrap = True

    p_path = path_tf.paragraphs[0]
    p_path.text = header_text
    p_path.font.size = Pt(12)  # 12pt로 변경
    p_path.font.bold = True
    p_path.font.color.rgb = RGBColor(0, 0, 255)  # 파란색
    p_path.alignment = PP_ALIGN.LEFT


def add_title_text(slide, header_info, prs):
    """상단에 큰 제목 추가"""
    # 계층적 헤더 경로에서 마지막 부분만 추출
    header_path = header_info.get("text", "")
    if " >> " in header_path:
        title_text = header_path.split(" >> ")[-1]
    else:
        title_text = header_path

    title_left = Inches(0.3)
    title_top = Inches(0.6)
    title_width = prs.slide_width - Inches(0.6)
    title_height = Inches(1)

    title_box = slide.shapes.add_textbox(
        title_left, title_top, title_width, title_height
    )
    title_tf = title_box.text_frame
    title_tf.word_wrap = True

    p_title = title_tf.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = RGBColor(80, 80, 80)
    p_title.alignment = PP_ALIGN.LEFT


def add_ai_notice_text(slide, prs):
    """하단 중앙에 생성형 AI 고지 문구 추가"""
    bottom_text = (
        "본 자료는 생성형 AI 기반으로 작성되었으며, 중요한 사실은 확인이 필요합니다."
    )

    box_width = Inches(6)
    bottom_left = (prs.slide_width - box_width) / 2
    bottom_top = prs.slide_height - Inches(0.7)
    bottom_height = Inches(0.5)

    ai_box = slide.shapes.add_textbox(bottom_left, bottom_top, box_width, bottom_height)
    ai_tf = ai_box.text_frame
    ai_tf.word_wrap = True

    p_ai = ai_tf.paragraphs[0]
    p_ai.text = bottom_text
    p_ai.font.size = Pt(12)
    p_ai.font.color.rgb = RGBColor(150, 150, 150)
    p_ai.alignment = PP_ALIGN.CENTER


def add_center_image(slide, header_info, prs):
    """중앙에 이미지 추가"""
    img_path = header_info.get("crop_image_path")
    if img_path and os.path.exists(img_path):
        with PILImage.open(img_path) as im:
            img_width, img_height = im.size
            slide_width = prs.slide_width
            slide_height = prs.slide_height
            slide_ratio = slide_width / slide_height
            img_ratio = img_width / img_height

            if img_ratio > slide_ratio:
                width = slide_width * 0.85
                height = width / img_ratio
            else:
                height = slide_height * 0.65
                width = height * img_ratio

        left = int((slide_width - width) / 2)
        top = int((slide_height - height) / 2)
        slide.shapes.add_picture(img_path, left, top, int(width), int(height))


def create_ppt_from_header_dict(
    header_list, output_pptx, pdf_name="교보마이플랜건강보험[2409](무배당)"
):
    prs = Presentation()
    blank_slide_layout = prs.slide_layouts[6]

    # 첫 번째 슬라이드에 표지 추가
    cover_slide = prs.slides.add_slide(blank_slide_layout)

    # 교보 로고 이미지 추가
    logo_path = os.path.join("temp_output", "kyobo_logo.jpg")
    if os.path.exists(logo_path):
        # 이미지 크기 계산 (가로가 슬라이드에 꽉 차도록)
        with PILImage.open(logo_path) as im:
            img_width, img_height = im.size
            slide_width = prs.slide_width
            slide_height = prs.slide_height

            # 가로가 슬라이드에 꽉 차도록 비율 계산
            width = slide_width
            height = (img_height / img_width) * width

            # 세로가 슬라이드보다 크면 세로 기준으로 다시 계산
            if height > slide_height:
                height = slide_height
                width = (img_width / img_height) * height

            # 이미지를 중앙에 배치
            left = int((slide_width - width) / 2)
            top = int((slide_height - height) / 2)

            cover_slide.shapes.add_picture(
                logo_path, left, top, int(width), int(height)
            )

    # PDF 이름과 날짜 텍스트 추가
    from datetime import datetime

    today = datetime.now().strftime("%Y년 %m월 %d일")

    # PDF 이름 텍스트 박스
    pdf_text_left = Inches(0.3)
    pdf_text_top = Inches(5)  # 로고 아래에 위치
    pdf_text_width = prs.slide_width - Inches(0.6)
    pdf_text_height = Inches(0.5)

    pdf_text_box = cover_slide.shapes.add_textbox(
        pdf_text_left, pdf_text_top, pdf_text_width, pdf_text_height
    )
    pdf_text_tf = pdf_text_box.text_frame
    pdf_text_tf.word_wrap = True

    p_pdf = pdf_text_tf.paragraphs[0]
    p_pdf.text = f'현재 PDF이름: "{pdf_name}"'
    p_pdf.font.size = Pt(14)
    p_pdf.font.color.rgb = RGBColor(150, 150, 150)
    p_pdf.alignment = PP_ALIGN.CENTER

    # 날짜 텍스트 박스
    date_text_left = Inches(0.3)
    date_text_top = Inches(5.5)  # PDF 이름 아래에 위치
    date_text_width = prs.slide_width - Inches(0.6)
    date_text_height = Inches(0.5)

    date_text_box = cover_slide.shapes.add_textbox(
        date_text_left, date_text_top, date_text_width, date_text_height
    )
    date_text_tf = date_text_box.text_frame
    date_text_tf.word_wrap = True

    p_date = date_text_tf.paragraphs[0]
    p_date.text = today
    p_date.font.size = Pt(12)
    p_date.font.color.rgb = RGBColor(150, 150, 150)
    p_date.alignment = PP_ALIGN.CENTER

    # 나머지 슬라이드 생성
    for header_info in header_list:
        # Header 1 레벨의 헤더는 건너뛰기
        if header_info.get("level") == 1:
            continue

        slide = prs.slides.add_slide(blank_slide_layout)

        # 1. 중앙에 이미지 추가
        add_center_image(slide, header_info, prs)

        # 2. 출처 텍스트 추가
        add_source_text(slide, header_info, prs, pdf_name)

        # 3. 헤더 경로 추가
        add_header_path(slide, header_info, prs)

        # 4. 제목 텍스트 추가
        add_title_text(slide, header_info, prs)

        # 5. AI 고지 문구 추가
        add_ai_notice_text(slide, prs)

    prs.save(output_pptx)


def get_deepest_header_and_level(metadata):
    for depth in reversed(range(1, 5)):  # Header4까지 확장 가능
        key = f"Header {depth}"
        if key in metadata and metadata[key]:
            return metadata[key], depth
    return None, None


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
            header_dict = st.session_state.get("header_dict", {})
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
                    # === 이미지 미리보기 추가 ===
                    header_text, level = get_deepest_header_and_level(doc.metadata)
                    matched_header = None
                    for h in header_dict.values():
                        if (
                            h["page"] == page
                            and h["level"] == level
                            and h["text"] == header_text
                        ):
                            matched_header = h
                            break
                    if (
                        matched_header
                        and matched_header.get("crop_image_path")
                        and os.path.exists(matched_header["crop_image_path"])
                    ):
                        st.image(
                            matched_header["crop_image_path"],
                            caption=f"이미지 ({header_text})",
                            use_column_width=True,
                        )
            else:
                doc = results
                page = doc.metadata.get("page_number", "알 수 없음")
                st.markdown(f"**결과 (페이지: {page})**")
                st.code(doc.page_content)
                st.json(doc.metadata)
                header_text, level = get_deepest_header_and_level(doc.metadata)
                matched_header = None
                for h in header_dict.values():
                    if (
                        h["page"] == page
                        and h["level"] == level
                        and h["text"] == header_text
                    ):
                        matched_header = h
                        break
                if (
                    matched_header
                    and matched_header.get("crop_image_path")
                    and os.path.exists(matched_header["crop_image_path"])
                ):
                    st.image(
                        matched_header["crop_image_path"],
                        caption=f"Crop 이미지 (헤더: {header_text})",
                        use_column_width=True,
                    )
            # === 검색 결과 PPT 다운로드 버튼 ===
            selected_headers = []

            if isinstance(results, list):
                for doc in results:
                    page = doc.metadata.get("page_number")
                    header_text, level = get_deepest_header_and_level(doc.metadata)
                    for h in header_dict.values():
                        if (
                            h["page"] == page
                            and h["level"] == level
                            and h["text"] == header_text
                        ):
                            selected_headers.append(h)
            else:
                doc = results
                page = doc.metadata.get("page_number")
                header_text, level = get_deepest_header_and_level(doc.metadata)
                for h in header_dict.values():
                    if (
                        h["page"] == page
                        and h["level"] == level
                        and h["text"] == header_text
                    ):
                        selected_headers.append(h)
            st.session_state["selected_headers"] = selected_headers
            # 검색 시점에 PPT를 미리 생성하고 경로를 세션에 저장
            pptx_path = os.path.join("temp_output", "search_results.pptx")

            if selected_headers:
                create_ppt_from_header_dict(selected_headers, pptx_path)
                st.session_state["search_pptx_path"] = pptx_path
            else:
                st.session_state["search_pptx_path"] = None

            # 다운로드 버튼은 세션 값만 사용
            if st.session_state.get("search_pptx_path") and os.path.exists(
                st.session_state["search_pptx_path"]
            ):
                with open(st.session_state["search_pptx_path"], "rb") as f:
                    st.download_button(
                        label="검색 결과 PPT 다운로드",
                        data=f,
                        file_name="search_results.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
            else:
                st.info("검색 결과에 해당하는 헤더 이미지가 없습니다.")
        else:
            st.warning("먼저 벡터스토어에 임베딩을 저장하고, 검색어를 입력하세요.")

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
        pptx_path = os.path.join(output_dir, "exported_slides.pptx")
        progress_bar = st.progress(0, text="대기 중...")
        status_text = st.empty()
        if "parsing_in_progress" not in st.session_state:
            st.session_state["parsing_in_progress"] = False
        if not st.session_state["parsing_in_progress"]:
            if ("pptx_path" in st.session_state and st.session_state["pptx_path"]) or (
                "md_save_path" in st.session_state and st.session_state["md_save_path"]
            ):
                st.success("파싱이 완료되었습니다. 아래에서 파일을 다운로드하세요.")
                pptx_path = st.session_state.get("pptx_path")
                md_save_path = st.session_state.get("md_save_path")
                if pptx_path:
                    try:
                        with open(pptx_path, "rb") as f:
                            st.download_button(
                                label="파싱된 PPT 파일 전체 다운로드",
                                data=f,
                                file_name="exported_slides.pptx",
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            )
                    except Exception as e:
                        st.error(f"PPT 파일을 찾을 수 없습니다: {e}")
                if md_save_path:
                    try:
                        with open(md_save_path, "rb") as f_md:
                            st.download_button(
                                label="파싱된 Markdown 파일 전체 다운로드",
                                data=f_md,
                                file_name=os.path.basename(md_save_path),
                                mime="text/markdown",
                            )
                    except Exception as e:
                        st.error(f"Markdown 파일을 찾을 수 없습니다: {e}")
            else:
                if st.button("파싱 시작", type="primary"):
                    st.session_state["parsing_in_progress"] = True
                    st.experimental_rerun()
        if st.session_state["parsing_in_progress"]:
            status_text.info("마크다운 파일로 변환중...")
            for fake_percent in range(5, 41, 5):
                progress_bar.progress(fake_percent, text=f"마크다운 파일로 변환중...")
                time.sleep(0.5)
            temp_pdf_path = os.path.join(output_dir, f"temp_{uploaded_file.name}")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
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
            # 페이지 수 예측 (없으면 7로 가정)
            total_pages = 7
            result = parser_obj.parse(temp_pdf_path)
            md_docs = result.get_markdown_documents(split_by_page=True)
            st.session_state["md_docs"] = md_docs
            st.session_state["llama_parse_result"] = result
            progress_bar.progress(40, text="마크다운 문서 생성 완료!")
            status_text.success("마크다운 문서 생성 완료!")
            for idx, page in enumerate(result.pages):
                percent = 40 + int(10 * (idx + 1) / total_pages)
                progress_bar.progress(
                    percent, text=f"마크다운 후처리중... ({idx+1}/{total_pages})"
                )
            for page in result.pages:
                images = getattr(page, "images", [])
                for img in images:
                    if getattr(img, "type", None) == "full_page_screenshot":
                        img_name = getattr(img, "name", None)
                        if img_name:
                            result.save_image(img_name, output_dir)
            if md_docs:
                with open(md_save_path, "w", encoding="utf-8") as f:
                    for doc in md_docs:
                        f.write(doc.text)
                st.session_state["md_save_path"] = md_save_path
                st.info(f"마크다운 파일이 저장되었습니다: {md_save_path}")
            progress_bar.progress(50, text="임베딩 및 벡터스토어 저장중...")
            status_text.info("임베딩 및 벡터스토어 저장중...")
            md_save_path = st.session_state["md_save_path"]
            with open(md_save_path, "r", encoding="utf-8") as f:
                md_text = f.read()
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
            progress_bar.progress(70, text="임베딩 계산중...")
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(all_chunks, embeddings)
            st.session_state["faiss_db"] = db
            progress_bar.progress(80, text="OCR 로 좌표 계산중...")
            status_text.info("OCR 로 좌표 계산중...")
            result = st.session_state.get("llama_parse_result")
            api_url = "https://8vb79ndbzb.apigw.ntruss.com/custom/v1/41373/3a26df9469f0c22bb024a70d5cc5e11a9fb28f6ea21993ba8eee769f4dda9216/general"
            secret_key = os.getenv("CLOVA_OCR_SECRET_KEY")
            headers = []
            ocr_results_dict = {}
            for page_idx, page in enumerate(result.pages):
                page_number = getattr(page, "page", page_idx + 1)
                items = getattr(page, "items", [])
                page_headers = extract_headers_from_llamaparse_items(items, page_number)
                page_img_path = os.path.join(output_dir, f"page_{page_number}.jpg")
                ocr_results_dict[page_number] = clova_ocr_multipart(
                    page_img_path, api_url, secret_key
                )
                for header in page_headers:
                    ocr_result = ocr_results_dict.get(page_number)
                    y, matched_text, bbox_list = extract_header_y_from_ocr_response(
                        ocr_result, header["text"], image_path=page_img_path
                    )
                    header["y"] = y
                headers.extend(page_headers)
                percent = 80 + int(15 * (page_idx + 1) / total_pages)
                progress_bar.progress(
                    percent, text=f"OCR 및 crop 진행중... ({page_idx+1}/{total_pages})"
                )
            header_dict = build_header_dictionary(
                headers, ocr_results_dict, output_dir=output_dir
            )
            st.session_state["header_dict"] = header_dict
            # 파싱이 끝난 직후 최신 header_dict로 PPT 생성
            create_ppt_from_header_dict(list(header_dict.values()), pptx_path)
            # header_dict를 표로 시각화
            df = pd.DataFrame(list(header_dict.values()))
            st.subheader("파싱된 헤더/이미지 정보 표")
            st.dataframe(df)
            progress_bar.progress(100, text="생성 완료!")
            status_text.success("모든 파일이 파싱된 PPT 생성 완료!")
            st.session_state["pptx_path"] = pptx_path
            st.session_state["parsing_in_progress"] = False
            # 파싱 직후 다운로드 버튼 바로 노출
            st.success("파싱이 완료되었습니다. 아래에서 파일을 다운로드하세요.")
            pptx_path = st.session_state.get("pptx_path")
            md_save_path = st.session_state.get("md_save_path")
            if pptx_path:
                try:
                    with open(pptx_path, "rb") as f:
                        st.download_button(
                            label="파싱된 PPT 파일 다운로드",
                            data=f,
                            file_name="exported_slides.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        )
                except Exception as e:
                    st.error(f"PPT 파일을 찾을 수 없습니다: {e}")
            if md_save_path:
                try:
                    with open(md_save_path, "rb") as f_md:
                        st.download_button(
                            label="파싱된 Markdown 파일 다운로드",
                            data=f_md,
                            file_name=os.path.basename(md_save_path),
                            mime="text/markdown",
                        )
                except Exception as e:
                    st.error(f"Markdown 파일을 찾을 수 없습니다: {e}")
