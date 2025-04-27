import streamlit as st
import os
from dotenv import load_dotenv
import nest_asyncio
import re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import time

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


def create_footer_text_streamlit(slide, header_info, prs):
    left = Inches(0.5)
    top = prs.slide_height - Inches(0.8)
    width = prs.slide_width - Inches(1)
    height = Inches(0.5)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    header_text = ""
    if header_info.get("Header 1"):
        header_text += header_info["Header 1"]
    if header_info.get("Header 2"):
        if header_text:
            header_text += " - "
        header_text += header_info["Header 2"]
    if header_info.get("Header 3"):
        if header_text:
            header_text += " - "
        header_text += header_info["Header 3"]
    if not header_text:
        header_text = header_info.get("text", "")
    p1 = tf.paragraphs[0]
    p1.text = header_text
    p1.font.size = Pt(8)
    p1.font.color.rgb = RGBColor(128, 128, 128)
    p2 = tf.add_paragraph()
    p2.text = (
        "본 자료는 생성형 AI 기반으로 작성되었으며, 중요한 사실은 확인이 필요합니다"
    )
    p2.font.size = Pt(8)
    p2.font.color.rgb = RGBColor(128, 128, 128)


def create_ppt_from_header_dict(headers, output_pptx):
    prs = Presentation()
    blank_slide_layout = prs.slide_layouts[6]
    for header in headers:
        slide = prs.slides.add_slide(blank_slide_layout)
        img_path = header.get("crop_image_path")
        if img_path and os.path.exists(img_path):
            from PIL import Image as PILImage

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
                    height = slide_height * 0.85
                    width = height * img_ratio
            left = int((slide_width - width) / 2)
            top = int((slide_height - height) / 2)
            slide.shapes.add_picture(img_path, left, top, int(width), int(height))
        create_footer_text_streamlit(slide, header, prs)
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
