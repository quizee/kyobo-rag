import streamlit as st
import os
from llama_cloud_services import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai


st.set_page_config(page_title="PDF Chat App", layout="wide")

# 2단 컬럼 레이아웃 (왼쪽: 채팅, 오른쪽: 파일 업로드)
col_chat, col_upload = st.columns([2, 1])

# FAISS DB를 세션에 저장
if "faiss_db" not in st.session_state:
    st.session_state["faiss_db"] = None

with col_chat:
    st.header("💬 교육 자료 검색하기")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    # 채팅 내역 표시
    for msg in st.session_state["chat_history"]:
        st.markdown(f"**{msg['role']}**: {msg['content']}")
    # 사용자 입력
    user_input = st.text_input("메시지를 입력하세요", key="user_input")
    if st.button("전송"):
        if user_input:
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_input}
            )
            st.session_state["user_input"] = ""  # 입력창 초기화
    # 검색 버튼 및 결과 출력
    if st.button("검색"):
        if st.session_state["faiss_db"] is not None and user_input:
            retriever = st.session_state["faiss_db"].as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )
            # invoke가 (doc, score) 튜플을 반환하도록 가정
            results = retriever.invoke(user_input, return_score=True)

            # 헤더 가중치 적용 함수
            def custom_score(doc, query, base_score):
                header_text = doc.metadata.get("header_text", "")
                if query in header_text:
                    return base_score + 0.2  # 헤더에 포함되면 0.2점 추가
                return base_score

            # 헤더 가중치 적용 및 정렬
            scored_results = [
                (doc, custom_score(doc, user_input, score)) for doc, score in results
            ]
            scored_results.sort(key=lambda x: x[1], reverse=True)
            st.subheader("검색 결과 (헤더 가중치 적용)")
            for i, (doc, score) in enumerate(scored_results):
                st.markdown(f"**결과 {i+1} (점수: {score:.3f})**")
                st.code(doc.page_content)
                st.json(doc.metadata)
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
        # 이미 마크다운 파일이 있으면 llama parser를 실행하지 않음
        if os.path.exists(md_save_path):
            st.info(f"이미 마크다운 파일이 존재합니다: {md_save_path}")
            st.session_state["md_save_path"] = md_save_path
        elif st.button("파싱 시작"):
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
                max_pages=6,
            )
            st.info("llama_parse 실행 중...")
            result = parser_obj.parse(temp_pdf_path)
            md_docs = result.get_markdown_documents()
            st.session_state["md_docs"] = md_docs  # 세션에 저장
            st.success("마크다운 문서 생성 완료!")
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
        # 1. Split
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        md_header_splits = markdown_splitter.split_text(md_text)
        # 각 chunk에 header_text 필드 추가
        for chunk in md_header_splits:
            headers = [v for k, v in chunk.metadata.items() if k.startswith("Header")]
            chunk.metadata["header_text"] = " ".join(headers)
        # 2. 임베딩
        embeddings = OpenAIEmbeddings()
        # 3. FAISS 벡터 DB 저장
        db = FAISS.from_documents(md_header_splits, embeddings)
        st.session_state["faiss_db"] = db  # 세션에 저장
        st.success(
            f"{len(md_header_splits)}개 chunk가 임베딩되어 FAISS에 저장되었습니다. 이제 검색이 가능합니다."
        )
