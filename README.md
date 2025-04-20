# Kyobo Insurance RAG Project

보험 상품 설명서 PDF를 분석하여 RAG(Retrieval Augmented Generation) 시스템을 구축하는 프로젝트입니다.

## 프로젝트 구조

```
kyobo-project/
├── data/
│   ├── 상품설명/        # 원본 PDF 파일
│   └── extracted/       # 추출된 데이터
│       ├── images/     # 섹션별 이미지
│       └── text/       # 섹션별 메타데이터 (JSON)
└── scripts/
    └── pdf_processor.py  # PDF 처리 및 섹션 추출 스크립트
```

## 주요 기능

### PDF 처리 및 섹션 추출 (`pdf_processor.py`)

1. **첫 페이지 특별 처리**
   - PDF 파일의 첫 페이지는 단일 섹션으로 처리
   - 파일명을 기반으로 메타데이터 생성

2. **시각적 분석**
   - GPT-4 Vision을 사용하여 페이지의 시각적 구조 분석
   - 섹션 타입 분류: text, list, table, box, image
   - 각 섹션의 위치 정보(bbox) 추출

3. **텍스트 추출 및 제목 생성**
   - 섹션별 텍스트를 원본 그대로 추출 (description)
   - 추출된 텍스트를 기반으로 명확한 제목 생성 (title)

4. **이미지 처리**
   - PyMuPDF(fitz)를 사용하여 섹션별 이미지 추출
   - 좌표 기반 정확한 이미지 크롭

5. **메타데이터 저장**
   - 각 섹션별 JSON 형식 메타데이터 생성
   - 이미지와 메타데이터를 구조화된 형식으로 저장

## 설치 및 실행

1. 환경 설정
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. OpenAI API 키 설정
   ```bash
   # .env 파일 생성
   OPENAI_API_KEY=your-api-key-here
   ```

3. 실행
   ```bash
   python scripts/pdf_processor.py
   ```

## 현재 진행 상황

- [x] PDF 처리 기본 구조 구현
- [x] GPT-4 Vision 기반 섹션 분석
- [x] 이미지 추출 및 저장 기능
- [x] 메타데이터 생성 및 저장
- [ ] Vector Store 구축
- [ ] RAG 시스템 구현
- [ ] 검색 및 질의응답 기능 개발

## 다음 단계

1. Vector Store 구축
   - 추출된 텍스트 데이터를 임베딩하여 Vector Store 생성
   - 효율적인 검색을 위한 인덱싱

2. RAG 시스템 구현
   - 검색 기능 구현
   - 컨텍스트 기반 응답 생성
   - 정확도 향상을 위한 프롬프트 최적화 