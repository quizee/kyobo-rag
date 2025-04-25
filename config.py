"""Configuration settings for the application."""

import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# LlamaCloud API 키
LLAMACLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not LLAMACLOUD_API_KEY:
    raise ValueError(
        "LLAMA_CLOUD_API_KEY가 설정되지 않았습니다. "
        ".env 파일에 LLAMA_CLOUD_API_KEY를 설정해주세요."
    )

# File paths
DEFAULT_JSON_PATH = (
    "data/상품설명/1.교보마이플랜건강보험 [2409](무배당) (1).pdf (2).json"
)
DEFAULT_MD_PATH = "data/상품설명/1.교보마이플랜건강보험 [2409](무배당) (1).pdf.md"

# Markdown settings
HEADER_X_THRESHOLD = 20  # x 좌표 차이 임계값
