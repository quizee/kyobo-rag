import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv


class UpstageParser:
    """Upstage API를 사용하여 문서를 파싱하는 클래스"""

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API 키가 필요합니다. .env 파일에 UPSTAGE_API_KEY를 설정하거나 초기화 시 제공해주세요."
            )

        self.url = "https://api.upstage.ai/v1/document-digitization"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def parse(self, file_path: str) -> Dict[str, Any]:
        """PDF 파일을 파싱하여 구조화된 데이터를 반환합니다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {
                "model": "document-parse",
                "chart_recognition": True,
                "align_orientation": True,
                "ocr": "force",
                "output_formats": "['markdown']",
                "coordinates": True,
            }

            response = requests.post(
                self.url, headers=self.headers, files=files, data=data
            )

            if response.status_code != 200:
                raise Exception(
                    f"API 호출 실패: {response.status_code} - {response.text}"
                )

            return response.json()

    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """파싱 결과를 JSON 파일로 저장합니다."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def _determine_heading_level(self, element: Dict[str, Any]) -> int:
        """요소의 카테고리와 내용을 기반으로 헤딩 레벨을 결정합니다."""
        category = element.get("category", "")
        content = element.get("content", {})
        html = content.get("html", "")

        # HTML 태그에서 헤딩 레벨 추출 시도
        if html:
            if "<h1" in html:
                return 1
            elif "<h2" in html:
                return 2
            elif "<h3" in html:
                return 3

        # 카테고리 기반 기본 레벨
        if category == "header":
            return 1
        elif category == "heading1":
            return 2
        elif category == "heading2":
            return 3
        elif category == "heading3":
            return 4

        # 기본값
        return 1

    def get_markdown_documents(
        self, result: Dict[str, Any], split_by_page: bool = True
    ) -> List[str]:
        """파싱 결과를 마크다운 형식으로 변환합니다."""
        if not result.get("elements"):
            return []

        # 페이지별로 요소들을 그룹화
        pages = {}
        for element in result["elements"]:
            page_num = element.get("page", 1)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(element)

        markdown_docs = []

        # 각 페이지의 요소들을 마크다운으로 변환
        for page_num in sorted(pages.keys()):
            page_md = []
            for element in pages[page_num]:
                category = element.get("category", "")
                content = element.get("content", {})

                # 마크다운 형식이 있으면 우선 사용
                if "markdown" in content and content["markdown"]:
                    page_md.append(content["markdown"])
                    continue

                # HTML이나 텍스트 기반 변환
                if category in ["header", "heading1", "heading2", "heading3"]:
                    level = self._determine_heading_level(element)
                    text = content.get("text", "")
                    page_md.append(f"{'#' * level} {text}")
                elif category == "paragraph":
                    page_md.append(content.get("text", ""))
                elif category == "table":
                    # 테이블 HTML을 마크다운으로 변환
                    if "html" in content:
                        # HTML 테이블을 마크다운으로 변환하는 로직 추가 필요
                        page_md.append(content.get("text", ""))
                    else:
                        page_md.append(content.get("text", ""))

            if split_by_page:
                markdown_docs.append("\n".join(page_md))
            else:
                markdown_docs.extend(page_md)

        return markdown_docs


def test_parse():
    """파서 테스트 함수"""
    pdf_path = Path("data/상품설명/1.교보마이플랜건강보험 [2409](무배당).pdf")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    parser = UpstageParser()
    print("PDF 파싱 시작...")
    result = parser.parse(str(pdf_path))
    print("파싱 완료!")

    # 결과 저장
    raw_json_path = output_dir / "upstage_parse_raw.json"
    parser.save_result(result, str(raw_json_path))
    print(f"파싱 결과 저장: {raw_json_path}")

    # 마크다운 변환 테스트
    md_docs = parser.get_markdown_documents(result, split_by_page=True)
    md_path = output_dir / "upstage_parse.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(md_docs))
    print(f"마크다운 파일 저장: {md_path}")


if __name__ == "__main__":
    test_parse()
