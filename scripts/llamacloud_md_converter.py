import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from config import LLAMACLOUD_API_KEY


class LlamaCloudMarkdownConverter:
    """LlamaCloud API를 사용하여 JSON 데이터를 Markdown으로 변환하는 클래스"""

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.output_path = (
            self.json_path.parent / f"{self.json_path.stem}_llamacloud.md"
        )
        self.api_key = LLAMACLOUD_API_KEY
        self.api_url = "https://api.llamacloud.ai/v1/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def load_json_data(self) -> Dict:
        """JSON 파일을 로드합니다."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 파일 로드 중 오류 발생: {e}")
            return {}

    def create_page_prompt(self, page_data: Dict) -> str:
        """페이지 데이터에 대한 프롬프트를 생성합니다."""
        return f"""Convert this JSON page data into well-structured Korean Markdown.
Follow these rules strictly:
1. Headers must follow proper hierarchy:
   - Main product name and major sections should be h1 (#)
   - Subsections should be h2 (##)
   - Further details should be h3 (###)
2. Maintain original Korean text formatting
3. Preserve all tables with proper markdown table syntax
4. Keep lists and bullet points
5. Special handling for "상품특징" section:
   - "상품특징" should be h1 (#)
   - Numbered features (1., 2., etc) should be h2 (##)
   - Sub-features should be h3 (###)

Page data:
{json.dumps(page_data, ensure_ascii=False, indent=2)}

Generate the markdown content in Korean:"""

    def convert_page_to_markdown(self, page_data: Dict) -> str:
        """한 페이지의 데이터를 Markdown으로 변환합니다."""
        prompt = self.create_page_prompt(page_data)

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "llama-2-70b-chat",
                    "prompt": prompt,
                    "temperature": 0.2,  # 일관된 출력을 위해 낮은 temperature 사용
                    "max_tokens": 2000,
                },
                timeout=30,  # 30초 타임아웃 설정
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            print(f"API 호출 중 오류 발생: {e}")
            return ""
        except Exception as e:
            print(f"예상치 못한 오류 발생: {e}")
            return ""

    def process_document(self) -> None:
        """전체 문서를 처리하고 Markdown으로 변환합니다."""
        data = self.load_json_data()
        if not data or "pages" not in data:
            print("유효한 JSON 데이터를 찾을 수 없습니다.")
            return

        print(f"총 {len(data['pages'])} 페이지 처리 시작...")
        all_markdown = []

        for i, page in enumerate(data["pages"], 1):
            print(f"페이지 {i} 처리 중...")
            markdown_content = self.convert_page_to_markdown(page)
            if markdown_content:
                all_markdown.append(markdown_content)
                all_markdown.append("\n---\n")  # 페이지 구분자
            else:
                print(f"페이지 {i} 변환 실패")

        # 최종 Markdown 저장
        final_content = "\n".join(all_markdown)
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(final_content)
            print(f"\nMarkdown 파일이 성공적으로 저장되었습니다: {self.output_path}")
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")


def main():
    # 파일 경로 설정
    from config import DEFAULT_JSON_PATH

    # Markdown 변환 실행
    converter = LlamaCloudMarkdownConverter(DEFAULT_JSON_PATH)
    converter.process_document()


if __name__ == "__main__":
    main()
