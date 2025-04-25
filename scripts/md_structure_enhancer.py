import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from config import LLAMACLOUD_API_KEY, HEADER_X_THRESHOLD


class MarkdownStructureEnhancer:
    def __init__(self, json_path: str, md_path: str):
        self.json_path = json_path
        self.md_path = md_path
        self.output_path = md_path.replace(".md", "_enhanced.md")
        self.api_key = LLAMACLOUD_API_KEY
        self.api_url = "https://api.llamacloud.ai/v1/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def load_json_data(self) -> Dict:
        """JSON 파일을 로드합니다."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_markdown_with_llama(self, page_data: Dict) -> str:
        """LlamaCloud API를 사용하여 페이지 데이터를 Markdown으로 변환합니다."""
        prompt = self._create_prompt(page_data)

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "llama-2-70b-chat",
                    "prompt": prompt,
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error calling LlamaCloud API: {e}")
            return ""

    def _create_prompt(self, page_data: Dict) -> str:
        """API 요청을 위한 프롬프트를 생성합니다."""
        return f"""Please convert this JSON page data into well-structured Markdown format.
Follow these rules:
1. Maintain proper header hierarchy (h1 > h2 > h3)
2. Keep the original text formatting and structure
3. Preserve tables and lists
4. Include all text content in the correct order

Page data:
{json.dumps(page_data, ensure_ascii=False, indent=2)}

Generate markdown content:"""

    def process_document(self) -> None:
        """전체 문서를 처리하고 Markdown으로 변환합니다."""
        data = self.load_json_data()
        all_markdown = []

        for page in data["pages"]:
            markdown_content = self.generate_markdown_with_llama(page)
            all_markdown.append(markdown_content)
            all_markdown.append("---")  # 페이지 구분자 추가

        # 최종 Markdown 저장
        final_content = "\n\n".join(all_markdown)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(final_content)


def main():
    # 파일 경로 설정
    from config import DEFAULT_JSON_PATH, DEFAULT_MD_PATH

    # Markdown 구조 개선
    enhancer = MarkdownStructureEnhancer(DEFAULT_JSON_PATH, DEFAULT_MD_PATH)
    enhancer.process_document()
    print(f"Enhanced Markdown saved to: {enhancer.output_path}")


if __name__ == "__main__":
    main()
