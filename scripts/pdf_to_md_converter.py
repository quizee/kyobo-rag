import os
import sys
import json
import asyncio
import httpx
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import aiohttp

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class PDFToMarkdownConverter:
    """PDF 파일을 Markdown으로 변환하는 클래스"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API 키가 필요합니다. .env 파일에 LLAMA_CLOUD_API_KEY를 설정하거나 초기화 시 제공해주세요."
            )

        self.base_url = "https://api.cloud.llamaindex.ai/api/parsing"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def _upload_file(self, file_path: str) -> Dict[str, Any]:
        """PDF 파일을 업로드하고 job ID를 반환합니다."""
        url = f"{self.base_url}/upload"

        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path),
                    content_type="application/pdf",
                )

                async with session.post(
                    url, data=form, headers=self.headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"API 오류 (상태 코드: {response.status}): {error_text}")
                        raise Exception(f"파일 업로드 실패: {error_text}")

                    result = await response.json()
                    print("업로드 응답:", result)  # 디버깅을 위한 출력
                    return result

    async def _get_result(self, job_id: str) -> Dict[str, Any]:
        """작업 결과를 가져옵니다."""
        url = f"{self.base_url}/job/{job_id}"

        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"API 오류 (상태 코드: {response.status}): {error_text}")
                        raise Exception(f"결과 조회 실패: {error_text}")

                    result = await response.json()
                    print("상태 확인 응답:", result)  # 디버깅을 위한 출력

                    status = result.get("status")
                    if status == "completed":
                        return result
                    elif status == "failed":
                        raise Exception(
                            "작업 실패: " + result.get("error", "알 수 없는 오류")
                        )

                    await asyncio.sleep(2)  # 2초 대기 후 다시 확인

    def convert_file(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
    ) -> None:
        """단일 PDF 파일을 Markdown으로 변환합니다.

        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 디렉토리 (기본값: PDF와 같은 디렉토리)
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = pdf_path.parent

        # PDF 파싱
        print(f"PDF 파일 파싱 중: {pdf_path}")

        try:
            # PDF 파싱 실행
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            upload_result = loop.run_until_complete(self._upload_file(str(pdf_path)))
            job_id = upload_result["id"]
            print(f"작업 ID: {job_id}")

            result = loop.run_until_complete(self._get_result(job_id))

            # 결과물 저장
            outputs = []

            # 1. Markdown 문서
            output_path = output_dir / f"{pdf_path.stem}.md"

            print("Markdown 파일 생성 중...")
            with open(output_path, "w", encoding="utf-8") as f:
                # 문서 메타데이터 추가
                f.write("---\n")
                f.write("title: " + pdf_path.stem + "\n")
                f.write(
                    "date: "
                    + str(result.get("metadata", {}).get("creation_date", ""))
                    + "\n"
                )
                f.write(
                    "pages: "
                    + str(result.get("metadata", {}).get("page_count", 0))
                    + "\n"
                )
                f.write("---\n\n")

                # Markdown 내용 작성
                f.write(result["markdown"])

            outputs.append(("Markdown", output_path))

            # 2. JSON 데이터
            json_output_path = output_dir / f"{pdf_path.stem}_structured.json"
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            outputs.append(("JSON", json_output_path))

            # 결과 출력
            print(f"\n변환 완료!")
            print(f"파일이 다음 위치에 저장되었습니다:")
            for output_type, path in outputs:
                print(f"- {output_type}: {path}")

        except Exception as e:
            print(f"PDF 변환 중 오류 발생: {e}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDF 파일을 Markdown으로 변환")
    parser.add_argument("input", help="입력 PDF 파일 또는 디렉토리 경로")
    parser.add_argument("--output", "-o", help="출력 디렉토리 경로 (선택사항)")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="병렬 처리를 위한 worker 수",
    )
    parser.add_argument(
        "--parsing-instruction",
        help="사용자 정의 파싱 지시사항",
    )

    args = parser.parse_args()

    converter = PDFToMarkdownConverter()

    input_path = Path(args.input)
    if input_path.is_file():
        converter.convert_file(input_path, args.output)
    else:
        print("디렉토리 변환은 아직 지원하지 않습니다.")


if __name__ == "__main__":
    main()
