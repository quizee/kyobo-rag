from pathlib import Path
from pdf_to_md_converter import PDFToMarkdownConverter


def test_pdf_conversion():
    # 테스트할 PDF 파일 경로
    pdf_path = Path(
        "/Users/jeeyoonlee/Desktop/kyobo-project/data/상품설명/1.교보마이플랜건강보험 [2409](무배당).pdf"
    )
    output_dir = Path("test_output")

    print(f"테스트 시작: {pdf_path}")
    print(f"출력 디렉토리: {output_dir}")

    try:
        # PDF 변환기 초기화
        converter = PDFToMarkdownConverter()

        # PDF 변환 실행
        converter.convert_file(pdf_path, output_dir)

        print("\n테스트 완료!")

        # 결과 파일 확인
        md_file = output_dir / f"{pdf_path.stem}.md"
        json_file = output_dir / f"{pdf_path.stem}_structured.json"

        if md_file.exists():
            print(f"Markdown 파일 생성됨: {md_file}")
            # 파일 내용 미리보기
            print("\nMarkdown 내용 미리보기:")
            with open(md_file, "r", encoding="utf-8") as f:
                print(f.read()[:500] + "...\n")

        if json_file.exists():
            print(f"JSON 파일 생성됨: {json_file}")

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")


if __name__ == "__main__":
    test_pdf_conversion()
