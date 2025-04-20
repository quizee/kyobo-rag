import os
from pathlib import Path
import json
from dotenv import load_dotenv
from openai import OpenAI
import logging
import tempfile
import shutil
import base64
from PIL import Image
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreBuilder:
    def __init__(self, input_dir: str = "data/extracted", name: str = "보험상품_검색"):
        self.input_dir = Path(input_dir)
        self.name = name
        self.vector_store_file = Path("data/vector_store_id.txt")

        # OpenAI 클라이언트 초기화
        load_dotenv()
        self.client = OpenAI()
        self.vector_store_id = self._load_vector_store_id()
        self.data_dir = self.input_dir
        self.text_dir = self.data_dir / "text"
        self.images_dir = self.data_dir / "images"

    def _load_vector_store_id(self) -> str:
        """저장된 벡터 스토어 ID를 로드합니다."""
        try:
            if self.vector_store_file.exists():
                with open(self.vector_store_file, "r") as f:
                    vector_store_id = f.read().strip()
                    if vector_store_id:
                        logger.info(f"기존 벡터 스토어 ID 로드: {vector_store_id}")
                        return vector_store_id
        except Exception as e:
            logger.warning(f"벡터 스토어 ID 로드 중 오류 발생: {e}")
        return None

    def _save_vector_store_id(self, vector_store_id: str):
        """벡터 스토어 ID를 파일에 저장합니다."""
        try:
            self.vector_store_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.vector_store_file, "w") as f:
                f.write(vector_store_id)
            logger.info(f"벡터 스토어 ID 저장 완료: {vector_store_id}")
        except Exception as e:
            logger.warning(f"벡터 스토어 ID 저장 중 오류 발생: {e}")

    def json_to_text_file(self, json_path: Path, temp_dir: Path) -> Path:
        """JSON 파일을 벡터 스토어용 텍스트 파일로 변환합니다."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 제목과 본문을 결합하되, 제목에 더 높은 가중치 부여
            # 제목을 3번 반복하여 검색 시 더 높은 우선순위를 가지도록 함
            content = f"""제목: {data['title']}
제목: {data['title']}
제목: {data['title']}
페이지: {json_path.stem.split('_p')[1].split('_')[0]}
섹션 번호: {json_path.stem.split('_section_')[1]}

--- 본문 ---
{data['text']}"""

            # 임시 파일 생성
            temp_file = temp_dir / f"{json_path.stem}.txt"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)

            return temp_file
        except Exception as e:
            logger.error(f"JSON을 텍스트로 변환 중 오류 발생: {e}")
            raise

    def build_vector_store(self):
        """벡터 스토어를 구축합니다."""
        # 저장된 벡터 스토어가 있는지 확인
        if self.check_stored_vector_store():
            return self.vector_store_id

        temp_files = []  # 임시 파일 추적용

        try:
            # 벡터 스토어 생성
            vector_store = self.client.vector_stores.create(name=self.name)
            self.vector_store_id = vector_store.id
            logger.info(f"새로운 벡터 스토어 생성 완료: {self.vector_store_id}")

            # 벡터 스토어 ID 저장
            self._save_vector_store_id(self.vector_store_id)

            # 텍스트 디렉토리에서 모든 JSON 파일 처리
            text_dir = self.input_dir / "text"

            # 각 섹션 파일을 벡터 스토어에 업로드
            for section_file in text_dir.glob("*.json"):
                logger.info(f"파일 처리 중: {section_file.name}")

                # JSON을 텍스트 파일로 변환
                text_file = self.json_to_text_file(section_file, self.data_dir)
                temp_files.append(text_file)

                # 텍스트 파일 업로드
                with open(text_file, "rb") as f:
                    file_response = self.client.vector_stores.files.upload_and_poll(
                        vector_store_id=self.vector_store_id, file=f
                    )

                logger.info(f"파일 업로드 완료: {section_file.name}")

            logger.info("벡터 스토어 구축 완료")
            return self.vector_store_id

        except Exception as e:
            logger.error(f"벡터 스토어 구축 중 오류 발생: {e}")
            raise

        finally:
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 중 오류 발생: {e}")

    def search_vector_store(self, query: str):
        """벡터 스토어에서 검색을 수행합니다."""
        try:
            # 벡터 스토어에서 검색 수행
            response = self.client.vector_stores.search(
                vector_store_id=self.vector_store_id, query=query
            )
            # 응답 구조 확인을 위한 로깅
            logger.info(f"검색 응답 구조: {response}")
            return response
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {e}")
            raise

    def list_uploaded_files(self, vector_store_id: str = None) -> dict:
        """벡터 스토어에 업로드된 파일 목록을 조회합니다.

        Args:
            vector_store_id (str, optional): 확인할 벡터 스토어의 ID.
                                           None인 경우 현재 인스턴스의 vector_store_id를 사용

        Returns:
            dict: 벡터 스토어 정보와 업로드된 파일 목록
        """
        try:
            if vector_store_id is None:
                vector_store_id = self.vector_store_id

            if not vector_store_id:
                raise ValueError("Vector store ID is required")

            # 파일 목록 조회
            files = self.client.vector_stores.files.list(vector_store_id)

            # 처리된 파일 수 계산
            processed_files = sum(1 for f in files.data if f.status == "processed")

            logger.info(f"\n업로드된 파일 현황:")
            logger.info(f"총 파일 수: {len(files.data)}")
            logger.info(f"처리 완료된 파일 수: {processed_files}")
            logger.info("\n파일 목록:")

            for file in files.data:
                logger.info(f"- {file.filename}")
                logger.info(f"  상태: {file.status}")
                logger.info(f"  생성 시간: {file.created_at}")

            return {
                "total_files": len(files.data),
                "processed_files": processed_files,
                "files": files.data,
            }

        except Exception as e:
            logger.error(f"파일 목록 조회 중 오류 발생: {str(e)}")
            raise

    def list_vector_stores(self) -> dict:
        """사용자의 모든 벡터 스토어 목록을 조회합니다.

        Returns:
            dict: 벡터 스토어 목록 정보
        """
        try:
            # 벡터 스토어 목록 조회
            stores = self.client.vector_stores.list()

            logger.info("\n벡터 스토어 목록:")
            for store in stores.data:
                logger.info(f"- 이름: {store.name}")
                logger.info(f"  ID: {store.id}")
                logger.info(f"  생성 시간: {store.created_at}")

            return stores

        except Exception as e:
            logger.error(f"벡터 스토어 목록 조회 중 오류 발생: {str(e)}")
            raise

    def check_stored_vector_store(self) -> bool:
        """저장된 벡터 스토어가 실제로 존재하는지 확인합니다."""
        if not self.vector_store_id:
            logger.info("저장된 벡터 스토어 ID가 없습니다.")
            return False

        try:
            # 벡터 스토어 목록에서 저장된 ID 검색
            stores = self.client.vector_stores.list()
            stored_store = next(
                (store for store in stores.data if store.id == self.vector_store_id),
                None,
            )

            if stored_store:
                logger.info(f"\n저장된 벡터 스토어 찾음:")
                logger.info(f"- 이름: {stored_store.name}")
                logger.info(f"- ID: {stored_store.id}")
                logger.info(f"- 생성 시간: {stored_store.created_at}")

                try:
                    # 파일 목록 조회
                    files = self.client.vector_stores.files.list(self.vector_store_id)

                    logger.info(f"\n업로드된 파일 목록 (총 {len(files.data)}개):")
                    for file in files.data:
                        # API 응답 형식이 변경될 수 있으므로 안전하게 처리
                        file_id = getattr(file, "id", "N/A")
                        file_status = getattr(file, "status", "N/A")
                        logger.info(f"- 파일 ID: {file_id}")
                        logger.info(f"  상태: {file_status}")
                except Exception as e:
                    logger.warning(f"파일 목록 조회 중 오류 발생: {str(e)}")
                    logger.warning(
                        "파일 목록을 확인할 수 없지만, 벡터 스토어는 존재합니다."
                    )

                return True
            else:
                logger.info(
                    f"저장된 벡터 스토어 ID({self.vector_store_id})를 찾을 수 없습니다."
                )
                return False

        except Exception as e:
            logger.error(f"벡터 스토어 확인 중 오류 발생: {str(e)}")
            return False

    def get_image_path(self, text_content):
        """텍스트 내용에 해당하는 이미지 파일 경로를 찾습니다."""
        # 모든 JSON 파일을 검사
        for json_file in self.text_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 텍스트 내용이 일치하는 섹션 찾기
                if data.get("text") == text_content:
                    # image_path 필드에서 이미지 경로 가져오기
                    image_path = data.get("image_path")
                    if image_path:
                        # 절대 경로를 상대 경로로 변환
                        image_path = Path(image_path)
                        if image_path.exists():
                            return image_path
        return None

    def display_search_results(self, search_results):
        """검색 결과와 해당하는 이미지를 표시합니다."""
        logger.info("\n검색 결과:")
        logger.info(f"검색 쿼리: {search_results.search_query}")
        logger.info(f"총 결과 수: {len(search_results.data)}")
        logger.info("=" * 80)

        # 응답 구조에 따라 결과 처리
        for i, result in enumerate(search_results.data, 1):
            logger.info(f"\n결과 {i}:")
            logger.info(f"파일 ID: {result.file_id}")
            logger.info(f"파일명: {result.filename}")
            logger.info(f"관련도 점수: {result.score:.4f}")

            if hasattr(result, "attributes") and result.attributes:
                logger.info("\n속성:")
                for key, value in result.attributes.items():
                    logger.info(f"- {key}: {value}")

            logger.info("\n내용:")
            # 모든 content 항목 표시
            for content_item in result.content:
                if content_item.type == "text":
                    # 원본 텍스트에서 제목과 본문 추출
                    text_parts = content_item.text.split("--- 본문 ---")
                    if len(text_parts) == 2:
                        metadata, content = text_parts
                        logger.info(metadata.strip())
                        logger.info("\n본문:")
                        logger.info(content.strip())

                    # 이미지 찾기 위한 파일명 생성
                    filename_parts = result.filename.replace("tmp", "").replace(
                        ".txt", ""
                    )
                    image_path = self.data_dir / "images" / f"{filename_parts}.png"

                    if image_path.exists():
                        logger.info(f"\n관련 이미지 파일: {image_path}")
                        # matplotlib으로 이미지 표시
                        img = Image.open(image_path)
                        plt.figure(figsize=(10, 10))
                        plt.imshow(img)
                        plt.axis("off")
                        plt.title(f"결과 {i} 이미지")
                        plt.show()
                    else:
                        logger.info("\n관련 이미지를 찾을 수 없습니다.")

            logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    # 테스트용 코드
    builder = VectorStoreBuilder()

    try:
        # 벡터 스토어 생성 또는 재사용
        if builder.check_stored_vector_store():
            logger.info("기존 벡터 스토어를 사용합니다.")

            # 검색 테스트
            test_query = "치아보험의 보장 내용은 무엇인가요?"
            logger.info(f"\n검색 쿼리: {test_query}")

            search_results = builder.search_vector_store(test_query)
            builder.display_search_results(search_results)
        else:
            logger.info("새로운 벡터 스토어를 생성합니다.")
            vector_store_id = builder.build_vector_store()
            if vector_store_id:
                logger.info(f"벡터 스토어 생성 완료: {vector_store_id}")
            else:
                logger.error("벡터 스토어 생성 실패")
                exit(1)

    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        exit(1)
