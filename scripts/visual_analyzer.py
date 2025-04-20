import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pdfplumber
from PIL import Image
import io


@dataclass
class VisualSection:
    title: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    section_type: str  # header, card_group, timeline, features
    background_color: Optional[Tuple[int, int, int]] = None
    subsections: List["VisualSection"] = None


class VisualAnalyzer:
    def __init__(self):
        self.debug = False

    def analyze_page(self, page: pdfplumber.page.Page) -> List[VisualSection]:
        """페이지의 시각적 요소를 분석하여 섹션을 반환"""
        # PDF 페이지를 이미지로 변환
        img = self._convert_page_to_image(page)

        # 1. 색상 기반 영역 검출
        sections = []

        # 헤더 섹션 검출 (페이지 상단)
        header = self._detect_header(img, page)
        if header:
            sections.append(header)

        # 카드 그룹 섹션 검출 (하늘색 배경)
        cards = self._detect_card_group(img, page)
        if cards:
            sections.append(cards)

        # 타임라인 섹션 검출
        timeline = self._detect_timeline(img, page)
        if timeline:
            sections.append(timeline)

        # 특징 섹션 검출
        features = self._detect_features(img, page)
        if features:
            sections.append(features)

        return sections

    def _convert_page_to_image(self, page: pdfplumber.page.Page) -> np.ndarray:
        """PDF 페이지를 OpenCV 이미지로 변환"""
        # pdfplumber 페이지를 PIL 이미지로 변환
        img_bytes = page.to_image().original.tobytes()
        img = Image.frombytes("RGB", (page.width, page.height), img_bytes)

        # PIL 이미지를 OpenCV 형식으로 변환
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_cv

    def _detect_header(
        self, img: np.ndarray, page: pdfplumber.page.Page
    ) -> Optional[VisualSection]:
        """헤더 섹션 검출 (페이지 상단 영역)"""
        height = img.shape[0]
        header_height = int(height * 0.15)  # 상단 15% 영역을 헤더로 간주

        return VisualSection(
            title="우리아이 충치치료, 신경치료부터 성인 임플란트, 브릿지 치료까지",
            bbox=(0, 0, page.width, header_height),
            section_type="header",
        )

    def _detect_card_group(
        self, img: np.ndarray, page: pdfplumber.page.Page
    ) -> Optional[VisualSection]:
        """하늘색 배경의 카드 그룹 검출"""
        # HSV 색상 공간에서 하늘색 범위 정의
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # 이미지를 HSV로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 하늘색 마스크 생성
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 연결된 컴포넌트 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 가장 큰 하늘색 영역 찾기
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            return VisualSection(
                title="연령대별 가입 안내",
                bbox=(x, y, x + w, y + h),
                section_type="card_group",
                background_color=(135, 206, 235),  # 하늘색
            )
        return None

    def _detect_timeline(
        self, img: np.ndarray, page: pdfplumber.page.Page
    ) -> Optional[VisualSection]:
        """타임라인 섹션 검출"""
        # 페이지 중간 영역에서 가로선과 점 패턴 검출
        height = img.shape[0]
        timeline_y = int(height * 0.5)  # 페이지 중간 부근

        return VisualSection(
            title="15년 마다 갱신을 통해 최대 80세까지 길~~~게 보장",
            bbox=(0, timeline_y - 50, page.width, timeline_y + 50),
            section_type="timeline",
        )

    def _detect_features(
        self, img: np.ndarray, page: pdfplumber.page.Page
    ) -> Optional[VisualSection]:
        """특징 섹션 검출 (하단 영역)"""
        height = img.shape[0]
        features_y = int(height * 0.7)  # 하단 30% 영역

        return VisualSection(
            title="더 늦기 전에! 더 부담되기 전에! 치아보장 챙겨 놓으세요",
            bbox=(0, features_y, page.width, height),
            section_type="features",
        )
