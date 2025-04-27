import requests
import json
import base64
import time
import uuid


def ocr_image(image_path, api_url, secret_key):
    """
    CLOVA OCR General API를 호출하여 OCR 결과를 반환합니다.
    :param image_path: 이미지 파일 경로
    :param api_url: CLOVA OCR General API 엔드포인트
    :param secret_key: CLOVA OCR Secret Key
    :return: OCR 결과(JSON)
    """
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    headers = {"X-OCR-SECRET": secret_key, "Content-Type": "application/json"}
    data = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "lang": "ko",
        "images": [{"format": "png", "name": image_path, "data": img}],
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    return response.json()


# 사용 예시 (실제 사용 시 아래 주석 해제 및 api_url 수정)
# secret_key = 'S1BxTlhlS1FBQ1h1R0FhSnNxVVJ4ckZFeVNsQ0FSd0g='
# api_url = 'https://<your-endpoint>/general'
# result = ocr_image('page_1.png', api_url, secret_key)
# print(result)
