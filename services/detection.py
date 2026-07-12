# services/detection.py
"""YOLO 기반 패션 아이템 탐지 + 주요 색상 인식 (실습과제1)"""
import cv2
import numpy as np
from PIL import Image

from config import DETECTION_CONF_THRESHOLD
from core.models import registry

# DeepFashion2 영문 클래스명 → 한국어 (색상과 합쳐 네이버 검색 키워드로 사용)
CLASS_NAME_KO = {
    'short_sleeved_shirt': '반팔 셔츠',
    'long_sleeved_shirt': '긴팔 셔츠',
    'short_sleeved_outwear': '반팔 아우터',
    'long_sleeved_outwear': '아우터',
    'vest': '조끼',
    'sling': '슬링',
    'shorts': '반바지',
    'trousers': '바지',
    'skirt': '스커트',
    'short_sleeved_dress': '반팔 원피스',
    'long_sleeved_dress': '긴팔 원피스',
    'vest_dress': '조끼 원피스',
    'sling_dress': '슬링 원피스',
}


def _dominant_color_from_pixels(pixels):
    """픽셀 배열에서 k-means로 대표색 추출 (군집 3개 중 가장 큰 것)"""
    if len(pixels) == 0:
        return None
    pixels = pixels.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 32:                       # 군집화하기엔 표본 부족 → 평균색
        return pixels.mean(axis=0).astype(int)
    if len(pixels) > 4096:                     # 속도용 다운샘플
        pixels = pixels[::len(pixels) // 4096 + 1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    return centers[np.argmax(counts)].astype(int)   # (R, G, B)


def extract_dominant_color_masked(image, polygon):
    """seg 마스크(폴리곤) 안쪽 = 옷 픽셀만으로 주요 색상 추출

    bbox 방식과 달리 배경/피부 픽셀이 섞이지 않아 색상 정확도가 높다.
    polygon은 원본 이미지 좌표계의 외곽선 (ultralytics masks.xy)
    """
    if polygon is None or len(polygon) < 3:
        return None
    img = np.array(image)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    # 마스크 경계를 살짝 깎아 배경이 섞이는 가장자리 픽셀 제거
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
    pixels = img[eroded == 1]
    if len(pixels) < 32:                       # 깎았더니 너무 작으면 원래 마스크로
        pixels = img[mask == 1]
    if len(pixels) < 32:
        return None
    return _dominant_color_from_pixels(pixels)


def extract_dominant_color(image, box):
    """(폴백) 탐지 영역(bbox) 중앙부에서 주요 색상 추출 — 마스크가 없을 때 사용"""
    x1, y1, x2, y2 = [int(v) for v in box]
    img = np.array(image)                      # PIL(RGB) -> numpy
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # 배경 영향 줄이기: 중앙 60% 영역만 사용
    h, w = crop.shape[:2]
    crop = crop[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (64, 64))          # 속도용 다운샘플
    return _dominant_color_from_pixels(crop)


def rgb_to_korean_color(rgb):
    """RGB를 HSV로 변환해 한국어 색상명으로 매핑"""
    if rgb is None:
        return ""
    r, g, b = [int(v) for v in rgb]
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])   # H:0~180, S/V:0~255

    # 무채색 (채도/명도 기준)
    if v < 50:
        return "검정"
    if s < 40:
        return "흰" if v > 200 else "회색"
    # 유채색 (Hue 구간, OpenCV H는 0~180)
    if h < 10 or h >= 170:
        return "빨간"
    if h < 22:
        return "주황" if v > 150 else "갈색"
    if h < 33:
        return "노란" if v > 150 else "베이지"
    if h < 78:
        return "초록"
    if h < 100:
        return "하늘"
    if h < 130:
        return "파란"
    if h < 155:
        return "보라"
    return "분홍"


def detect_fashion_objects(image):
    """이미지에서 패션 아이템을 탐지합니다. (탐지만 담당 — 검색/저장은 assistant가)

    Args:
        image: PIL Image 객체

    Returns:
        (바운딩 박스가 그려진 PIL 이미지, "빨간 반팔 셔츠" 형태의 라벨 리스트)
    """
    results = registry.yolo(image)
    img_with_boxes = np.array(image).copy()
    detected_items = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        masks = r.masks   # seg 모델이 주는 마스크 (박스와 같은 순서)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            if conf < DETECTION_CONF_THRESHOLD:
                continue

            cls = int(box.cls[0].item())
            class_name = registry.yolo.names[cls]

            # 옷 영역(seg 마스크) 픽셀만으로 색상 추출 → "빨간 반팔 셔츠" 형태
            rgb = None
            if masks is not None and i < len(masks.xy):
                rgb = extract_dominant_color_masked(image, masks.xy[i])
            if rgb is None:   # 마스크가 없거나 너무 작으면 bbox 방식으로 폴백
                rgb = extract_dominant_color(image, (x1, y1, x2, y2))
            color_name = rgb_to_korean_color(rgb)
            class_ko = CLASS_NAME_KO.get(class_name, class_name)
            item_label = f"{color_name} {class_ko}".strip()
            print(f"탐지된 아이템: {item_label} (신뢰도: {conf:.2f})")

            cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"{class_name} {conf:.2f}",
                        (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_items.append(item_label)

    return Image.fromarray(img_with_boxes), detected_items
