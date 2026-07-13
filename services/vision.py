# services/vision.py
"""CLIP 기반 상품 이미지 유사도 검색 (실습과제3)"""
import io

import chromadb
import requests
import torch
from PIL import Image

from config import CHROMA_PERSIST_DIR, CLIP_SAVE_MAX_ITEMS
from core.models import registry

image_collection = None   # 상품 이미지 CLIP 벡터 컬렉션 (이 모듈이 소유)

# ── 크로스모달 검색용 한국어→영어 매핑 ──────────────────────────────
# CLIP은 영어로 학습돼 한국어 텍스트 검색이 약하다. 앱 어휘(색상·아이템)를
# 영어로 번역해 기존 CLIP 텍스트 인코더에 넣으면 이미지 벡터와 같은 공간에 놓인다.
_COLOR_KO_EN = {
    "검정": "black", "흰": "white", "회색": "gray", "빨간": "red",
    "주황": "orange", "갈색": "brown", "노란": "yellow", "베이지": "beige",
    "초록": "green", "하늘": "sky blue", "파란": "blue", "보라": "purple", "분홍": "pink",
}
# 더 구체적인 단어를 먼저 매칭하도록 순서 유지 (예: '청바지'를 '바지'보다 먼저)
_ITEM_KO_EN = {
    "청바지": "jeans", "반바지": "shorts", "티셔츠": "t-shirt", "원피스": "dress",
    "셔츠": "shirt", "니트": "knit sweater", "스커트": "skirt", "치마": "skirt",
    "자켓": "jacket", "재킷": "jacket", "아우터": "jacket", "조끼": "vest",
    "코트": "coat", "후드": "hoodie", "바지": "pants",
}
_SLEEVE_KO_EN = {"반팔": "short-sleeve", "긴팔": "long-sleeve"}


def _korean_to_english_query(korean_text):
    """한국어 설명에서 색상/소매/아이템 단어를 뽑아 CLIP용 영어 문장으로 변환.

    매핑되는 아이템이 하나도 없으면 None (→ 호출부에서 크로스모달 검색 건너뜀).
    """
    text = korean_text.replace(" ", "")
    color = next((en for ko, en in _COLOR_KO_EN.items() if ko in text), "")
    sleeve = next((en for ko, en in _SLEEVE_KO_EN.items() if ko in text), "")
    item = next((en for ko, en in _ITEM_KO_EN.items() if ko in text), "")
    if not item:
        return None
    desc = " ".join(part for part in (color, sleeve, item) if part)
    return f"a photo of a {desc}"   # CLIP은 "a photo of ~" 캡션 형태에 강함


def init_image_collection(reset=False):
    """상품 이미지 CLIP 벡터용 별도 Chroma 컬렉션 초기화"""
    global image_collection
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if reset:
        try:
            client.delete_collection("product_images")
        except Exception:
            pass
    image_collection = client.get_or_create_collection(
        "product_images", metadata={"hnsw:space": "cosine"})


def extract_clip_features(image):
    """CLIP 모델로 이미지의 특징 벡터를 추출합니다.

    Args:
        image: PIL Image 객체

    Returns:
        이미지 특징 벡터 (numpy array, shape (1, 512))
    """
    inputs = registry.clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(registry.device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = registry.clip_model.get_image_features(**inputs)
    return image_features.cpu().numpy()


def extract_clip_text_features(text):
    """CLIP 텍스트 인코더로 문장의 특징 벡터를 추출합니다. (이미지 벡터와 같은 공간)

    주의: fast 프로세서는 text 인자에 버그가 있어 tokenizer를 직접 사용한다.
    """
    tok = registry.clip_processor.tokenizer(text=text, return_tensors="pt", padding=True)
    tok = {k: v.to(registry.device) for k, v in tok.items()}
    with torch.no_grad():
        text_features = registry.clip_model.get_text_features(**tok)
    return text_features.cpu().numpy()


def search_by_text_clip(korean_text, k=5):
    """한국어 설명 → 영어 변환 → CLIP 텍스트 벡터로 상품 이미지 검색 (크로스모달)"""
    if image_collection is None or image_collection.count() == 0:
        return []
    en_query = _korean_to_english_query(korean_text)
    if en_query is None:   # 매핑되는 패션 아이템이 없으면 크로스모달 검색 생략
        return []
    vec = extract_clip_text_features(en_query)[0]
    res = image_collection.query(query_embeddings=[vec.tolist()],
                                 n_results=min(k, image_collection.count()))
    products = []
    for meta in (res.get('metadatas') or [[]])[0]:
        products.append({
            'title': meta.get('title', ''), 'link': meta.get('link', ''),
            'lprice': meta.get('price', '0'), 'hprice': '0',
            'mallName': meta.get('mall', ''), 'image': meta.get('image', ''),
            'brand': meta.get('brand', ''), 'maker': '',
            'category1': '', 'category2': '',
        })
    print(f"[CLIP] '{korean_text}' → '{en_query}' 텍스트 검색 {len(products)}개")
    return products


def save_product_images_clip(products, max_items=CLIP_SAVE_MAX_ITEMS):
    """상품 이미지 URL → CLIP 벡터 → product_images 컬렉션 저장"""
    if not products or image_collection is None:
        return
    ids, embs, metas = [], [], []
    for p in products[:max_items]:            # 비용 고려: 상위 N개만 벡터화
        url = p.get('image', '')
        pid = str(p.get('productId') or p.get('link', ''))
        if not url or not pid:
            continue
        try:
            resp = requests.get(url, timeout=5)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"[CLIP] 이미지 다운로드 실패, 건너뜀: {url} ({e})")
            continue
        vec = extract_clip_features(img)[0]   # (1,512) → (512,)
        ids.append(pid)
        embs.append(vec.tolist())
        metas.append({
            'title': p.get('title', ''), 'link': p.get('link', ''),
            'price': str(p.get('lprice', '0')), 'image': url,
            'mall': p.get('mallName', ''), 'brand': p.get('brand', ''),
        })
    if ids:
        image_collection.upsert(ids=ids, embeddings=embs, metadatas=metas)
        print(f"[CLIP] 상품 이미지 벡터 {len(ids)}개 저장")


def search_by_image_clip(image, k=5):
    """업로드 이미지를 CLIP 벡터화해 시각적으로 유사한 상품 검색"""
    if image_collection is None or image_collection.count() == 0:
        return []
    vec = extract_clip_features(image)[0]
    res = image_collection.query(query_embeddings=[vec.tolist()],
                                 n_results=min(k, image_collection.count()))
    products = []
    for meta in (res.get('metadatas') or [[]])[0]:
        products.append({
            'title': meta.get('title', ''), 'link': meta.get('link', ''),
            'lprice': meta.get('price', '0'), 'hprice': '0',
            'mallName': meta.get('mall', ''), 'image': meta.get('image', ''),
            'brand': meta.get('brand', ''), 'maker': '',
            'category1': '', 'category2': '',
        })
    print(f"[CLIP] 시각 유사 상품 {len(products)}개")
    return products


def merge_products(image_products, text_products):
    """이미지 유사 결과 + 텍스트 검색 결과 병합 (link 기준 중복 제거)"""
    seen, merged = set(), []
    for p in image_products + text_products:   # 시각적으로 유사한 것 우선
        link = p.get('link', '')
        if link and link not in seen:
            seen.add(link)
            merged.append(p)
    return merged
