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
