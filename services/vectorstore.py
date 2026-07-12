# services/vectorstore.py
"""상품 텍스트 벡터스토어(Chroma) 관리"""
import os

from langchain_community.vectorstores import Chroma

from config import CHROMA_PERSIST_DIR, SEARCH_DISPLAY_COUNT
from core.models import registry

vector_store = None   # 상품 텍스트 벡터스토어 (이 모듈이 소유)


def init_vector_store():
    """Chroma 벡터스토어에 연결합니다. (임베딩 모델 로드 후에 호출해야 함)"""
    global vector_store
    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=registry.embeddings,
    )


def reset_product_store():
    """이전 사진의 상품이 섞이지 않게 상품 벡터 컬렉션을 비움"""
    try:
        vector_store.delete_collection()
    except Exception:
        pass
    init_vector_store()


def save_products_to_vectorstore(products):
    """검색된 상품 정보를 벡터 스토어에 저장합니다."""
    if not products:
        return

    texts = []
    metadatas = []

    for product in products:
        # 상품 정보를 텍스트로 변환 (검색 효율성을 위해 더 많은 정보 포함)
        text_parts = [
            product.get('title', ''),
            product.get('mallName', ''),
            f"가격: {product.get('lprice', '0')}원",
            product.get('brand', ''),
            product.get('maker', ''),
            product.get('category1', ''),
            product.get('category2', '')
        ]
        texts.append(' '.join([part for part in text_parts if part]))

        metadatas.append({
            'title': product.get('title', ''),
            'link': product.get('link', ''),
            'price': product.get('lprice', '0'),
            'hprice': product.get('hprice', '0'),
            'mall': product.get('mallName', ''),
            'image': product.get('image', 'https://placehold.co/150'),
            'productId': str(product.get('productId', '')),
            'brand': product.get('brand', ''),
            'maker': product.get('maker', ''),
            'category1': product.get('category1', ''),
            'category2': product.get('category2', '')
        })

    vector_store.add_texts(texts=texts, metadatas=metadatas)


def _meta_to_product(meta):
    """Chroma 메타데이터 → 상품 dict 복원 (중복 로직 통합)"""
    return {
        'title': meta.get('title', ''),
        'link': meta.get('link', ''),
        'lprice': meta.get('price', '0'),
        'hprice': meta.get('hprice', '0'),
        'mallName': meta.get('mall', ''),
        'image': meta.get('image', 'https://placehold.co/150'),
        'productId': meta.get('productId', ''),
        'brand': meta.get('brand', ''),
        'maker': meta.get('maker', ''),
        'category1': meta.get('category1', ''),
        'category2': meta.get('category2', ''),
    }


def search_from_vectorstore(query, k=SEARCH_DISPLAY_COUNT):
    """벡터 스토어에서 유사한 상품을 검색합니다."""
    docs = vector_store.similarity_search(query, k=k)
    return [_meta_to_product(doc.metadata) for doc in docs if doc.metadata]
