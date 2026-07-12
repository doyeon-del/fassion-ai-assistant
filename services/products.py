# services/products.py
"""네이버 쇼핑 API 상품 검색"""
import requests

from config import (MAX_SEARCH_COUNT, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET,
                    SEARCH_DISPLAY_COUNT)
from core.state import state
from services import vectorstore


def search_products(query):
    """네이버 쇼핑 API로 상품을 검색합니다.

    API 호출 한도를 넘으면 벡터 스토어에서, API 키가 없으면 더미 데이터를 반환합니다.
    """
    if state.search_count >= MAX_SEARCH_COUNT:
        return vectorstore.search_from_vectorstore(query)

    if not NAVER_CLIENT_ID or NAVER_CLIENT_ID == "YOUR_NAVER_CLIENT_ID":
        return generate_dummy_products(query)

    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {"query": query, "display": SEARCH_DISPLAY_COUNT, "sort": "sim"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            state.search_count += 1
            items = response.json()['items']

            # productType이 1, 2, 3인 일반상품만 필터링 (그 외는 중고/단종 상품)
            filtered_items = []
            for item in items:
                product_type = int(item.get('productType', 0))
                if product_type not in [1, 2, 3]:
                    continue

                # 데이터 정합성 처리
                if 'image' not in item or not item['image']:
                    item['image'] = 'https://placehold.co/150'
                item['lprice'] = str(item.get('lprice', 0))
                item['hprice'] = str(item.get('hprice', 0))
                if 'title' in item:   # title의 <b> 태그 제거
                    item['title'] = item['title'].replace('<b>', '').replace('</b>', '')
                item['brand'] = item.get('brand', '')
                item['maker'] = item.get('maker', '')
                item['category1'] = item.get('category1', '')
                item['category2'] = item.get('category2', '')

                filtered_items.append(item)

            print(f"[네이버] 상품 {len(filtered_items)}개 검색 (호출 {state.search_count}/{MAX_SEARCH_COUNT})")
            return filtered_items
    except Exception as e:
        print(f"API 호출 오류: {e}")

    return generate_dummy_products(query)


def generate_dummy_products(query):
    """테스트용 더미 상품 데이터를 생성합니다."""
    categories = ['패션의류', '패션잡화', '화장품/미용', '디지털/가전', '가구/인테리어']
    brands = ['브랜드A', '브랜드B', '브랜드C', '브랜드D', '브랜드E']

    products = []
    for i in range(SEARCH_DISPLAY_COUNT):
        products.append({
            'title': f'{query} 상품 {i+1}',
            'link': f'https://example.com/product/{i+1}',
            'lprice': str(10000 + i * 5000),
            'hprice': str(15000 + i * 5000),
            'mallName': f'쇼핑몰 {i+1}',
            'image': f'https://placehold.co/150?text=Product{i+1}',
            'productId': 1000000 + i,
            'productType': 1,
            'brand': brands[i % len(brands)],
            'maker': f'제조사 {i+1}',
            'category1': categories[i % len(categories)],
            'category2': '서브카테고리'
        })
    return products
