import re

#### 불용어 설정
STOPWORDS = {"추천","보여줘","찾아줘","알려줘","제품","상품","좀","해줘","주세요",
             "이거","그거","관련","가격","가격대","정도","비슷한","같은","디자인",
             "만원","이하","이상","미만","까지","예산","저렴한","저렴","싸게","걸로"}

def parse_query(message):
    words = re.findall(r"[가-힣A-Za-z]+", message)   # 한글/영문 토큰만
    keywords = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    price = parse_price(message)                     # 실습과제2: 가격 조건
    return {"keywords": " ".join(keywords), **price}


#### 가격 필터링 함수들  ===============
def _to_won(num_str, unit):
    """ 숫자 문자ㅣ열을 원 단위 정수로 파싱"""
    n = int(num_str.replace(',',''))
    return n*10000 if unit =='만' else n

def parse_price(message):
    """자연어 예산 표현에서 자격 조건 추출"""
    pmin, pmax, cheaper = None, None, False
    text = message.replace(' ','') # 공백 제거

    # 패턴1: 범위 — "3만원~5만원", "3~5만원", "3만원에서 5만원"
    m = re.search(r'(\d[\d,]*)(만)?원?(?:~|에서|부터)(\d[\d,]*)(만)?원', text)
    if m:
        a = _to_won(m.group(1), m.group(2) or m.group(4))   # "3~5만원"이면 앞 숫자도 '만' 적용
        b = _to_won(m.group(3), m.group(4))
        pmin, pmax = min(a, b), max(a, b)
    else:
        # 패턴2: 상한 — "N만원 이하/까지/미만/아래"
        m = re.search(r'(\d[\d,]*)(만)?원?(?:이하|까지|미만|아래)', text)
        if m:
            pmax = _to_won(m.group(1), m.group(2))
        # 패턴3: 하한 — "N만원 이상/부터/넘는"
        m = re.search(r'(\d[\d,]*)(만)?원?(?:이상|넘)', text)
        if m:
            pmin = _to_won(m.group(1), m.group(2))
        # 패턴4: "예산 N만원" / "N만원대" → 상한으로 처리
        if pmax is None:
            m = re.search(r'예산(\d[\d,]*)(만)?원?|(\d[\d,]*)(만)?원대', text)
            if m:
                if m.group(1):
                    pmax = _to_won(m.group(1), m.group(2))
                else:   # "5만원대" → 5만~6만 미만
                    base = _to_won(m.group(3), m.group(4))
                    pmin, pmax = base, base + 9999

    # 상대 표현 — "더 저렴한/더 싼/싸게" → 오름차순 정렬 강제
    if re.search(r'더저렴|더싸|저렴한|저렴하|싸게|싼거|싼걸', text):
        cheaper = True

    return {"price_min": pmin, "price_max": pmax, "cheaper": cheaper}

def filter_by_price(products, price_min=None, price_max=None):
    """가격 조건으로 필터링 후 가격 낮은 순 정렬 (과제 요구사항)"""
    result = []
    for p in products:
        try:
            price = int(str(p.get('lprice', 0)).replace(',', ''))
        except (ValueError, TypeError):
            continue
        if price <= 0:
            continue
        if price_min is not None and price < price_min:
            continue
        if price_max is not None and price > price_max:
            continue
        result.append(p)
    result.sort(key=lambda p: int(str(p.get('lprice', 0)).replace(',', '') or 0))
    return result


## 사용자의 쿼리가 웹 검색이 필요한지 판단하기.
def should_search_web(query):

    search_keywords = ['최신', '신상', '재고', '실시간', '현재', '오늘']
    
    return any(keyword in query for keyword in search_keywords)






