
# 샹품 리스트를 텍스트 형식으로 포맷팅. products에는 상품 리스트, 포멧팅된 상품 정보 문자열을 return
def format_product_list(products):

    if not products:
        return "상품을 찾을 수 없습니다."
    
    formatted = "추천 상품:\n"
    for i, product in enumerate(products[:3], 1):
        # 가격 처리 (문자열이거나 숫자일 수 있음)
        try:
            price = int(product.get('lprice', 0))
        except (ValueError, TypeError):
            price = 0
            
        formatted += f"{i}. {product.get('title', '상품명 없음')}\n"
        formatted += f"   가격: {price:,}원\n"
        formatted += f"   쇼핑몰: {product.get('mallName', '정보 없음')}\n"
        formatted += f"   구매하기: {product.get('link', '#')}\n\n"
    
    return formatted


## 상품 리스트를 html 형식으로 포맷팅하기.

def format_product_html(products):

    if not products:
        return "<p>상품을 찾을 수 없습니다.</p>"
    
    html = "<h3>추천 상품</h3>"
    html += "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; max-width: 900px;'>"
    
    for i, product in enumerate(products[:6], 1):  # 3개에서 6개로 변경
        # 가격 처리 (문자열이거나 숫자일 수 있음)
        try:
            price = int(product.get('lprice', 0))
        except (ValueError, TypeError):
            price = 0
            
        # 브랜드/제조사 정보
        brand_info = ""
        if product.get('brand'):
            brand_info = f"<p style='color: #888; font-size: 12px; margin: 2px 0;'>브랜드: {product['brand']}</p>"
        elif product.get('maker'):
            brand_info = f"<p style='color: #888; font-size: 12px; margin: 2px 0;'>제조사: {product['maker']}</p>"
        
        # 카테고리 정보
        category_info = ""
        if product.get('category1'):
            categories = [product.get('category1', '')]
            if product.get('category2'):
                categories.append(product.get('category2'))
            category_info = f"<p style='color: #999; font-size: 11px; margin: 2px 0;'>{' > '.join(categories)}</p>"
        
        html += f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa;'>
            <img src='{product.get('image', 'https://placehold.co/150')}' 
                 style='width: 100%; height: 120px; object-fit: cover; border-radius: 5px;'
                 onerror="this.onerror=null; this.src='https://placehold.co/150';">
            <h4 style='margin: 8px 0; font-size: 13px; line-height: 1.3; height: 32px; color: black; overflow: hidden;'>{product.get('title', '')[:40]}{'...' if len(product.get('title', '')) > 40 else ''}</h4>
            {brand_info}
            {category_info}
            <p style='color: #666; margin: 5px 0; font-size: 12px;'>쇼핑몰: {product.get('mallName', '')}</p>
            <p style='font-size: 16px; font-weight: bold; color: #ff6b6b;'>
                {price:,}원
            </p>
            <a href='{product.get('link', '#')}' target='_blank' rel='noreferrer noopener' 
               style='display: inline-block; background: #007bff; color: white; 
                      padding: 6px 14px; text-decoration: none; border-radius: 4px;
                      margin-top: 8px; font-size: 13px;'>
                구매하기
            </a>
        </div>
        """
    
    html += "</div>"
    return html