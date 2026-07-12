# services/assistant.py
"""오케스트레이션 계층 — 이미지 업로드 흐름과 채팅 흐름을 조립합니다.

각 서비스(탐지/검색/저장/LLM)를 순서대로 호출하는 '지휘자' 역할만 하고,
실제 일은 각 서비스 모듈이 합니다.
"""
import traceback

from config import (CLIP_IMAGE_TOP_K, MAX_CONTEXT_TOKENS, MAX_SEARCH_COUNT,
                    VECTOR_SEARCH_K)
from core.models import registry
from core.state import state
from services import detection, llm, products, query, vectorstore, vision
from ui.formatters import format_product_html, format_product_list


def process_image(image):
    """업로드 이미지 처리: 탐지 → 상품 검색/저장 → 결과 HTML + 첫 인사

    Returns:
        (박스가 그려진 이미지, 탐지/추천 결과 HTML)
    """
    state.reset_for_new_image()
    vectorstore.reset_product_store()

    if image is None:
        return None, "<p>이미지를 업로드해주세요.</p>"

    try:
        img_with_boxes, detected_items = detection.detect_fashion_objects(image)
        if not detected_items:
            return img_with_boxes, "<p>패션 아이템을 찾을 수 없습니다.</p>"

        print(f"총 탐지된 아이템: {detected_items}")
        state.last_detected_items = detected_items

        # 탐지된 아이템으로 초기 상품 검색 + 저장
        found = products.search_products(' '.join(detected_items[:2]))
        vectorstore.save_products_to_vectorstore(found)
        vision.save_product_images_clip(found)                    # 상품 이미지 CLIP 벡터화
        clip_products = vision.search_by_image_clip(image, k=CLIP_IMAGE_TOP_K)
        merged = vision.merge_products(clip_products, found)      # 시각 유사 결과 우선 병합

        html_output = f"<h3>탐지된 아이템</h3><p>{', '.join(detected_items)}</p><hr>"
        html_output += format_product_html(merged)

        # 첫 인사 메시지를 대화 히스토리에 추가
        detected_items_str = ', '.join(detected_items)
        greeting = (
            f"안녕하세요! 이미지에서 {detected_items_str}을(를) 발견했습니다. "
            "어떤 스타일이나 브랜드를 선호하시나요? 예산도 알려주시면 더 정확한 추천을 도와드릴 수 있어요."
        )
        state.conversation_history.append({"role": "assistant", "content": greeting})

        return img_with_boxes, html_output

    except Exception as e:
        print(f"이미지 탐지 오류: {e}")
        traceback.print_exc()
        return image, f"<p>오류가 발생했습니다: {str(e)}</p>"


def _sync_gradio_history(history):
    """Gradio 히스토리에는 있는데 내부 히스토리에 없는 메시지를 보충합니다."""
    if not history:
        return
    for h in history:
        if not (isinstance(h, list) and len(h) == 2):
            continue
        user_content, assistant_content = h[0], h[1]
        if not user_content or not assistant_content:
            continue
        user_content = str(user_content)
        assistant_content = str(assistant_content)

        user_exists = any(m.get('content') == user_content
                          for m in state.conversation_history if m.get('role') == 'user')
        assistant_exists = any(m.get('content') == assistant_content
                               for m in state.conversation_history if m.get('role') == 'assistant')
        if not user_exists:
            state.conversation_history.append({"role": "user", "content": user_content})
        if not assistant_exists:
            state.conversation_history.append({"role": "assistant", "content": assistant_content})


def chat_response(message, history):
    """사용자 메시지 처리: 조건에 따라 웹/벡터 검색 → LLM 응답 생성

    Returns:
        (업데이트된 Gradio 히스토리, 상품 HTML)
    """
    token_manager = registry.token_manager

    _sync_gradio_history(history)

    # 현재 메시지 추가 및 토큰 관리
    state.conversation_history, _ = token_manager.manage_conversation_history(
        state.conversation_history, {"role": "user", "content": message})

    parsed = query.parse_query(message)
    keyword = parsed["keywords"] or " ".join(state.last_detected_items)  # 비면 탐지 아이템으로

    pmin, pmax = parsed["price_min"], parsed["price_max"]
    has_price = pmin is not None or pmax is not None or parsed["cheaper"]

    if (query.should_search_web(message) or has_price) and state.search_count < MAX_SEARCH_COUNT:
        # 웹에서 새로운 상품 검색
        print(f'\n웹 검색 : {keyword}\n')
        found = products.search_products(keyword)
        if has_price:
            found = query.filter_by_price(found, pmin, pmax)   # 예산 필터 + 가격순 정렬
        vectorstore.save_products_to_vectorstore(found)

        search_results = format_product_list(found)
        html_output = format_product_html(found)
        response = llm.generate_response_with_context(state.conversation_history, search_results)
        if found:
            response += f"\n\n{search_results}"
    else:
        # 저장해둔 벡터 DB에서 상품 검색
        print(f'\n벡터 DB 검색 : {keyword or message}\n')
        found = vectorstore.search_from_vectorstore(keyword or message, k=VECTOR_SEARCH_K)
        if has_price:
            found = query.filter_by_price(found, pmin, pmax)

        html_output = format_product_html(found)
        if found:
            search_results = format_product_list(found)
            response = llm.generate_response_with_context(state.conversation_history, search_results)
        else:
            response = llm.generate_response_with_context(state.conversation_history)

    # 응답을 대화 히스토리에 추가
    state.conversation_history, _ = token_manager.manage_conversation_history(
        state.conversation_history, {"role": "assistant", "content": response})

    # 토큰 통계 정보 추가
    token_stats = token_manager.get_token_stats(state.conversation_history)
    response += "\n\n(Tip. '최신', '신상', '재고', '실시간', '현재', '오늘' 키워드를 포함해보세요.)"
    response += (f"\n\n[검색: {state.search_count}/{MAX_SEARCH_COUNT}] "
                 f"[토큰: {token_stats['total']}/{MAX_CONTEXT_TOKENS}] "
                 f"[메시지: {token_stats['messages']}]")

    if history is None:
        history = []
    history.append([message, response])

    return history, html_output
