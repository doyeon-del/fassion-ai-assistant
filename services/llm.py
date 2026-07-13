# services/llm.py
"""LLM 응답 생성 + RAG 컨텍스트 구성"""
import torch

from config import LLM_TEMPERATURE, MAX_GENERATION_TOKENS, RAG_CONTEXT_MAX_ITEMS
from core.models import registry

# 프롬프트 개선: 역할·근거·인용·빈 결과 처리·형식을 규칙으로 명시해 환각을 억제한다.
SYSTEM_PROMPT = (
    "당신은 패션 쇼핑을 돕는 친절하고 전문적인 AI 어시스턴트입니다.\n\n"
    "[규칙]\n"
    "1. 반드시 아래 제공된 '상품 목록'의 정보만 근거로 답하세요. "
    "목록에 없는 상품명·가격·브랜드를 절대 지어내지 마세요.\n"
    "2. 상품을 추천할 때는 '[상품 1]'처럼 번호로 지칭하세요.\n"
    "3. 상품 목록이 비어 있으면 정중히 찾지 못했다고 말하고 다른 검색어나 조건을 제안하세요.\n"
    "4. 사용자가 예산이나 취향을 말했다면 그에 맞는 상품을 우선 추천하고 이유를 한 문장으로 설명하세요.\n"
    "5. 답변은 자연스러운 한국어로 2~4문장. 예시 대화나 '[제품 이름]' 같은 빈칸 표시를 만들지 말고, 한 번만 답하세요."
)


def build_context(products, max_items=RAG_CONTEXT_MAX_ITEMS):
    """상품 리스트를 LLM이 근거로 삼기 좋은 번호형 컨텍스트로 변환합니다.

    사용자에게 보여주는 목록(format_product_list)과 달리, URL 등 불필요한 정보를
    빼고 한 줄로 압축해 토큰을 아끼고 모델이 상품을 번호로 인용하기 쉽게 만든다.
    """
    if not products:
        return "[상품 목록]\n(검색된 상품이 없습니다.)"

    lines = ["[상품 목록]"]
    for i, p in enumerate(products[:max_items], 1):
        try:
            price = int(str(p.get('lprice', 0)).replace(',', ''))
        except (ValueError, TypeError):
            price = 0
        parts = [f"[상품 {i}] {p.get('title', '')[:60]}", f"{price:,}원"]
        brand = p.get('brand') or p.get('maker')
        if brand:
            parts.append(f"브랜드: {brand}")
        if p.get('mallName'):
            parts.append(f"쇼핑몰: {p.get('mallName')}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def generate_response_with_context(conversation_history, products=None):
    """대화 컨텍스트(+상품 목록)를 바탕으로 LLM 응답을 생성합니다.

    Args:
        conversation_history: 대화 히스토리
        products: 검색된 상품 리스트. None이면 상품 컨텍스트를 넣지 않음(순수 대화).
                  빈 리스트([])면 '상품 없음' 컨텍스트를 넣어 규칙3이 동작하게 함.
    """
    tokenizer = registry.tokenizer
    model = registry.llm

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    history = list(conversation_history)
    # 상품 컨텍스트를 '현재 질문' 바로 옆에 붙여 모델이 확실히 근거로 보게 함
    if products is not None and history and history[-1]["role"] == "user":
        question = history[-1]["content"]
        context = build_context(products)
        history[-1] = {
            "role": "user",
            "content": (
                f"{context}\n\n"
                f"위 상품 목록을 근거로 다음 질문에 답해주세요.\n"
                f"질문: {question}"
            )
        }
    messages.extend(history)

    # 채팅 템플릿 적용 (attention_mask까지 함께 받기)
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to(model.device)

    terminators = [tokenizer.eos_token_id,
                   tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_GENERATION_TOKENS,
            eos_token_id=terminators,
            do_sample=True, temperature=LLM_TEMPERATURE, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 생성된 부분만 추출 (입력 토큰 이후)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()

    if len(response) < 5:
        response = "죄송합니다. 이해하지 못했습니다. 다시 한 번 말씀해 주시겠어요?"

    return response
