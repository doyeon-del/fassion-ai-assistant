# services/llm.py
"""LLM 응답 생성"""
import torch

from config import MAX_GENERATION_TOKENS
from core.models import registry

SYSTEM_PROMPT = (
    "당신은 친절한 AI 쇼핑 어시스턴트입니다. 패션 상품에 대한 정보를 제공하고 추천합니다. "
    "반드시 아래 '검색 결과'에 있는 실제 정보만 사용하고, 정보가 없으면 모른다고 답하세요. "
    "[제품 이름] 같은 빈 자리 표시자나 예시 대화를 절대 만들지 마세요. 한 번만 답하세요."
)


def generate_response_with_context(conversation_history, search_results=None):
    """대화 컨텍스트(+검색 결과)를 바탕으로 LLM 응답을 생성합니다."""
    tokenizer = registry.tokenizer
    model = registry.llm

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    history = list(conversation_history)
    # 검색 결과를 '현재 질문' 바로 옆에 붙여서 모델이 확실히 보게 함
    if search_results and history and history[-1]["role"] == "user":
        question = history[-1]["content"]
        history[-1] = {
            "role": "user",
            "content": (
                f"[검색 결과]\n{search_results}\n\n"
                f"위 검색 결과를 참고해서 아래 질문에 답해줘.\n"
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
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 생성된 부분만 추출 (입력 토큰 이후)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()

    if len(response) < 5:
        response = "죄송합니다. 이해하지 못했습니다. 다시 한 번 말씀해 주시겠어요?"

    return response
