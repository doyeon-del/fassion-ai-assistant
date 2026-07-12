# app.py
"""AI 쇼핑 어시스턴트 ver2.0 — 엔트리포인트

실행 흐름: 모델 로드 → 벡터스토어/이미지 컬렉션 초기화 → Gradio 서버 실행
기능별 코드는 core/, services/, ui/ 패키지에 있습니다.
"""
from config import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT
from core.models import load_models
from services import vectorstore, vision
from ui.interface import create_interface

if __name__ == "__main__":
    print("AI 쇼핑 어시스턴트 시작...")
    load_models()
    vectorstore.init_vector_store()
    vision.init_image_collection()   # CLIP 이미지 검색 컬렉션 (기존엔 호출 누락으로 미동작)
    print("서버 시작...")

    demo = create_interface()
    demo.launch(
        server_port=GRADIO_SERVER_PORT,
        server_name=GRADIO_SERVER_NAME,
        share=False  # 공개 URL 생성 여부
    )
