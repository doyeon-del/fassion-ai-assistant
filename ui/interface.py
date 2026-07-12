# ui/interface.py
"""Gradio 화면 구성 — 이벤트를 assistant의 함수에 연결만 합니다."""
import gradio as gr

from core.state import state
from services import assistant


def create_interface():
    """Gradio Blocks 인터페이스를 생성합니다."""
    with gr.Blocks(title="AI 쇼핑 어시스턴트") as demo:
        gr.Markdown("# AI 쇼핑 어시스턴트 ver2.0")
        gr.Markdown("패션 이미지를 업로드하고 AI와 대화하며 쇼핑을 즐겨보세요!")

        with gr.Row():
            # 왼쪽 열: 이미지 업로드 및 탐지
            with gr.Column(scale=1):
                image_input = gr.Image(label="상품 이미지", type="pil")
                detect_btn = gr.Button("상품 탐지", variant="primary")
                output_image = gr.Image(label="탐지 결과")

            # 오른쪽 열: 채팅 인터페이스
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    label="메시지",
                    placeholder="상품에 대해 궁금한 점을 물어보세요...",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("전송", variant="primary")
                    clear = gr.Button("초기화")

        # 두 번째 행: 탐지 및 추천 상품 목록 표시
        with gr.Row():
            detection_info = gr.HTML(label="탐지 정보")

        def handle_detection_and_chat(image):
            """이미지 탐지 후 채팅창에 첫 인사 표시"""
            img, html = assistant.process_image(image)
            if state.conversation_history and state.conversation_history[-1]['role'] == 'assistant':
                chat_history = [[None, state.conversation_history[-1]['content']]]
            else:
                chat_history = []
            return img, html, chat_history

        def handle_chat(message, history):
            """빈 메시지는 무시 (gr.update()로 상품 HTML은 그대로 유지)"""
            if not message or not message.strip():
                return history, gr.update()
            return assistant.chat_response(message, history)

        def clear_conversation():
            state.reset_conversation()
            return None, ""  # chatbot과 msg 둘 다 초기화

        detect_btn.click(
            fn=handle_detection_and_chat,
            inputs=image_input,
            outputs=[output_image, detection_info, chatbot]
        )
        msg.submit(fn=handle_chat, inputs=[msg, chatbot], outputs=[chatbot, detection_info]).then(
            fn=lambda: "", outputs=msg
        )
        submit.click(fn=handle_chat, inputs=[msg, chatbot], outputs=[chatbot, detection_info]).then(
            fn=lambda: "", outputs=msg
        )
        clear.click(clear_conversation, outputs=[chatbot, msg])

    return demo
