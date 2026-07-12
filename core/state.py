# core/state.py
"""세션 상태 — 대화 히스토리, 검색 횟수, 마지막 탐지 아이템"""


class SessionState:
    """앱 전체에서 공유하는 세션 상태 저장소"""

    def __init__(self):
        self.search_count = 0           # 네이버 API 호출 횟수
        self.conversation_history = []  # LLM에 전달할 대화 히스토리
        self.last_detected_items = []   # 마지막 이미지에서 탐지된 아이템 라벨

    def reset_conversation(self):
        """'초기화' 버튼: 대화와 검색 횟수를 리셋"""
        self.conversation_history = []
        self.search_count = 0

    def reset_for_new_image(self):
        """새 이미지 업로드: 대화와 탐지 아이템을 리셋"""
        self.conversation_history = []
        self.last_detected_items = []


state = SessionState()   # 앱 전체에서 공유하는 유일한 인스턴스
