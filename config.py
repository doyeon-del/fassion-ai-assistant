# config.py
import os
from huggingface_hub import hf_hub_download

# 네이버 API 설정
# 직접 발급 받은 네이버 API 키로 설정하여주세요.
try:
    from secrets_local import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
except ImportError:
    NAVER_CLIENT_ID = "YOUR_NAVER_CLIENT_ID"      # secrets_local.py에 실제 키 작성
    NAVER_CLIENT_SECRET = "YOUR_NAVER_CLIENT_SECRET"

# 모델 설정
YOLO_MODEL_PATH = hf_hub_download("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt")  # DeepFashion2 사전학습 모델
LLM_MODEL_PATH = "Bllossom/llama-3.2-Korean-Bllossom-3B"
EMBEDDING_MODEL_PATH = "jhgan/ko-sroberta-multitask"
CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"

# ChromaDB 설정
CHROMA_PERSIST_DIR = "./chroma_db"

# 크롤링 설정
CRAWL_CATEGORIES = ['셔츠', '바지', '원피스', '자켓', '스커트', '니트', '코트']
CRAWL_MAX_PAGES = 3

# 검색 설정
MAX_SEARCH_COUNT = 10  # 최대 API 검색 횟수
SEARCH_DISPLAY_COUNT = 10  # 한 번에 가져올 상품 수
VECTOR_SEARCH_K = 5  # 벡터 스토어에서 검색할 문서 수

# 탐지/CLIP 설정
DETECTION_CONF_THRESHOLD = 0.5  # YOLO 탐지 신뢰도 임계값
CLIP_SAVE_MAX_ITEMS = 6  # CLIP 벡터로 저장할 상품 이미지 수
CLIP_IMAGE_TOP_K = 3  # 업로드 이미지와 유사도 검색할 상품 수

# RAG / LLM 생성 설정
LLM_TEMPERATURE = 0.3  # 사실 기반 답변용 낮은 온도 (창의성↓ 충실도↑)
RAG_CONTEXT_MAX_ITEMS = 5  # LLM 컨텍스트에 넣을 최대 상품 수
RAG_MAX_DISTANCE = 0.75  # 이 거리보다 먼(덜 유사한) 검색 결과는 제외 (cosine 거리 0~2, 낮을수록 유사)
                         # ko-sroberta 기준 관련 상품 ~0.5, 무관 상품 ~0.8+ 이라 그 사이로 설정

# 토큰 관리 설정
MAX_CONTEXT_TOKENS = 2048  # 최대 컨텍스트 토큰 수
MAX_GENERATION_TOKENS = 512  # 최대 생성 토큰 수

# Gradio 서버 설정
GRADIO_SERVER_PORT = 7860
GRADIO_SERVER_NAME = "0.0.0.0"  # 모든 IP에서 접근 가능, "127.0.0.1"로 변경하면 로컬만 접근 가능