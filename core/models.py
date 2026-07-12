# core/models.py
"""AI 모델 로딩/보관 — 앱 전체가 공유하는 ModelRegistry"""
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import YOLO_MODEL_PATH, CLIP_MODEL_PATH, LLM_MODEL_PATH, EMBEDDING_MODEL_PATH
from utils import TokenManager


class ModelRegistry:
    """로드된 모델들을 속성으로 보관하는 저장소"""

    def __init__(self):
        self.device = "cpu"
        self.yolo = None
        self.clip_processor = None
        self.clip_model = None
        self.tokenizer = None
        self.llm = None
        self.embeddings = None
        self.token_manager = None


registry = ModelRegistry()   # 앱 전체에서 공유하는 유일한 인스턴스


def load_models():
    """YOLO, CLIP, LLM, 임베딩 모델을 로드해 registry에 채웁니다."""
    print("YOLO 모델 로딩 중...")
    registry.yolo = YOLO(YOLO_MODEL_PATH, task='segment')

    print("CLIP 모델 로딩 중...")
    registry.device = "cuda" if torch.cuda.is_available() else "cpu"
    registry.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH, use_fast=True)
    registry.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(registry.device)

    print("LLM 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    if tokenizer.pad_token is None:          # pad_token 없으면 eos로 대체
        tokenizer.pad_token = tokenizer.eos_token
    registry.tokenizer = tokenizer

    registry.llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    if torch.cuda.is_available():
        print(f"LLM이 GPU에서 실행됩니다: {torch.cuda.get_device_name()}")
    else:
        print("LLM이 CPU에서 실행됩니다")

    registry.token_manager = TokenManager(tokenizer)

    print("임베딩 모델 로딩 중...")
    registry.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    print("모든 모델 로딩 완료!")
