# AI 쇼핑 어시스턴트 프로젝트 ver2.0

## 프로젝트 구조
```
fashion-ai-assistant/
├── app.py                 # 엔트리포인트 (모델 로드 + 서버 실행)
├── config.py              # 환경 설정
├── utils.py               # 유틸리티 함수 (토큰 관리 등)
├── crawl_products.py      # 상품 크롤링
├── core/
│   ├── models.py          # AI 모델 로딩/보관 (ModelRegistry)
│   └── state.py           # 세션 상태 (대화 히스토리, 검색 횟수)
├── services/
│   ├── detection.py       # YOLO 패션 아이템 탐지 + 색상 인식
│   ├── vision.py          # CLIP 이미지 특징 추출/유사도 검색
│   ├── products.py        # 네이버 쇼핑 API 상품 검색
│   ├── vectorstore.py     # ChromaDB 상품 벡터스토어
│   ├── query.py           # 쿼리 파싱 (키워드/가격 조건)
│   ├── llm.py             # LLM 응답 생성
│   └── assistant.py       # 이미지/채팅 처리 오케스트레이션
├── ui/
│   ├── formatters.py      # 상품 HTML/텍스트 포맷팅
│   └── interface.py       # Gradio 화면 구성
├── requirements.txt       # 패키지 목록
├── chroma_db/             # 벡터 DB 저장소
├── models/                # 모델 캐시
├── fashion_products.csv   # 크롤링 데이터 (크롤러 실행 시)
└── fashion_products.json
```

## 빠른 시작
```bash
# 수동 실행
pip install -r requirements.txt
mkdir -p chroma_db
mkdir -p models
#python crawl_products.py  # (선택) 초기 데이터 네이버 쇼핑 API 크롤링하여 생성
python app.py
```

## 네이버 API 설정
1. https://developers.naver.com 접속
2. 애플리케이션 등록
3. 검색 API 사용 신청
4. config.py에 Client ID/Secret 입력

## 사용 방법
1. http://localhost:7860 접속
2. 상품 이미지 업로드
3. "상품 탐지" 클릭
4. 탐지된 상품 확인
5. 채팅으로 추가 정보 질문

## 주요 기능
- DeepFashion2 기반 패션 아이템 탐지
- CLIP 이미지 특징 추출
- LangChain + ChromaDB RAG 시스템
- Llama 3.2 한국어 모델 대화
- 네이버 쇼핑 API 연동
- 검색 횟수 제한 관리 (10회)
- Gradio 웹 기반 사용자 인터페이스