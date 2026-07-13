# 변경 이력 (CHANGELOG)

이 프로젝트의 주요 변경 사항을 기록합니다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 따르며,
버전은 [유의적 버전](https://semver.org/lang/ko/)을 사용합니다.

## [2.0.0] - 2026-07-13

`ver1.0`(SSAFY ai-shopping-assistant 실습)을 리팩토링하고 AI 기능을 고도화한 버전.

### Changed — 구조 리팩토링
- **1,035줄짜리 `app.py`를 역할별 패키지로 분리** (`refactor` 커밋)
  - `core/` — `ModelRegistry`(모델 보관), `SessionState`(세션 상태)로 흩어진 전역변수 정리
  - `services/` — detection, vision, products, vectorstore, query, llm, assistant
  - `ui/` — formatters, interface
  - 의존 방향을 `ui → services → core` 한 방향으로 통일
  - `app.py`는 24줄 엔트리포인트로 축소 (모델 로드 → 초기화 → 서버 실행)

### Added — AI 기능 고도화
- **세그멘테이션 마스크 기반 색상 추출** (고도화 ①, `feat` 커밋)
  - 기존: 바운딩 박스 중앙 60% 영역에서 k-means → 배경·피부 픽셀 혼입
  - 개선: YOLO seg 마스크(`masks.xy`)로 옷 픽셀만 골라 색상 추출, 경계는 erode로 정리
  - 마스크가 없으면 기존 bbox 방식으로 자동 폴백
  - 빨간 배경 합성 이미지로 마스크 정렬 검증 (배경색 혼입 0%)
- **CLIP 크로스모달 텍스트→이미지 검색** (고도화 ②)
  - "빨간 원피스" 같은 한국어 설명으로 상품 이미지를 직접 검색
  - 영어 CLIP의 한국어 취약점을 색상·아이템 번역 매핑(번역 브릿지)으로 우회
  - 텍스트와 이미지가 동일 모델·동일 벡터공간이라 정렬이 보장됨 (새 모델 의존성 없음)
  - 매핑되는 패션 어휘가 없으면 기존 텍스트 벡터 검색으로 폴백
  - 흰/빨강 상품 구분 검색으로 정확도 검증 완료
- **RAG 파이프라인 및 프롬프트 개선** (고도화 ③)
  - **R(검색)**: `similarity_search_with_score`로 관련도(거리) 기반 필터링 추가 —
    무관한 상품(예: 셔츠 검색 시 이어폰)이 LLM 컨텍스트를 오염시키지 않도록 제외.
    ko-sroberta 임베딩 분포에 맞춰 임계값 0.75 설정
  - **A(증강)**: LLM 전용 번호형 컨텍스트 `build_context()` 추가 —
    사용자용 목록과 분리해 URL 등 불필요 정보를 빼고 압축, 토큰 절약 + 번호 인용 유도
  - **G(생성)**: 시스템 프롬프트를 규칙 기반으로 재작성(근거 제한·번호 인용·빈 결과 처리·형식),
    사실 기반 답변을 위해 temperature 0.7 → 0.3으로 하향
  - `generate_response_with_context`가 상품 리스트를 직접 받아 컨텍스트 구성을 담당(관심사 분리)
  - GPU에서 개선 전/후 비교 검증: 답변 내 URL 누출 제거, 상품 번호 인용, 응답 간결화 확인

### Fixed — 버그 수정
- `init_image_collection()` 호출 누락으로 CLIP 이미지 검색이 동작하지 않던 문제 수정 (앱 시작 시 초기화)
- 빈 메시지 전송 시 반환값 개수 불일치로 발생하던 Gradio 오류 수정
- Chroma 메타데이터 → 상품 dict 변환 로직 3곳 중복을 `_meta_to_product()`로 통합
- `generate_response_with_context()`의 미사용 `user_input` 파라미터 제거
- deprecated `langchain.embeddings` / `langchain.vectorstores` import를 `langchain_community`로 교체

### 다음 계획
- 정규식 쿼리 파싱을 LLM 의도 추출로 전환
- 대화 요약 기반 장기 메모리
- 재순위화(가격+유사도 결합 스코어) 및 few-shot 예시 프롬프트

## [1.0.0]

- SSAFY ai-shopping-assistant 실습 기반 최초 버전
- DeepFashion2 패션 아이템 탐지, CLIP 이미지 특징 추출, LangChain + ChromaDB RAG,
  Llama 3.2 한국어 모델 대화, 네이버 쇼핑 API 연동, Gradio 웹 UI
