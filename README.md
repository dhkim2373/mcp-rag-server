지금까지 구축하신 **3PL RAG Server** 프로젝트의 구조와 기능(의도 분할, 벡터 검색, 리랭커, 커넥션 풀, 핫스왑, Tkinter 설정 UI)을 완벽하게 담아낸 상세하고 전문적인 `README.md` 문서입니다.

---

```markdown
# 🚀 3PL RAG Server (Enterprise Retrieval-Augmented Generation System)

사내 통합 지식 보관소 및 3PL(3자 물류) 도메인에 특화된 고성능 **RAG(Retrieval-Augmented Generation) 백엔드 서버**입니다. 로컬 Ollama 모델과 외부 LLM(Gemini)을 하이브리드 형태로 연동하고, OpenWebUI 표준 규격과 완벽하게 호환됩니다.

---

## 🌟 주요 특징 및 아키텍처 (Key Features)

1. **지능형 질문 의도 분할 (Query Splitting)**
   * 복합적이거나 긴 사용자의 질문을 Ollama LLM(예: EXAONE 3.5)을 통해 독립된 하위 검색어로 정제 및 분할하여 검색 정확도를 극대화합니다.
2. **고성능 벡터 검색 및 커넥션 풀 (PostgreSQL + pgvector)**
   * `psycopg_pool` 기반의 비동기 커넥션 풀을 탑재하여 다중 요청 환경에서도 부하 없이 초고속 벡터 유사도 검색을 수행합니다.
3. **크로스 인코더 리랭커 (Cross-Encoder Reranking)**
   * `bge-reranker-base` 모델을 활용하여 1차 검색된 청크들의 연관성을 정밀 재채점하며, CPU 블로킹 방지를 위해 `asyncio.to_thread`로 비동기 스레드 분리 처리가 적용되어 있습니다.
4. **실시간 핫스왑(Hot-Reload) 설정 관리 GUI**
   * 데스크톱 환경에서 터미널 없이 `.env` 환경 변수와 모델별 프롬프트를 직관적으로 관리할 수 있는 **Tkinter 설정 앱**을 제공하며, 저장 즉시 백엔드 서버에 실시간 반영(`/reload-config`)됩니다.
5. **OpenWebUI 완벽 호환 스트리밍**
   * OpenAI 표준 스트리밍(`chat.completion.chunk`) 포맷을 지원하여 OpenWebUI 인터페이스와 매끄럽게 연동됩니다.

---

## 🛠️ 시스템 파이프라인 흐름

```text
[사용자 질문 (OpenWebUI)] 
       │
       ▼
 1단계: 질문 의도 분할 (Ollama LLM)
       │
       ▼
 2단계: 벡터 DB 유사도 검색 (PostgreSQL + pgvector + 커넥션 풀)
       │
       ▼
 3단계: 크로스 인코더 리랭크 및 필터링 (Cross-Encoder / 비동기 처리)
       │
       ▼
 4단계: 최신성 검토 및 최종 답안 스트리밍 생성 (Google Gemini 2.5 Flash)

```

---

## 📂 프로젝트 구조 (Directory Structure)

```text
📦 3pl-rag-server
 ┣ 📂 routers
 ┃  ┗ 📜 knowledge.py         # 지식 CRUD 및 관리 라우터
 ┣ 📂 templates              # 웹 UI 템플릿 (HTML)
 ┣ 📜 main.py                # FastAPI 메인 서버 및 RAG 파이프라인 코어
 ┣ 📜 config_ui_app.py       # Tkinter 기반 종합 설정 관리 데스크톱 UI
 ┣ 📜 model_config_manager.py # 모델 ID 및 시스템 프롬프트 관리 모듈
 ┣ 📜 embedding_batch.py     # 문서 배치 임베딩 생성 스크립트
 ┣ 📜 vector_3d_visualizer.py# 벡터 공간 3D 시각화 도구
 ┣ 📜 requirements.txt       # 파이썬 패키지 의존성 목록
 ┗ 📜 .env                   # 환경 변수 설정 파일 (자동 생성/관리)

```

---

## ⚙️ 설치 및 실행 방법 (Installation & Usage)

### 1. 사전 요구 사항

* Python 3.10 이상
* PostgreSQL (pgvector 확장 기능 설치 필수)
* Ollama 서버 (로컬 또는 원격 GPU 서버)

### 2. 패키지 의존성 설치

```bash
pip install -r requirements.txt

```

### 3. 환경 변수 설정 (`.env`)

프로젝트 루트 디렉토리에 `.env` 파일을 생성하거나 Tkinter 설정 GUI를 통해 아래 항목들을 설정합니다.

```ini
RAG_BASE_URL=[https://your-domain.duckdns.org:8000](https://your-domain.duckdns.org:8000)
GEMINI_API_KEY=your_google_gemini_api_key
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
QUERY_SPLIT_MODEL=exaone3.5:7.8b
VECTOR_SEARCH_LIMIT=10
SIMILARITY_THRESHOLD=0.35
USE_RERANK=True
MAX_TARGET_CHUNKS=5

DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password

```

### 4. 서버 및 설정 GUI 동시 실행

`main.py`를 실행하면 FastAPI 백엔드 서버와 데스크톱 설정 GUI가 동시에 구동됩니다.

```bash
python main.py

```

---

## 🎛️ 데스크톱 설정 관리자 (`config_ui_app.py`)

서버가 실행되면 관리용 Tkinter GUI 창이 나타납니다.

* **[기본 환경 및 RAG 제어]**: URL, API 키, 임베딩 모델, 수집 한도(Limit), 유사도 컷라인, 리랭커 사용 여부 등을 탭 형식으로 손쉽게 조절할 수 있습니다.
* **[데이터베이스(DB)]**: PostgreSQL 접속 정보를 안전하게 관리합니다.
* **[모델 & 프롬프트]**: OpenWebUI에서 선택할 수 있는 모델 ID별 시스템 프롬프트(지침)를 동적으로 추가·수정할 수 있습니다.
* **[실시간 반영]**: `💾 전체 설정 저장 및 반영` 버튼을 누르면 `.env`가 갱신됨과 동시에 백엔드에 핫스왑(`POST /reload-config`) 신호가 전송되어 서버 재시작 없이 즉시 적용됩니다.

---

## 📜 라이선스 (License)

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
