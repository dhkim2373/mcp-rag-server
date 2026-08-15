import os
import json
import re
import time
import math
import asyncio
import urllib.parse
from datetime import datetime
from contextlib import asynccontextmanager

import pytz
import psycopg
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

import tkinter as tk
from tkinter import messagebox
import threading
import sys

# 💡 .env 환경 변수 로드
load_dotenv()

# 🔥 지식 CRUD 라우터 불러오기
from routers import knowledge

# 🛠️ 모델 ID 및 프롬프트 관리 동적 모듈 불러오기
from model_config_manager import load_model_configs, get_model_system_instruction

# ==========================================================
# ⏱️ [FastAPI 통합 Lifespan]
# ==========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [3PL RAG 서버] 가동 시작 및 리소스 초기화 완료 (모델 설정 동기화 활성화)")
    yield
    print("🛑 [3PL RAG 서버] 안전 종료 중...")

app = FastAPI(title="3PL RAG Server", lifespan=lifespan)

# 🎯 하위 지식 라우터 패키지 병합 등록
app.include_router(knowledge.router)

# ==========================================================
# 🛡️ [CORS 미들웨어 추가]: UI 깜빡임 및 API 통신 오류 방지
# ==========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요에 따라 특정 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 구글 제미나이 API 클라이언트 선언 (환경변수 적용)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GCP_API_KEY"))
client = genai.Client(api_key=GEMINI_API_KEY)

# DB 연결 정보 (환경변수 기본값 적용)
MY_DATABASE_INFO = {
    "host": os.getenv("DB_HOST", "localhost"),
    "dbname": os.getenv("DB_NAME", ""),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": int(os.getenv("DB_PORT", 5432))
}

# 🎯 [동적 설정 연동] 의도분할 LLM 및 임베딩 엔진 모델명 환경변수 매핑
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QUERY_SPLIT_MODEL = os.getenv("QUERY_SPLIT_MODEL", "exaone3.5:7.8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

print(f"⚙️ [로컬 백엔드 최적화] 의도분할 모델({QUERY_SPLIT_MODEL}) 및 임베딩 모델({EMBEDDING_MODEL}) 로드 중...")
local_llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=QUERY_SPLIT_MODEL, temperature=0)
embeddings_engine = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)

# 🎯 [리랭커]: 가볍고 빠른 base 모델 유지
print("⚙️ [Reranker 초기화] BAAI/bge-reranker-base 엔진 바인딩 중...")
reranker_engine = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

# 🎯 [의도 분할 프롬프트]: 무조건 한글로 분할/정제되도록 강제
query_splitter_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 입력된 질문을 지식 검색에 용이한 독립된 검색어(질문)로 분할하는 엔진입니다.

[엄격 규칙]
1. 🚨 [필수] 입력 질문의 언어가 무엇이든, **분할된 모든 출력 결과는 반드시 '한국어(한글)'로 번역 및 작성**하세요.
2. 인사말, 설명, 생각 과정, 서론은 절대 출력하지 마세요.
3. 마크다운 기호나 JSON, 번호 매기기(1., 2.)를 쓰지 마세요.
4. 오직 분할된 검색어만 한 줄에 하나씩(줄바꿈으로만 구분) 즉시 출력하세요.
5. 질문 내용이 단일하거나 분할할 필요가 없다면, 내용을 한국어로 정제하여 딱 한 줄만 출력하세요."""),
    ("user", "분석할 사용자 질문:\n\n{user_query}")
])

# ==========================================================
# 🛠️ 스트리밍 전용 패킷 포맷터 헬퍼 함수
# ==========================================================
def format_stream_chunk(text: str, model_name: str) -> str:
    """OpenWebUI 표준 OpenAI 스트리밍 JSON 델타 포맷 데이터 변환"""
    chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None
        }]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

# ==========================================================
# 🛠️ 비동기 RAG 파이프라인 Generator (단계별 함수 분할)
# ==========================================================
async def step_split_query(user_last_text: str, use_query_splitting: bool) -> list:
    sub_queries = [user_last_text]
    if not use_query_splitting or len(user_last_text.strip()) < 20:
        return sub_queries
    
    try:
        chain = query_splitter_prompt | local_llm
        local_res = await chain.ainvoke({"user_query": user_last_text})
        local_res_text = str(local_res.content).strip()
        lines = [line.strip() for line in local_res_text.split('\n') if line.strip()]
        if lines:
            sub_queries = lines
    except Exception as e:
        print(f"⚠️ 질문 분할 장애 ({e})")
    
    return sub_queries


async def step_vector_search(sub_queries: list, selected_model: str, vector_search_limit: int) -> dict:
    sub_query_chunks = {sub_q: [] for sub_q in sub_queries}
    seen_chunk_ids = set()
    
    def db_query_task():
        conn = None
        try:
            conn = psycopg.connect(**MY_DATABASE_INFO)
            with conn.cursor() as cur:
                for sub_q in sub_queries:
                    q_vector = embeddings_engine.embed_query(sub_q.strip())
                    search_query = f"""
                        SELECT reference_number, content, (1 - (embedding <=> %s::vector)) AS similarity, created_at, chunk_id
                        FROM tb_document_chunk 
                        WHERE is_deleted = 0 
                          AND model_id = %s
                        ORDER BY embedding <=> %s::vector ASC, chunk_id DESC 
                        LIMIT {vector_search_limit};
                    """
                    cur.execute(search_query, (q_vector, selected_model, q_vector))
                    for res in cur.fetchall():
                        ref_num, content_text, sim_score, created_at, chunk_id = res
                        if chunk_id in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(chunk_id)
                        
                        date_str = created_at.strftime('%Y-%m-%d') if isinstance(created_at, datetime) else (str(created_at)[:10] if created_at else datetime.now().strftime('%Y-%m-%d'))
                        
                        sub_query_chunks[sub_q].append({
                            "chunk_id": chunk_id,
                            "ref_num": ref_num,
                            "content": content_text,
                            "date_str": date_str,
                            "vector_similarity": float(sim_score)
                        })
        except Exception as db_err:
            print(f"❌ 벡터 DB 탐색 실패: {db_err}")
        finally:
            if conn:
                conn.close()

    await asyncio.to_thread(db_query_task)
    return sub_query_chunks


def step_rerank_and_filter(sub_queries: list, sub_query_chunks: dict, use_rerank: bool, similarity_threshold: float, max_target_chunks: int):
    final_retrieved_contexts = []
    selected_chunk_objects = []
    # 💡 설정값(max_target_chunks) 반영
    MAX_TARGET_CHUNKS = max(max_target_chunks, len(sub_queries))
    
    try:
        if use_rerank:
            for sub_q, chunks in sub_query_chunks.items():
                if not chunks:
                    continue
                rerank_pairs = [[sub_q, chunk["content"]] for chunk in chunks]
                scores = reranker_engine.predict(rerank_pairs)
                for idx, raw_score in enumerate(scores):
                    chunks[idx]["rerank_score"] = float(raw_score)
                chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
                
            merged_chunks = []
            max_pool_depth = max(len(chunks) for chunks in sub_query_chunks.values()) if sub_query_chunks else 0
            for depth in range(max_pool_depth):
                for sub_q in sub_queries:
                    pool = sub_query_chunks[sub_q]
                    if depth < len(pool):
                        merged_chunks.append(pool[depth])
                        if len(merged_chunks) >= MAX_TARGET_CHUNKS:
                            break
                if len(merged_chunks) >= MAX_TARGET_CHUNKS:
                    break
            target_pool = merged_chunks
            status_msg = "크로스리랭크 완료 (활성화)"
        else:
            flat_chunks = []
            for chunks in sub_query_chunks.values():
                flat_chunks.extend(chunks)
            flat_chunks.sort(key=lambda x: x["vector_similarity"], reverse=True)
            target_pool = flat_chunks[:MAX_TARGET_CHUNKS]
            status_msg = "리랭크 생략 (비활성화 - 유사도순)"

        for target in target_pool:
            vec_sim = target.get("vector_similarity", 0.0)
            if vec_sim < similarity_threshold:
                continue
            fmt_context = f"[{target['ref_num']}] (기록일자: {target['date_str']}) (벡터유사도: {vec_sim*100:.1f}%) {target['content']}"
            final_retrieved_contexts.append(fmt_context)
            selected_chunk_objects.append(target)
                
    except Exception as rerank_err:
        print(f"⚠️ [리랭커 처리 중 예외] 원인: {rerank_err}")
        for chunks in sub_query_chunks.values():
            for target in chunks[:MAX_TARGET_CHUNKS]:
                fmt_context = f"[{target['ref_num']}] (기록일자: {target['date_str']}) {target['content']}"
                final_retrieved_contexts.append(fmt_context)
                selected_chunk_objects.append(target)
        status_msg = "리랭커 폴백 실행"

    return final_retrieved_contexts, selected_chunk_objects, status_msg


async def step_generate_answer_stream(
    user_last_text: str, 
    sub_queries: list, 
    final_retrieved_contexts: list, 
    openai_messages: list, 
    selected_model: str, 
    base_url: str, 
    user_context_instruction: str
):
    joined_context = "\n".join(final_retrieved_contexts) if final_retrieved_contexts else "관련 지식 데이터 없음."
    
    custom_model_prompt = get_model_system_instruction(
        selected_model, 
        "당신은 사내 통합 지식 보관소 데이터를 바탕으로 답변을 도출하는 팩트 검토 AI 보좌관입니다."
    )

    rag_system_instruction = (
        f"{user_context_instruction}\n"
        f"{custom_model_prompt}\n\n"
        "아래 제공되는 [참조 지식 컨텍스트]에 명확히 명시된 팩트만을 기반으로 자연스럽고 읽기 쉽게 답변해야 합니다.\n\n"
        "[답변 철칙]\n"
        "1. 본인의 사전 지식을 활용하여 절대 그럴싸한 거짓말이나 없는 문장을 지어내지 마세요.\n"
        "2. 🚨 [최신성 우선 원칙] 제공된 컨텍스트 내에서 서로 상충되거나 유사한 내용의 자료가 발견될 경우, 본문에 적힌 [기록일자] 정보를 확인하여 가장 최근(최신)에 기록된 데이터를 우선 참조하세요.\n"
        "3. 💡 일반 지식/코드 생성/HTML 작성/포맷팅 요청인 경우:\n"
        "    - [참조 지식 컨텍스트]에 관련 내용이 없더라도, 당신의 기본 지능을 활용하여 요청한 포맷(HTML, 서식 등) 및 질문에 완벽하게 답변하세요.\n\n"        
        "4. 🚨 [하이퍼링크 출처 표기 규칙] 본문 문장 뒤에 출처를 계속 중복해서 붙이지 마세요. 답변이 모두 끝난 맨 마지막 줄에 '---' 구분선을 그은 뒤, 답변 생성에 참고한 모든 출처 정보(태그)를 중복 없이 단 한 번만 리스트 형태로 모아서 명시하세요.\n"
        f"반드시 아래 제공된 [마크다운 링크 형식]을 엄격하게 준수하여 출처를 출력해야 합니다.\n\n"
        "[마크다운 링크 형식]\n"
        f"- 🔗 [[문서태그명]]({base_url}/view/document/문서태그명?model_id={selected_model}) (기록일자: YYYY-MM-DD)\n\n"
        "출처 표기 양식 예시:\n"
        "---\n"
        "📌 [참조 출처 안내]\n"
        f"- 🔗 [[USER_MEM_95]]({base_url}/view/document/USER_MEM_95?model_id={selected_model}) (기록일자: 2026-07-28)\n\n"
        "5. 제공된 컨텍스트의 내용만으로 질문에 답할 수 없다면, 근거가 부족하다고 솔직하게 인정하고 답변을 패스하세요.\n\n"
        f"[참조 지식 컨텍스트]\n{joined_context}"
    )

    gemini_contents = []
    history_messages = openai_messages[-6:-1]
    for msg in history_messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        
    formatted_sub_queries = "\n".join([f"- {q}" for q in sub_queries])
    enhanced_user_payload = (
        f"{user_last_text}\n\n"
        f"[시스템 세부 분석 의도 리스트]\n"
        f"{formatted_sub_queries}"
    )
    gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=enhanced_user_payload)]))

    try:
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=gemini_contents,
            config=types.GenerateContentConfig(temperature=0.0, system_instruction=rag_system_instruction)
        )
        for chunk in response_stream:
            if chunk.text:
                yield format_stream_chunk(chunk.text, selected_model)
    except Exception as gemini_err:
        yield format_stream_chunk(f"\n⚠️ 최종 답변 생성 장애 발생 ({gemini_err})", selected_model)


async def handle_knowledge_retrieval_stream(
    user_last_text: str, 
    user_context_instruction: str, 
    openai_messages: list, 
    selected_model: str, 
    base_url: str, 
    start_time: float,
    use_query_splitting: bool = True,
    vector_search_limit: int = 10,
    similarity_threshold: float = 0.35,
    use_rerank: bool = True,
    max_target_chunks: int = 5
):
    print("\n" + "="*60)
    print(f"📥 [RAG 파이프라인 가동] 대상 모델: \"{selected_model}\" | 유저 질문: \"{user_last_text}\"")
    print(f"🎛️ [옵션 상태] 의도분할: {use_query_splitting} | 수집한도: {vector_search_limit} | 유사도컷라인: {similarity_threshold} | 리랭커: {use_rerank} | 통과상한: {max_target_chunks}")
    print("-"*60)

    yield format_stream_chunk("> 🛠️ **[RAG 시스템 파이프라인 가동 분석]**\n>\n", selected_model)
    await asyncio.sleep(0.01)

    # 1단계 실행: 의도 분할
    t_start = time.time()
    sub_queries = await step_split_query(user_last_text, use_query_splitting)
    dt_split = time.time() - t_start
    
    yield format_stream_chunk(f"> * 🟢 **질문의도분할 완료** : {len(sub_queries)}건 분리 ({dt_split*1000:.1f} ms)\n", selected_model)
    log_intent = "> \t💡 *[분할된 상세 의도 목록]*\n"
    for idx, sub_q in enumerate(sub_queries, 1):
        log_intent += f"> \t  - 📌 의도 {idx}: {sub_q}\n"
    yield format_stream_chunk(log_intent + ">\n", selected_model)
    await asyncio.sleep(0.01)

    # 2단계 실행: 벡터 DB 검색
    t_start = time.time()
    sub_query_chunks = await step_vector_search(sub_queries, selected_model, vector_search_limit)
    dt_chunk = time.time() - t_start
    total_raw_chunks = sum(len(chunks) for chunks in sub_query_chunks.values())
    
    yield format_stream_chunk(f"> * 🟢 **청킹문서매칭 완료** : 총 {total_raw_chunks}건 후보군 매칭 ({dt_chunk*1000:.1f} ms)\n", selected_model)
    await asyncio.sleep(0.01)

    # 3단계 실행: 리랭크 및 필터링
    t_start = time.time()
    final_contexts, selected_chunks, status_msg = step_rerank_and_filter(
        sub_queries, sub_query_chunks, use_rerank, similarity_threshold, max_target_chunks
    )
    dt_rerank = time.time() - t_start
    
    yield format_stream_chunk(f"> * 🟢 **{status_msg}** : 상위 {len(selected_chunks)}개 청크 엄선 ({dt_rerank*1000:.1f} ms)\n", selected_model)

    if selected_chunks:
        log_chunk_detail = "> \t📑 *[최종 엄선된 참조 청크 목록]*\n"
        for chunk in selected_chunks:
            ref = chunk.get("ref_num", f"ID-{chunk.get('chunk_id')}")
            vec_score_percent = chunk.get("vector_similarity", 0.0) * 100
            summary = chunk.get("content", "").replace("\n", " ").strip()[:40]
            log_chunk_detail += f"> \t  - 🔹 **[{ref}]** 유사도: `{vec_score_percent:.1f}%` | {summary}...\n"
    else:
        log_chunk_detail = "> \t⚠️ *[안내] 조건에 부합하는 참조 청크가 발견되지 않았습니다.*\n"
        
    yield format_stream_chunk(log_chunk_detail + ">\n", selected_model)
    await asyncio.sleep(0.01)

    # 4단계 실행: 제미나이 답변 생성 스트리밍
    yield format_stream_chunk("> ✍️ *최종 지식 융합 및 답변 구성 중...*\n\n---\n\n", selected_model)
    
    async for chunk_packet in step_generate_answer_stream(
        user_last_text, sub_queries, final_contexts, openai_messages, selected_model, base_url, user_context_instruction
    ):
        yield chunk_packet

    total_elapsed = time.time() - start_time
    print(f" ⏱️ 전체 파이프라인 총 연산 완료: {total_elapsed:.4f}초")
    print("="*60 + "\n")
    
    yield "data: [DONE]\n\n"

# 🎯 OpenWebUI 선택 모델 목록 반환
@app.get("/v1/models")
def list_models():
    configs = load_model_configs()
    model_list = [
        {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "redbombz"}
        for model_id in configs.keys()
    ]
    return {
        "object": "list",
        "data": model_list
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    raw_user_name = request.headers.get("x-openwebui-user-name", "Unknown_User")
    user_name = urllib.parse.unquote(raw_user_name)
    user_id = request.headers.get("x-openwebui-user-id", "Unknown_ID")
    user_email = request.headers.get("x-openwebui-user-email", "unknown@company.com")

    base_url = os.getenv("RAG_BASE_URL", "https://aimeow.duckdns.org:8000")
    start_time = time.time()            
    kst = pytz.timezone('Asia/Seoul')
    current_time_str = datetime.now(kst).strftime("%Y년 %m월 %d일 %A %H시 %M분 %S초")

    body = await request.json()
    openai_messages = body.get("messages", [])
    selected_model = body.get("model", "3PL지식저장소") 

    user_last_text = ""
    if openai_messages:
        last_msg = openai_messages[-1].get("content", "")
        user_last_text = "".join([item.get("text", "") for item in last_msg if item.get("type") == "text"]) if isinstance(last_msg, list) else str(last_msg)

    if any(k in user_last_text for k in ["Generate a concise", "summarizing the chat history"]):
        title_response = client.models.generate_content(model="gemini-2.5-flash", contents=user_last_text)
        return {
            "id": f"chatcmpl-{int(time.time())}", 
            "object": "chat.completion", 
            "created": int(time.time()), 
            "model": selected_model, 
            "choices": [{"index": 0, "message": {"role": "assistant", "content": title_response.text if title_response.text else "추출 실패"}, "finish_reason": "stop"}]
        }
    
    # 사용자 컨텍스트 문구 생성
    user_context_instruction = f"현재 대화 중인 사용자의 이름은 '{user_name}'이고, 이메일은 '{user_email}'입니다. 기준 시간은 '{current_time_str}'입니다. "

    # 💡 [동적 설정 반영] .env 환경 변수에서 RAG 제어 파라미터 로드
    vector_search_limit = int(os.getenv("VECTOR_SEARCH_LIMIT", 10))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))
    use_rerank = os.getenv("USE_RERANK", "True").lower() in ("true", "1", "yes")
    max_target_chunks = int(os.getenv("MAX_TARGET_CHUNKS", 5))

    return StreamingResponse(
        handle_knowledge_retrieval_stream(
            user_last_text=user_last_text,
            user_context_instruction=user_context_instruction,
            openai_messages=openai_messages,
            selected_model=selected_model,
            base_url=base_url,
            start_time=start_time,
            use_query_splitting=True,      
            vector_search_limit=vector_search_limit,         
            similarity_threshold=similarity_threshold,      
            use_rerank=use_rerank,
            max_target_chunks=max_target_chunks
        ),
        media_type="text/event-stream"
    )

def run_tkinter_gui():
    """Tkinter 설정 UI를 별도 스레드에서 실행"""
    from config_ui_app import AdvancedConfigApp
    
    root = tk.Tk()
    app = AdvancedConfigApp(root)
    
    def on_closing():
        if messagebox.askokcancel("종료", "RAG 설정 관리 창을 닫으시겠습니까?\n(백엔드 서버도 함께 종료됩니다.)"):
            root.destroy()
            os._exit(0) 

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    ui_thread = threading.Thread(target=run_tkinter_gui, daemon=True)
    ui_thread.start()
    print("🖥️ [설정 UI] 데스크톱 관리 창 가동 완료")

    print("🚀 [3PL RAG 서버] 지식 검색 백엔드 가동 시작 (포트: 8000)")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )