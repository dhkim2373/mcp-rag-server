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
from psycopg_pool import ConnectionPool
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

# 전역 DB 커넥션 풀 및 AI 엔진 변수 선언
db_pool: ConnectionPool = None
local_llm = None
embeddings_engine = None

def init_ai_engines():
    """AI 엔진들을 현재 설정된 환경 변수 기준으로 초기화/재바인딩"""
    global local_llm, embeddings_engine
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    split_model = os.getenv("QUERY_SPLIT_MODEL", "exaone3.5:7.8b")
    embed_model = os.getenv("EMBEDDING_MODEL", "bge-m3")
    
    print(f"⚙️ [AI 엔진 바인딩] 의도분할 모델({split_model}) 및 임베딩 모델({embed_model}) 설정 적용 중...")
    local_llm = ChatOllama(base_url=ollama_url, model=split_model, temperature=0)
    embeddings_engine = OllamaEmbeddings(base_url=ollama_url, model=embed_model)

# ==========================================================
# ⏱️ [FastAPI 통합 Lifespan 및 커넥션 풀 초기화]
# ==========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    print("🚀 [3PL RAG 서버] 가동 시작 및 리소스 초기화 완료 (핫스왑 동기화 활성화)")
    
    conninfo = f"host={os.getenv('DB_HOST', 'localhost')} dbname={os.getenv('DB_NAME', '')} user={os.getenv('DB_USER', '')} password={os.getenv('DB_PASSWORD', '')} port={os.getenv('DB_PORT', 5432)}"
    db_pool = ConnectionPool(conninfo=conninfo, min_size=2, max_size=10)
    print("🗄️ [DB Pool 초기화] PostgreSQL 커넥션 풀 생성 완료")
    
    # 초기 AI 엔진 바인딩
    init_ai_engines()
    
    yield
    
    if db_pool:
        db_pool.close()
    print("🛑 [3PL RAG 서버] 안전 종료 중...")

app = FastAPI(title="3PL RAG Server", lifespan=lifespan)

# 🎯 하위 지식 라우터 패키지 병합 등록
app.include_router(knowledge.router)

# ==========================================================
# 🛡️ [CORS 미들웨어 추가]
# ==========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 구글 제미나이 API 클라이언트 선언
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GCP_API_KEY"))
client = genai.Client(api_key=GEMINI_API_KEY)

# 🎯 [리랭커]
print("⚙️ [Reranker 초기화] BAAI/bge-reranker-base 엔진 바인딩 중...")
reranker_engine = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

# 🎯 [의도 분할 프롬프트]
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

def format_stream_chunk(text: str, model_name: str) -> str:
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
# 🔄 [핫스왑 엔드포인트] 설정 변경 실시간 반영 API
# ==========================================================
@app.post("/reload-config")
async def reload_config_api():
    try:
        load_dotenv(override=True)
        init_ai_engines()
        print("🔄 [핫스왑 성공] 변경된 환경 설정 및 AI 모델이 실시간으로 반영되었습니다.")
        return {"status": "success", "message": "설정이 실시간으로 핫스왑되었습니다."}
    except Exception as e:
        print(f"❌ [핫스왑 실패] {e}")
        raise HTTPException(status_code=500, detail=str(e))

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


async def step_vector_search(sub_queries: list, selected_model: str, vector_search_limit: int, search_mode: str) -> dict:
    sub_query_chunks = {sub_q: [] for sub_q in sub_queries}
    seen_chunk_ids = set()
    
    def db_query_task():
        if not db_pool:
            raise RuntimeError("데이터베이스 커넥션 풀이 초기화되지 않았습니다.")
            
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                for sub_q in sub_queries:
                    q_vector = embeddings_engine.embed_query(sub_q.strip())
                    
                    if search_mode == "hybrid":
                        fts_keyword = " & ".join([w for w in sub_q.strip().split() if len(w) > 1])
                        if not fts_keyword:
                            fts_keyword = sub_q.strip()

                        search_query = f"""
                            SELECT reference_number, content, 
                                   (1 - (embedding <=> %s::vector)) AS similarity, 
                                   created_at, chunk_id,
                                   ts_rank(content_vector, to_tsquery('simple', %s)) AS fts_rank
                            FROM tb_document_chunk 
                            WHERE is_deleted = 0 
                              AND model_id = %s
                            ORDER BY (1 - (embedding <=> %s::vector)) * 0.7 + ts_rank(content_vector, to_tsquery('simple', %s)) * 0.3 DESC, chunk_id DESC 
                            LIMIT {vector_search_limit};
                        """
                        cur.execute(search_query, (q_vector, fts_keyword, selected_model, q_vector, fts_keyword))
                    else:
                        search_query = f"""
                            SELECT reference_number, content, 
                                   (1 - (embedding <=> %s::vector)) AS similarity, 
                                   created_at, chunk_id,
                                   0.0 AS fts_rank
                            FROM tb_document_chunk 
                            WHERE is_deleted = 0 
                              AND model_id = %s
                            ORDER BY embedding <=> %s::vector ASC, chunk_id DESC 
                            LIMIT {vector_search_limit};
                        """
                        cur.execute(search_query, (q_vector, selected_model, q_vector))

                    for res in cur.fetchall():
                        ref_num, content_text, sim_score, created_at, chunk_id, fts_score = res
                        if chunk_id in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(chunk_id)
                        
                        date_str = created_at.strftime('%Y-%m-%d') if isinstance(created_at, datetime) else (str(created_at)[:10] if created_at else datetime.now().strftime('%Y-%m-%d'))
                        
                        sub_query_chunks[sub_q].append({
                            "chunk_id": chunk_id,
                            "ref_num": ref_num,
                            "content": content_text,
                            "date_str": date_str,
                            "vector_similarity": float(sim_score),
                            "fts_rank": float(fts_score)
                        })

    try:
        await asyncio.to_thread(db_query_task)
    except Exception as db_err:
        print(f"❌ 데이터베이스 검색 실패: {db_err}")
        
    return sub_query_chunks


async def step_rerank_and_filter(sub_queries: list, sub_query_chunks: dict, use_rerank: bool, similarity_threshold: float, max_target_chunks: int):
    final_retrieved_contexts = []
    selected_chunk_objects = []
    MAX_TARGET_CHUNKS = max(max_target_chunks, len(sub_queries)*2)
    
    # 1. [1단계 필터] 컷라인을 통과한 데이터만 후보군으로 구성
    candidate_pool = []
    for chunks in sub_query_chunks.values():
        for chunk in chunks:
            if chunk.get("vector_similarity", 0.0) >= similarity_threshold:
                candidate_pool.append(chunk)

    try:
        if use_rerank and candidate_pool:
            # 2. [2단계 리랭크] 통과한 후보들만 리랭크 수행
            rerank_pairs = [[sub_queries[0], chunk["content"]] for chunk in candidate_pool]
            scores = await asyncio.to_thread(reranker_engine.predict, rerank_pairs)
            
            for idx, score in enumerate(scores):
                candidate_pool[idx]["rerank_score"] = float(score)
            
            candidate_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
            status_msg = "크로스리랭크 완료 (활성화)"
        else:
            candidate_pool.sort(key=lambda x: x["vector_similarity"], reverse=True)
            status_msg = "리랭크 생략 (비활성화 - 유사도순)"

        # 3. [최종 선정] 리랭크 결과 상위 항목만 선정
        selected_chunk_objects = candidate_pool[:MAX_TARGET_CHUNKS]
        
        for target in selected_chunk_objects:
            vec_sim = target.get("vector_similarity", 0.0)
            fmt_context = f"[{target['ref_num']}] (기록일자: {target['date_str']}) (벡터유사도: {vec_sim*100:.1f}%) {target['content']}"
            final_retrieved_contexts.append(fmt_context)
            
    except Exception as rerank_err:
        print(f"⚠️ [리랭커 처리 중 예외] 원인: {rerank_err}")
        selected_chunk_objects = candidate_pool[:MAX_TARGET_CHUNKS]
        for target in selected_chunk_objects:
            final_retrieved_contexts.append(f"[{target['ref_num']}] {target['content']}")
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

    all_configs = load_model_configs()
    model_conf = all_configs.get(selected_model, {})
    
    if isinstance(model_conf, dict):
        model_temperature = float(model_conf.get("temperature", 0.0))
    else:
        model_temperature = 0.0
        
    print(f"🎛️ [모델별 파라미터 적용] 모델: \"{selected_model}\" | Temperature: {model_temperature}")
    
    custom_model_prompt = get_model_system_instruction(
        selected_model, 
        "당신은 사내 통합 지식 보관소 데이터를 바탕으로 답변을 도출하는 팩트 검토 AI 보좌관입니다."
    )

    rag_system_instruction = (
        f"{user_context_instruction}\n"
        f"{custom_model_prompt}\n\n"
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
            config=types.GenerateContentConfig(
                temperature=model_temperature, 
                system_instruction=rag_system_instruction
            )
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
    use_query_splitting: bool,
    search_mode: str,
    vector_search_limit: int,
    similarity_threshold: float,
    use_rerank: bool,
    max_target_chunks: int
):
    print("\n" + "="*60)
    print(f"📥 [RAG 파이프라인 가동] 대상 모델: \"{selected_model}\" | 유저 질문: \"{user_last_text}\"")
    print(f"🎛️ [파라미터 상태] 검색모드: {search_mode.upper()} | 의도분할: {use_query_splitting} | 수집한도: {vector_search_limit} | 유사도컷라인: {similarity_threshold} | 리랭커: {use_rerank} | 통과상한: {max_target_chunks}")
    print("-"*60)

    all_configs = load_model_configs()
    model_conf = all_configs.get(selected_model, {})
    if isinstance(model_conf, dict):
        current_temp = float(model_conf.get("temperature", 0.0))
    else:
        current_temp = 0.0    

    # 🛠️ 로그 스타일 복구: 설정값 출력
    mode_display = "하이브리드 검색 (벡터 + 키워드 FTS)" if search_mode == "hybrid" else "벡터 검색 전용 (Dense Vector)"
    pipeline_status_msg = (
        f"> 🛠️ **[RAG 시스템 파이프라인 분석]**\n>\n"
        f"> * 📌 **대상 모델** : `{selected_model}` (검색 모드: `{mode_display}`)\n"
        f"> * 📌 **검색 파라미터** : 후보 수집 한도 `{vector_search_limit}건` | 유사도 컷라인 `{similarity_threshold}`\n"
        f"> * 📌 **리랭커/상한 설정** : 크로스 인코더 사용 `{use_rerank}` | 최종 전달 상한 `{max_target_chunks}개`\n>\n"
    )
    yield format_stream_chunk(pipeline_status_msg, selected_model)
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

    # 2단계 실행: 데이터베이스 검색 (Vector 또는 Hybrid)
    t_start = time.time()
    sub_query_chunks = await step_vector_search(sub_queries, selected_model, vector_search_limit, search_mode)
    dt_chunk = time.time() - t_start
    total_raw_chunks = sum(len(chunks) for chunks in sub_query_chunks.values())
    
    yield format_stream_chunk(f"> * 🟢 **청킹문서매칭 완료** : 총 {total_raw_chunks}건 후보군 매칭 ({dt_chunk*1000:.1f} ms)\n", selected_model)
    await asyncio.sleep(0.01)

    # 3단계 실행: 리랭크 및 필터링
    t_start = time.time()
    final_contexts, selected_chunks, status_msg = await step_rerank_and_filter(
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
    yield format_stream_chunk(f"> ✍️ *최종 지식 융합 및 답변 구성 중... (Temperature: `{current_temp}`)*\n\n---\n\n", selected_model)
    
    async for chunk_packet in step_generate_answer_stream(
        user_last_text, sub_queries, final_contexts, openai_messages, selected_model, base_url, user_context_instruction
    ):
        yield chunk_packet

    total_elapsed = time.time() - start_time
    print(f" ⏱️ 전체 파이프라인 총 연산 완료: {total_elapsed:.4f}초")
    print("="*60 + "\n")
    
    yield "data: [DONE]\n\n"

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
    
    user_context_instruction = f"현재 대화 중인 사용자의 이름은 '{user_name}'이고, 이메일은 '{user_email}'입니다. 기준 시간은 '{current_time_str}'입니다. "

    all_configs = load_model_configs()
    model_conf = all_configs.get(selected_model, {})
    
    if isinstance(model_conf, dict):
        search_mode = str(model_conf.get("search_mode", "vector"))
        vector_search_limit = int(model_conf.get("vector_search_limit", 10))
        similarity_threshold = float(model_conf.get("similarity_threshold", 0.35))
        use_rerank_val = str(model_conf.get("use_rerank", "True"))
        use_rerank = use_rerank_val.lower() in ("true", "1", "yes")
        max_target_chunks = int(model_conf.get("max_target_chunks", 5))
    else:
        search_mode = "vector"
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
            search_mode=search_mode,
            vector_search_limit=vector_search_limit,         
            similarity_threshold=similarity_threshold,       
            use_rerank=use_rerank,
            max_target_chunks=max_target_chunks
        ),
        media_type="text/event-stream"
    )

def run_tkinter_gui():
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