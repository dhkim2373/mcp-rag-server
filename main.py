import os
import json
import re
import time
import asyncio
import urllib.parse
from datetime import datetime
from contextlib import asynccontextmanager
import pytz
import psycopg
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

# 🔥 지식 CRUD 라우터 및 MCP 모듈 불러오기
from routers import knowledge
from routers.mcp_router import mcp_server_lifespan

# ==========================================================
# ⏱️ [FastAPI 통합 Lifespan]: MCP 백그라운드 서버 연결
# ==========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 원격 MCP SSE 서버 백그라운드 가동
    async with mcp_server_lifespan():
        yield

app = FastAPI(title="3PL RAG & MCP Integrated Server", lifespan=lifespan)

# 🎯 하위 라우터 패키지 병합 등록
app.include_router(knowledge.router)

# 구글 제미나이 API 클라이언트 선언
client = genai.Client(api_key="AQ.Ab8RN6IHlYpkWtVxLrlwVzPx8D5wwl0XjymxLdX90RKvFauudA")

MY_DATABASE_INFO = {
    "host": "localhost", # Nginx 우회 사설 IP 고정
    "dbname": "redbombz",
    "user": "redbombz",
    "password": "a11223344*",
    "port": 5432
}

print("⚙️ [로컬 백엔드 최적화] EXAONE 3.5 및 bge-m3 임베딩 엔진 로드 중...")
local_llm = ChatOllama(base_url="http://localhost:11434", model="exaone3.5:7.8b", temperature=0)
embeddings_engine = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3")

# 🎯 [리랭커]: 가볍고 빠른 base 모델 유지
print("⚙️ [Reranker 초기화] BAAI/bge-reranker-base 엔진 바인딩 중...")
reranker_engine = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

query_splitter_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 입력된 문장에 여러 개의 요구사항이 섞여 있을 경우, 이를 지식 검색에 용이한 독립된 검색어(질문)로 분할하는 엔진입니다.

[엄격 규칙]
1. 인사말, 설명, 생각 과정, 서론은 절대 출력하지 마세요.
2. 마크다운 기호나 JSON, 번호 매기기(1., 2.)를 쓰지 마세요.
3. 오직 분할된 검색어만 한 줄에 하나씩(줄바꿈으로만 구분) 즉시 출력하세요.
4. 질문 내용이 단일하거나 분할할 필요가 없다면, 입력된 문장 그대로 딱 한 줄만 출력하세요."""),
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
# 🛠️ 비동기 RAG 파이프라인 Generator
# ==========================================================
async def handle_knowledge_retrieval_stream(user_last_text: str, user_context_instruction: str, openai_messages: list, selected_model: str, base_url: str, start_time: float):
    print("\n" + "="*60)
    print(f"📥 [RAG 파이프라인 가동] 대상 모델: \"{selected_model}\" | 유저 질문: \"{user_last_text}\"")
    print("-"*60)

    yield format_stream_chunk("> 🛠️ **[RAG 시스템 파이프라인 가동 분석]**\n>\n", selected_model)
    await asyncio.sleep(0.01)

    # 1단계: 질문 의도 분할 구간
    t_start = time.time()
    sub_queries = [user_last_text]
    split_count = 1
    
    if len(user_last_text.strip()) < 20:
        dt_split = time.time() - t_start
        print(f" 1. 질문의도분할 패스 : 단문 커트라인 적용 | 소요시간: {dt_split:.4f}초 ({dt_split*1000:.1f} ms)")
        yield format_stream_chunk(f"> * 🟢 **질문의도분할** : 패스 (단문 예외 처리 적용 / {dt_split*1000:.1f} ms)\n", selected_model)
    else:
        try:
            chain = query_splitter_prompt | local_llm
            local_res = await chain.ainvoke({"user_query": user_last_text})
            local_res_text = local_res.content.strip()
            
            lines = [line.strip() for line in local_res_text.split('\n') if line.strip()]
            if lines:
                sub_queries = lines
                split_count = len(sub_queries)
        except Exception as e:
            print(f"⚠️ 질문 분할 장애 ({e})")
        
        dt_split = time.time() - t_start
        print(f" 1. 질문의도분할 완료 : {split_count}건 분리 | 소요시간: {dt_split:.4f}초 ({dt_split*1000:.1f} ms)")
        
        log_1 = f"> * 🟢 **질문의도분할 완료** : {split_count}건 분리 ({dt_split*1000:.1f} ms)\n"
        yield format_stream_chunk(log_1, selected_model)

    print(f"    💡 [분할된 상세 의도 목록]")
    log_intent = "> \t💡 *[분할된 상세 의도 목록]*\n"
    for idx, sub_q in enumerate(sub_queries, 1):
        print(f"       📌 의도 {idx}: {sub_q}")
        log_intent += f"> \t  - 📌 의도 {idx}: {sub_q}\n"
    yield format_stream_chunk(log_intent + ">\n", selected_model)
    print("-"*60)
    await asyncio.sleep(0.01)

    # 2단계: 벡터 DB 검색 및 청킹 수집 구간
    t_start = time.time()
    sub_query_chunks = {sub_q: [] for sub_q in sub_queries}
    seen_chunk_ids = set()
    
    conn = None
    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        cur = conn.cursor()
        for sub_q in sub_queries:
            q_vector = embeddings_engine.embed_query(sub_q.strip())
            
            search_query = """
                SELECT reference_number, content, (1 - (embedding <=> %s::vector)) AS similarity, created_at, chunk_id
                FROM tb_document_chunk 
                WHERE is_deleted = 0 
                  AND model_id = %s
                ORDER BY embedding <=> %s::vector ASC, chunk_id DESC 
                LIMIT 10;
            """
            cur.execute(search_query, (q_vector, selected_model, q_vector))
            for res in cur.fetchall():
                ref_num, content_text, sim_score, created_at, chunk_id = res
                
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                
                if isinstance(created_at, datetime):
                    date_str = created_at.strftime('%Y-%m-%d')
                elif created_at:
                    date_str = str(created_at)[:10]
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                
                sub_query_chunks[sub_q].append({
                    "chunk_id": chunk_id,
                    "ref_num": ref_num,
                    "content": content_text,
                    "date_str": date_str
                })
        cur.close()
    except Exception as db_err:
        print(f"❌ 벡터 DB 탐색 실패: {db_err}")
    finally:
        if conn: conn.close()
        
    dt_chunk = time.time() - t_start
    total_raw_chunks = sum(len(chunks) for chunks in sub_query_chunks.values())
    print(f" 2. 청킹문서매칭 완료 : {total_raw_chunks}건 수집 (모델: {selected_model}) | 소요시간: {dt_chunk:.4f}초 ({dt_chunk*1000:.1f} ms)")
    yield format_stream_chunk(f"> * 🟢 **청킹문서매칭 완료** : [{selected_model}] 대상 총 {total_raw_chunks}건 후보군 격리 매칭 ({dt_chunk*1000:.1f} ms)\n", selected_model)
    await asyncio.sleep(0.01)

    # 3단계: 로컬 크로스 인코더 리랭크 연산 구간
    t_start = time.time()
    final_retrieved_contexts = []
    
    try:
        for sub_q, chunks in sub_query_chunks.items():
            if not chunks:
                continue
            
            rerank_pairs = [[sub_q, chunk["content"]] for chunk in chunks]
            scores = reranker_engine.predict(rerank_pairs)
            
            for idx, score in enumerate(scores):
                chunks[idx]["rerank_score"] = float(score)
                
            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            
        merged_chunks = []
        max_pool_depth = max(len(chunks) for chunks in sub_query_chunks.values()) if sub_query_chunks else 0
        
        for depth in range(max_pool_depth):
            for sub_q in sub_queries:
                pool = sub_query_chunks[sub_q]
                if depth < len(pool):
                    merged_chunks.append(pool[depth])
                    if len(merged_chunks) >= 5:
                        break
            if len(merged_chunks) >= 5:
                break
                
        for target in merged_chunks:
            fmt_context = f"[{target['ref_num']}] (기록일자: {target['date_str']}) (리랭킹점수: {target['rerank_score']:.4f}) {target['content']}"
            final_retrieved_contexts.append(fmt_context)
                
    except Exception as rerank_err:
        print(f"⚠️ [리랭커 라운드로빈 크래시 대응 폴백 전송] 원인: {rerank_err}")
        fallback_list = []
        for chunks in sub_query_chunks.values():
            fallback_list.extend(chunks)
        for target in fallback_list[:5]:
            fmt_context = f"[{target['ref_num']}] (기록일자: {target['date_str']}) (폴백백업) {target['content']}"
            final_retrieved_contexts.append(fmt_context)
                
    dt_rerank = time.time() - t_start
    print(f" 3. 크로스리랭크 완료 : 라운드로빈 융합 정렬 완료 | 소요시간: {dt_rerank:.4f}초 ({dt_rerank*1000:.1f} ms)")
    yield format_stream_chunk(f"> * 🟢 **크로스리랭크 완료** : 의도별 1등 균등 분배 기반 상위 5개 교차 엄선 ({dt_rerank*1000:.1f} ms)\n>\n", selected_model)
    await asyncio.sleep(0.01)
    
    # 4단계: 제미나이 최종 답안 작성 구간
    t_start = time.time()
    yield format_stream_chunk("> ✍️ *최종 지식 융합 및 답변 구성 중...*\n\n---\n\n", selected_model)
    
    joined_context = "\n".join(final_retrieved_contexts)
    rag_system_instruction = (
        f"{user_context_instruction}\n"
        "당신은 사내 통합 지식 보관소 데이터를 바탕으로 답변을 도출하는 팩트 검토 AI 보좌관입니다.\n"
        "아래 제공되는 [참조 지식 컨텍스트]에 명확히 명시된 팩트만을 기반으로 자연스럽고 읽기 쉽게 답변해야 합니다.\n\n"
        "[답변 철칙]\n"
        "1. 본인의 사전 지식을 활용하여 절대 그럴싸한 거짓말이나 없는 문장을 지어내지 마세요.\n"
        "2. 🚨 [최신성 우선 원칙] 제공된 컨텍스트 내에서 서로 상충되거나 유사한 내용의 자료가 발견될 경우, 본문에 적힌 [기록일자] 정보를 확인하여 가장 최근(최신)에 기록된 데이터를 우선 참조하세요.\n"
        "3. 💡 일반 지식/코드 생성/HTML 작성/포맷팅 요청인 경우:\n"
        "    - [참조 지식 컨텍스트]에 관련 내용이 없더라도, 당신의 기본 지능을 활용하여 요청한 포맷(HTML, 서식 등) 및 질문에 완벽하게 답변하세요.\n\n"        
        "4. 🚨 [하이퍼링크 출처 표기 규칙] 본문 문장 뒤에 출처를 계속 중복해서 붙이지 마세요. 답변이 모두 끝난 맨 마지막 줄에 '---' 구분선을 그은 뒤, 답변 생성에 참고한 모든 출처 정보(태그)를 중복 없이 단 한 번만 리스트 형태로 모아서 명시하세요.\n"
        f"반드시 아래 제공된 [마크다운 링크 형식]을 엄격하게 준수하여 출처를 출력해야 합니다. (model_id 쿼리 파라미터를 절대로 누락하지 마세요)\n\n"
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

    dt_llm = time.time() - t_start
    print(f" 4. 최종답안작성 완료 : 소요시간: {dt_llm:.4f}초 ({dt_llm*1000:.1f} ms)")
    
    total_elapsed = time.time() - start_time
    print(f" ⏱️ 전체 파이프라인 총 연산 완료: {total_elapsed:.4f}초")
    print("="*60 + "\n")
    
    yield "data: [DONE]\n\n"

# 🎯 OpenWebUI 선택 모델 목록 반환
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "3PL지식저장소", "object": "model", "created": int(time.time()), "owned_by": "redbombz"},
            {"id": "Nexacro14", "object": "model", "created": int(time.time()), "owned_by": "redbombz"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    raw_user_name = request.headers.get("x-openwebui-user-name", "Unknown_User")
    user_name = urllib.parse.unquote(raw_user_name)
    user_id = request.headers.get("x-openwebui-user-id", "Unknown_ID")
    user_email = request.headers.get("x-openwebui-user-email", "unknown@company.com")

    base_url = "http://aimeow.ddns.net:20080"
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

    return StreamingResponse(
        handle_knowledge_retrieval_stream(
            user_last_text=user_last_text,
            user_context_instruction=user_context_instruction,
            openai_messages=openai_messages,
            selected_model=selected_model,
            base_url=base_url,
            start_time=start_time
        ),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)