import asyncio
from contextlib import asynccontextmanager
import psycopg
import uvicorn
from mcp import FastMCP
from langchain_ollama import OllamaEmbeddings

# ⚙️ DB 및 임베딩 엔진 설정
MY_DATABASE_INFO = {
    "host": "localhost",
    "dbname": "redbombz",
    "user": "redbombz",
    "password": "a11223344*",
    "port": 5432
}

embeddings_engine = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3")

# ==========================================================
# 🛠️ [FastMCP 인스턴스 생성 및 RAG 검색 툴 정의]
# ==========================================================
mcp = FastMCP("3PL-RAG-Knowledge-MCP")

@mcp.tool()
def search_knowledge_chunks(query: str, model_id: str = "3PL지식저장소", top_k: int = 5) -> str:
    """
    외부 MCP 클라이언트(Cursor, Claude Desktop 등)용 RAG 검색 툴.
    입력된 질의(query)를 벡터 임베딩하여 지정된 model_id 내의 가장 연관성 높은 지식 청크를 탐색합니다.
    
    :param query: 검색할 질문 또는 키워드 문장
    :param model_id: 탐색할 지식 저장소 모델 ID (기본값: '3PL지식저장소')
    :param top_k: 추출할 상위 문서 개수 (기본값: 5)
    :return: 연관 지식 문서 컨텍스트 텍스트
    """
    conn = None
    retrieved_contexts = []
    
    try:
        # 1. 질의 텍스트 벡터 임베딩 변환
        q_vector = embeddings_engine.embed_query(query.strip())
        
        # 2. pgvector 코사인 유사도 검색 (model_id 격리 반영)
        conn = psycopg.connect(**MY_DATABASE_INFO)
        with conn.cursor() as cur:
            search_query = """
                SELECT reference_number, content, (1 - (embedding <=> %s::vector)) AS similarity, created_at
                FROM tb_document_chunk 
                WHERE is_deleted = 0 
                  AND model_id = %s
                ORDER BY embedding <=> %s::vector ASC, chunk_id DESC 
                LIMIT %s;
            """
            cur.execute(search_query, (q_vector, model_id, q_vector, top_k))
            rows = cur.fetchall()
            
            for row in rows:
                ref_num, content_text, sim_score, created_at = row
                date_str = created_at.strftime('%Y-%m-%d') if created_at else "-"
                
                fmt_item = f"📌 [{ref_num}] (기록일자: {date_str} / 유사도: {sim_score:.4f})\n{content_text}"
                retrieved_contexts.append(fmt_item)
                
    except Exception as db_err:
        return f"❌ [RAG MCP 검색 실패] 원인: {str(db_err)}"
    finally:
        if conn:
            conn.close()
            
    if not retrieved_contexts:
        return f"🔍 [{model_id}] 모델 내에서 '{query}' 관련 지식 데이터를 찾지 못했습니다."
        
    return f"💡 [{model_id}] RAG 지식 검색 결과 (총 {len(retrieved_contexts)}건):\n\n" + "\n\n---\n\n".join(retrieved_contexts)


# ==========================================================
# ⏱️ [MCP 생명주기 관리자]: FastAPI Lifespan에서 구동할 비동기 스케줄러
# ==========================================================
@asynccontextmanager
async def mcp_server_lifespan():
    print("🚀 [원격 MCP SSE 서버] 포트 8001 가동 시작 (Endpoint: http://0.0.0.0:8001/sse)")
    
    # 🎯 FastMCP 내장 SSE 앱 접근 객체 추출 (버전 차이 대비 예외 안전 처리)
    sse_app = getattr(mcp, "_sse_app", None) or getattr(mcp, "sse_app", None) or mcp
    
    mcp_config = uvicorn.Config(sse_app, host="0.0.0.0", port=8001, log_level="warning")
    mcp_server = uvicorn.Server(mcp_config)
    mcp_task = asyncio.create_task(mcp_server.serve())
    
    try:
        yield
    finally:
        print("🛑 [원격 MCP SSE 서버] 안전 종료 중...")
        mcp_server.should_exit = True
        await mcp_task