import time
import re
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings

# ============================================================
# ⚙️ 데이터베이스 & 임베딩 엔진 설정
# ============================================================
MY_DATABASE_INFO = {
    "host": "localhost",
    "dbname": "redbombz",
    "user": "redbombz",
    "password": "a11223344*",
    "port": 5432
}

embeddings_engine = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3")

# main.py 와의 커널 병합을 위한 Router 선언
router = APIRouter(tags=["Webhook & Knowledge Ingestion"])


# ============================================================
# 📩 웹훅 수신 전용 Pydantic 모델 (🎯 최신 규격 반영)
# ============================================================
class ChunkLine(BaseModel):
    page_no: Optional[str] = "1"
    text: str

class WebhookPayload(BaseModel):
    user_name: Optional[str] = "SYSTEM"
    global_prefix: Optional[str] = ""
    source_filename: Optional[str] = "WEBHOOK_INPUT"
    model_id: Optional[str] = "3PL지식저장소"  # 🎯 모델격리 ID (기본값 설정)
    chunks: List[ChunkLine]

# ============================================================
# 🔄 배치(Batch) 루프 프로세서
# ============================================================
def process_batch():
    """tb_raw_document에서 READY 상태 데이터를 가져와 임베딩 후 tb_document_chunk에 이관"""
    raw_id = None
    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        cur = conn.cursor()
        
        # 1) 처리 대기 중인 스테이징 데이터 픽업
        cur.execute("""
            SELECT raw_id, user_name, content, source_filename, model_id 
            FROM tb_raw_document 
            WHERE status = 'READY' 
            ORDER BY raw_id ASC 
            LIMIT 1;
        """)
        row = cur.fetchone()
        
        if not row:
            cur.close(); conn.close(); return
            
        raw_id, user_name, content, source_filename, model_id = row
        print(f"\n🚀 [배치 엔진] 연산 시작 - Raw ID: {raw_id} | 모델 ID: {model_id} (파일명: {source_filename})")
        
        # 선점 상태 변경 (PROCESSING)
        cur.execute("UPDATE tb_raw_document SET status = 'PROCESSING' WHERE raw_id = %s;", (raw_id,))
        conn.commit()
        
        # 2) 파일 버전 오버라이드 (해당 model_id 대상 멱등성 보장)
        if source_filename and source_filename != 'DIRECT_INPUT':
            file_match_pattern = f"[{source_filename} | %"
            
            purge_old_version_query = """
                UPDATE tb_document_chunk 
                SET is_deleted = 1 
                WHERE content LIKE %s 
                  AND model_id = %s 
                  AND is_deleted = 0;
            """
            cur.execute(purge_old_version_query, (file_match_pattern, model_id))
            purg_count = cur.rowcount
            if purg_count > 0:
                print(f"   ℹ️ [파일 버전 오버라이드] ({model_id}) 기존 지식 자산 {purg_count}개 조항을 논리 삭제했습니다.")
        
        # 3) 임베딩 벡터 생성
        final_content = f"{content.strip()}"
        data_vector = embeddings_engine.embed_query(final_content)
        
        cur.execute("SELECT COALESCE(MAX(chunk_order), 0) + 1 FROM tb_document_chunk;")
        next_order = cur.fetchone()[0]
        
        # 4) 출처 태그 격리 분기
        ref_match = re.search(r'\|\s*REF:(.*?)(?=\])', final_content)
        if ref_match:
            ref_tag = ref_match.group(1).strip()
        else:
            ref_tag = f"USER_MEM_{next_order}"
        
        # 🎯 RAG 검색 테이블 적재 (model_id 저장)
        insert_query = """
            INSERT INTO tb_document_chunk (chunk_order, reference_number, content, embedding, model_id, is_deleted)
            VALUES (%s, %s, %s, %s, %s, 0);
        """
        cur.execute(insert_query, (next_order, ref_tag, final_content, data_vector, model_id))
        
        # 5) 완료 처리 마킹
        cur.execute("UPDATE tb_raw_document SET status = 'COMPLETED', processed_at = NOW() WHERE raw_id = %s;", (raw_id,))
        conn.commit()
        print(f"   ➔ 🟢 최종 RAG 벡터 공간 이관 성공! (모델: [{model_id}] / 태그: [{ref_tag}])")
        
        cur.close(); conn.close()
    except Exception as e:
        print(f"   ❌ 배치 파이프라인 에러 발생: {e}")
        if raw_id:
            try:
                conn = psycopg.connect(**MY_DATABASE_INFO)
                cur = conn.cursor()
                cur.execute("UPDATE tb_raw_document SET status = 'ERROR' WHERE raw_id = %s;", (raw_id,))
                conn.commit()
                cur.close(); conn.close()
            except: pass


# ============================================================
# ⏱️ 백그라운드 스케줄러 태스크
# ============================================================
async def run_batch_loop():
    print("⏳ 명시적 REF 추적 및 지식 DB 배치 스케줄러 가동 중...")
    while True:
        try:
            await asyncio.to_thread(process_batch)
        except Exception as e:
            print(f"   ⚠️ 배치 스케줄러 실행 예외: {e}")
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    batch_task = asyncio.create_task(run_batch_loop())
    yield
    batch_task.cancel()


# ============================================================
# 🌐 FastAPI 앱 및 라우터 정의
# ============================================================
app = FastAPI(title="Knowledge Base Ingestion & Embedding Service", lifespan=lifespan)


@router.post("/api/webhook/ingest")
def webhook_ingest_chunks(payload: WebhookPayload):
    """
    🎯 AI Meow에서 정제 및 결합이 완료된 청크 데이터를 수신하여
    tb_raw_document 테이블에 'READY' 상태로 적재
    """
    conn = None
    chunk_count = 0
    target_model_id = payload.model_id if payload.model_id else "3PL지식저장소"
    filename_val = payload.source_filename if payload.source_filename else "WEBHOOK_INPUT"

    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO public.tb_raw_document (user_name, content, status, source_filename, model_id) 
            VALUES (%s, %s, 'READY', %s, %s);
        """
        
        # 🎯 최신 수신 규격(chunks: [{ page_no, text }]) 1:1 적재
        for chunk in payload.chunks:
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue
                
            cursor.execute(insert_query, (
                payload.user_name, 
                chunk_text, 
                filename_val, 
                target_model_id
            ))
            chunk_count += 1
        
        conn.commit()
        cursor.close()
    except Exception as e:
        if conn: 
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Webhook DB 적재 실패: {str(e)}")
    finally:
        if conn: 
            conn.close()
                
    return {
        "status": "success", 
        "message": f"성공적으로 {chunk_count}개의 청크가 [{target_model_id}] 지식 DB에 수신/적재되었습니다!"
    }


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)