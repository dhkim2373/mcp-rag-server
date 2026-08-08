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
# 📩 웹훅 수신 전용 Pydantic 모델 & 마크다운 정제 헬퍼
# ============================================================
class ChunkLine(BaseModel):
    line_index: str
    page_number: Optional[int] = 1
    text: str
    is_split_point: Optional[bool] = False
    is_deleted: Optional[bool] = False

class WebhookPayload(BaseModel):
    user_name: Optional[str] = "SYSTEM"
    global_prefix: Optional[str] = ""
    source_filename: Optional[str] = "WEBHOOK_INPUT"
    model_id: Optional[str] = "3PL지식저장소"  # 🎯 모델격리 ID 추가 (기본값 설정)
    chunks: List[ChunkLine]

def strip_markdown(text_content: str) -> str:
    if not text_content: 
        return ""
    text_content = re.sub(r'<br\s*/?>', ' ', text_content, flags=re.IGNORECASE)
    text_content = re.sub(r'#{1,6}\s+', '', text_content)
    text_content = re.sub(r'\*\*([^*]+)\*\*?', r'\1', text_content)
    text_content = re.sub(r'\*([^*]+)\*', r'\1', text_content)
    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
    return "\n".join(lines)


# ============================================================
# 🔄 배치(Batch) 루프 프로세서 (🎯 model_id 이관 및 document_id 제거)
# ============================================================
def process_batch():
    """tb_raw_document에서 READY 상태 데이터를 가져와 임베딩 후 tb_document_chunk에 이관"""
    raw_id = None
    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        cur = conn.cursor()
        
        # 1) 처리 대기 중인 스테이징 데이터 픽업 (model_id 함께 추출)
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
        
        # 🎯 RAG 검색 테이블 적재 (document_id 제거, model_id 저장)
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
    print("⏳ 명시적 REF 추적 및 오픈웹UI 메모리 배치 스케줄러 가동 중...")
    while True:
        try:
            await asyncio.to_thread(process_batch)
        except Exception as e:
            print(f"   ⚠️ 배치 스케줄러 실행 예외: {e}")
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 스타트업 시 백그라운드 배치 프로세스 시작
    batch_task = asyncio.create_task(run_batch_loop())
    yield
    # 서버 종료 시 배치 태스크 취소
    batch_task.cancel()


# ============================================================
# 🌐 FastAPI 앱 및 라우터 정의
# ============================================================
app = FastAPI(title="Knowledge Base Ingestion & Embedding Service", lifespan=lifespan)


@router.post("/api/webhook/ingest")
def webhook_ingest_chunks(payload: WebhookPayload):
    """
    AI Meow 등 외부 시스템에서 가공이 끝난 청크 데이터를 수신하여
    tb_raw_document 테이블에 지정된 model_id와 함께 'READY' 상태로 적재
    """
    valid_chunks = [item for item in payload.chunks if not getattr(item, 'is_deleted', False)]
    current_chunk_buffer = []
    chunk_group_idx = 0
    conn = None

    target_model_id = payload.model_id if payload.model_id else "3PL지식저장소"

    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        cursor = conn.cursor()
        
        # 🎯 model_id 컬럼 바인딩 추가
        insert_query = """
            INSERT INTO public.tb_raw_document (user_name, content, status, source_filename, model_id) 
            VALUES (%s, %s, 'READY', %s, %s);
        """
        
        for idx, item in enumerate(valid_chunks):
            current_chunk_buffer.append(item.text)
            
            if item.is_split_point or idx == len(valid_chunks) - 1:
                raw_markdown_content = "\n".join(current_chunk_buffer).strip()
                if raw_markdown_content:
                    clean_plain_text = strip_markdown(raw_markdown_content)
                    prefix_str = payload.global_prefix.strip() if payload.global_prefix else ""
                    if prefix_str: 
                        clean_plain_text = f"[{prefix_str}]\n{clean_plain_text}"
                    
                    filename_val = payload.source_filename if payload.source_filename else "WEBHOOK_INPUT"
                    
                    cursor.execute(insert_query, (payload.user_name, clean_plain_text, filename_val, target_model_id))
                    chunk_group_idx += 1
                    current_chunk_buffer = [] 
        
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
        "message": f"성공적으로 {chunk_group_idx}개의 청크가 [{target_model_id}] 지식 DB에 수신/적재되었습니다!"
    }


# 라우터 엔드포인트를 독립 실행 시에도 동작하도록 병합
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)