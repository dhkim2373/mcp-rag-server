import os
import re
from datetime import datetime, timedelta
from fastapi import APIRouter, Request, Query, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import psycopg
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(os.path.dirname(current_dir), "templates")
templates = Jinja2Templates(directory=template_path)

router = APIRouter(
    prefix="/view",
    tags=["knowledge-admin"]
)

MY_DATABASE_INFO = {
    "host": "localhost",
    "dbname": "redbombz",
    "user": "redbombz",
    "password": "a11223344*",
    "port": 5432
}

embeddings_engine = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3")


# ==========================================================
# ❶ [웹 화면] 20080/view/list - model_id 필수 검증 적용
# ==========================================================
@router.get("/list", response_class=HTMLResponse)
async def get_knowledge_list(
    request: Request,
    from_date: str = Query(None, alias="from"),
    to_date: str = Query(None, alias="to"),
    keyword: str = Query(None),
    model_id: str = Query(None)  # 🎯 필수값 검증을 위해 기본값 None으로 변경
):
    # 🚨 model_id 누락 시 400 Bad Request 에러 차단
    if not model_id or not model_id.strip():
        raise HTTPException(
            status_code=400, 
            detail="[요청 에러] model_id 파라미터가 누락되었습니다. 정확한 지식 저장소 모델을 지정해 주세요."
        )

    documents = []
    conn = None
    target_model_id = model_id.strip()
    
    # 일주일 전 ~ 오늘 날짜 디폴트 채우기 규칙 유지
    if not from_date or not to_date:
        today = datetime.now()
        one_week_ago = today - timedelta(days=7)
        if not from_date: from_date = one_week_ago.strftime('%Y-%m-%d')
        if not to_date: to_date = today.strftime('%Y-%m-%d')

    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        with conn.cursor() as cur:
            query = """
                SELECT reference_number, SUBSTRING(content FROM 1 FOR 100) as summary, created_at, model_id
                FROM tb_document_chunk
                WHERE is_deleted = 0
                  AND model_id = %s
                  AND created_at BETWEEN %s AND %s
            """
            params = [target_model_id, f"{from_date} 00:00:00", f"{to_date} 23:59:59"]
            
            if keyword and keyword.strip():
                query += " AND content LIKE %s"
                params.append(f"%{keyword.strip()}%")
                
            query += " ORDER BY chunk_id DESC;"
            
            cur.execute(query, params)
            for row in cur.fetchall():
                ref_num, summary, created_at, m_id = row
                
                if isinstance(created_at, datetime):
                    date_str = created_at.strftime('%Y-%m-%d')
                elif created_at:
                    date_str = str(created_at)[:10]
                else:
                    date_str = "-"
                    
                documents.append({
                    "ref_num": ref_num, 
                    "summary": summary,
                    "created_at": date_str,
                    "model_id": m_id
                })
    except Exception as e:
        print(f"❌ [DB 로직 장애] 리스트 조회 실패: {e}")
    finally:
        if conn: conn.close()

    context = {
        "documents": documents, 
        "from_date": from_date, 
        "to_date": to_date,
        "keyword": keyword,
        "selected_model_id": target_model_id
    }
    return templates.TemplateResponse(name="list.html", context=context, request=request)


# ==========================================================
# ❷ [웹 화면] 20080/view/document/{ref_num} - model_id 필수 검증 적용
# ==========================================================
@router.get("/document/{ref_num}", response_class=HTMLResponse)
async def view_and_edit_document(
    request: Request, 
    ref_num: str,
    model_id: str = Query(None)  # 🎯 필수 검증 대상
):
    # 🚨 model_id 누락 시 400 Bad Request 에러 차단
    if not model_id or not model_id.strip():
        raise HTTPException(
            status_code=400, 
            detail="[요청 에러] model_id 파라미터가 누락되었습니다. 어떤 지식 저장소의 문서를 열람할지 명시해야 합니다."
        )

    conn = None
    content_text = ""
    chunk_id = None
    target_ref = ref_num.strip().upper()
    target_model_id = model_id.strip()
    
    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        with conn.cursor() as cur:
            query = """
                SELECT content, chunk_id, model_id FROM tb_document_chunk 
                WHERE TRIM(reference_number) = %s AND model_id = %s AND is_deleted = 0
                ORDER BY chunk_id DESC LIMIT 1;
            """
            cur.execute(query, (target_ref, target_model_id))
            db_row = cur.fetchone()
            
        if not db_row:
            return HTMLResponse(content=f"<html><body><h3>🔍 [{target_model_id}] 모델 내에서 [{target_ref}] 자산을 식별할 수 없습니다.</h3></body></html>", status_code=404)
        
        content_text = db_row[0]
        chunk_id = db_row[1]
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h3>❌ 오류: {str(e)}</h3></body></html>", status_code=500)
    finally:
        if conn: conn.close()
        
    context = {
        "ref_num": target_ref, 
        "content": content_text,
        "chunk_id": chunk_id,
        "model_id": target_model_id,
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return templates.TemplateResponse(name="edit.html", context=context, request=request)


# ==========================================================
# ❸ [비즈니스 기능] chunk_id 및 model_id 검증 수정 저장
# ==========================================================
@router.post("/document/{ref_num}/save")
async def save_and_reembed_document(
    ref_num: str, 
    content: str = Form(...), 
    chunk_id: int = Form(...),
    model_id: str = Form(...)  # 🎯 Form에서도 model_id를 필수 전송값으로 지정
):
    if not model_id or not model_id.strip():
        raise HTTPException(
            status_code=400, 
            detail="[수정 에러] model_id가 누락되어 저장을 완료할 수 없습니다."
        )

    conn = None
    target_model_id = model_id.strip()
    try:
        clean_text = content.strip()
        q_vector = embeddings_engine.embed_query(clean_text)
        
        conn = psycopg.connect(**MY_DATABASE_INFO)
        with conn.cursor() as cur:
            update_query = """
                UPDATE tb_document_chunk 
                SET content = %s, 
                    embedding = %s::vector,
                    created_at = NOW()
                WHERE chunk_id = %s AND model_id = %s;
            """
            cur.execute(update_query, (clean_text, q_vector, chunk_id, target_model_id))
            conn.commit()
            
        return RedirectResponse(url=f"/view/list?model_id={target_model_id}", status_code=303)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"정밀 고유 키 업데이트 중 실패: {str(e)}")
    finally:
        if conn: conn.close()