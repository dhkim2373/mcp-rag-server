import os
import re
import asyncio
import threading
from typing import List, Optional
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv, set_key

import tkinter as tk
from tkinter import messagebox

# ============================================================
# ⚙️ 전역 설정 및 동적 로드 관리 클래스
# ============================================================
ENV_FILE_PATH = ".env"
load_dotenv(ENV_FILE_PATH, override=True)

# UI 및 관리 대상 키 목록에 EMBEDDING_MODEL 추가
CONFIG_KEYS = [
    "BASE_URL",
    "GEMINI_API_KEY",
    "DB_HOST",
    "DB_PORT",    
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",
    "DEFAULT_MODEL_ID"
]

class SettingsManager:
    @classmethod
    def reload(cls):
        load_dotenv(ENV_FILE_PATH, override=True)
        cls.DB_INFO = {
            "host": os.getenv("DB_HOST", "localhost"),
            "dbname": os.getenv("DB_NAME", ""),
            "user": os.getenv("DB_USER", ""),
            "password": os.getenv("DB_PASSWORD", ""),
            "port": int(os.getenv("DB_PORT", 5432))
        }
        cls.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cls.DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "3PL지식저장소")
        cls.BASE_URL = os.getenv("BASE_URL", "http://host.docker.internal:8050")
        cls.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
        
        print(f"⚙️ [설정 갱신] Ollama 엔진 재적재 중... (URL: {cls.OLLAMA_BASE_URL} / 모델: {cls.EMBEDDING_MODEL})")
        global embeddings_engine
        embeddings_engine = OllamaEmbeddings(base_url=cls.OLLAMA_BASE_URL, model=cls.EMBEDDING_MODEL)

SettingsManager.reload()

router = APIRouter(tags=["Webhook & Knowledge Ingestion"])


# ============================================================
# 📩 웹훅 수신 전용 Pydantic 모델
# ============================================================
class ChunkLine(BaseModel):
    page_no: Optional[str] = "1"
    text: str

class WebhookPayload(BaseModel):
    user_name: Optional[str] = "SYSTEM"
    global_prefix: Optional[str] = ""
    source_filename: Optional[str] = "WEBHOOK_INPUT"
    model_id: Optional[str] = None
    chunks: List[ChunkLine]


# ============================================================
# 🔄 배치(Batch) 루프 프로세서
# ============================================================
def process_batch():
    raw_id = None
    try:
        conn = psycopg.connect(**SettingsManager.DB_INFO)
        cur = conn.cursor()
        
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
        
        cur.execute("UPDATE tb_raw_document SET status = 'PROCESSING' WHERE raw_id = %s;", (raw_id,))
        conn.commit()
        
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
        
        final_content = f"{content.strip()}"
        data_vector = embeddings_engine.embed_query(final_content)
        
        cur.execute("SELECT COALESCE(MAX(chunk_order), 0) + 1 FROM tb_document_chunk;")
        next_order = cur.fetchone()[0]
        
        ref_match = re.search(r'\|\s*REF:(.*?)(?=\])', final_content)
        if ref_match:
            ref_tag = ref_match.group(1).strip()
        else:
            ref_tag = f"USER_MEM_{next_order}"
        
        insert_query = """
            INSERT INTO tb_document_chunk (chunk_order, reference_number, content, embedding, model_id, is_deleted)
            VALUES (%s, %s, %s, %s, %s, 0);
        """
        cur.execute(insert_query, (next_order, ref_tag, final_content, data_vector, model_id))
        
        cur.execute("UPDATE tb_raw_document SET status = 'COMPLETED', processed_at = NOW() WHERE raw_id = %s;", (raw_id,))
        conn.commit()
        print(f"  ➔ 🟢 최종 RAG 벡터 공간 이관 성공! (모델: [{model_id}] / 태그: [{ref_tag}])")
        
        cur.close(); conn.close()
    except Exception as e:
        print(f"  ❌ 배치 파이프라인 에러 발생: {e}")
        if raw_id:
            try:
                conn = psycopg.connect(**SettingsManager.DB_INFO)
                cur = conn.cursor()
                cur.execute("UPDATE tb_raw_document SET status = 'ERROR' WHERE raw_id = %s;", (raw_id,))
                conn.commit()
                cur.close(); conn.close()
            except: pass


# ============================================================
# ⏱️ 백그라운드 스케줄러 태스크
# ============================================================
batch_stop_event = asyncio.Event()

async def run_batch_loop():
    print("⏳ 명시적 REF 추적 및 지식 DB 배치 스케줄러 가동 중...")
    while not batch_stop_event.is_set():
        try:
            await asyncio.to_thread(process_batch)
        except Exception as e:
            print(f"   ⚠️ 배치 스케줄러 실행 예외: {e}")
        try:
            await asyncio.wait_for(batch_stop_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            continue

@asynccontextmanager
async def lifespan(app: FastAPI):
    batch_task = asyncio.create_task(run_batch_loop())
    yield
    batch_stop_event.set()
    batch_task.cancel()


# ============================================================
# 🌐 FastAPI 앱 및 라우터 정의
# ============================================================
app = FastAPI(title="Knowledge Base Ingestion & Embedding Service", lifespan=lifespan)

@router.post("/api/webhook/ingest")
def webhook_ingest_chunks(payload: WebhookPayload):
    conn = None
    chunk_count = 0
    target_model_id = payload.model_id if payload.model_id else SettingsManager.DEFAULT_MODEL_ID
    filename_val = payload.source_filename if payload.source_filename else "WEBHOOK_INPUT"

    try:
        conn = psycopg.connect(**SettingsManager.DB_INFO)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO public.tb_raw_document (user_name, content, status, source_filename, model_id) 
            VALUES (%s, %s, 'READY', %s, %s);
        """
        
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


# ============================================================
# 🖥️ 사용자 편의적 2단 레이아웃 Tkinter 설정 UI 클래스
# ============================================================
class ConfigUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚙️ RAG 시스템 환경 설정 및 웹훅 가이드")
        self.root.geometry("620x840")
        self.root.resizable(False, False)

        self.entries = {}

        # 상단 타이틀
        title_label = tk.Label(root, text="RAG 지식 수신 서버 설정 관리", font=("Arial", 14, "bold"), fg="#333")
        title_label.pack(pady=10)

        # ---------------- 패널 1: 웹훅 URL 안내 박스 ----------------
        webhook_frame = tk.LabelFrame(root, text=" 🔗 외부 연동 웹훅(Webhook) 안내 ", font=("Arial", 10, "bold"), fg="#0056b3", padx=10, pady=10)
        webhook_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(webhook_frame, text="외부 시스템(AI Meow 등)에서 지식 데이터를 전송할 엔드포인트 주소입니다:", font=("Arial", 9), anchor="w").pack(fill="x")
        
        self.webhook_entry = tk.Entry(webhook_frame, font=("Arial", 10, "bold"), bg="#eef2f7", fg="#333", justify="center")
        self.webhook_entry.pack(fill="x", pady=5, ipady=3)
        self.update_webhook_display()

        # ---------------- 패널 2: 환경 변수 입력 폼 (2줄 레이아웃 적용) ----------------
        form_outer_frame = tk.LabelFrame(root, text=" 🗄️ PostgreSQL DB 및 서비스 환경 변수 설정 ", font=("Arial", 10, "bold"), fg="#28a745", padx=10, pady=10)
        form_outer_frame.pack(fill="both", expand=True, padx=20, pady=10)

        descriptions = {
            "GEMINI_API_KEY": "• 구글 제미나이 API 인증 키",
            "DB_HOST": "• PostgreSQL DB 호스트 주소 (예: localhost)",
            "DB_NAME": "• 연동할 PostgreSQL 데이터베이스 이름",
            "DB_USER": "• 데이터베이스 접속 계정 아이디",
            "DB_PASSWORD": "• 데이터베이스 접속 비밀번호",
            "DB_PORT": "• PostgreSQL 포트 번호 (기본: 5432)",
            "OLLAMA_BASE_URL": "• 로컬 Ollama AI 서버 주소",
            "BASE_URL": "• 서비스 대표 기본 도메인/URL",
            "DEFAULT_MODEL_ID": "• 기본 지식 저장소 모델 ID",
            "EMBEDDING_MODEL": "• 사용할 텍스트 벡터 임베딩 모델 명칭 (예: bge-m3)"
        }

        for idx, key in enumerate(CONFIG_KEYS):
            row_frame = tk.Frame(form_outer_frame)
            row_frame.pack(fill="x", pady=4)

            top_sub_frame = tk.Frame(row_frame)
            top_sub_frame.pack(fill="x")

            lbl = tk.Label(top_sub_frame, text=key, width=18, anchor="w", font=("Arial", 9, "bold"), fg="#222")
            lbl.pack(side="left")

            val = os.getenv(key, "")
            show_char = "*" if "PASSWORD" in key or "KEY" in key else None
            
            entry = tk.Entry(top_sub_frame, width=45, show=None, font=("Arial", 9))
            entry.insert(0, val)
            entry.pack(side="right", padx=2)
            self.entries[key] = entry

            bottom_sub_frame = tk.Frame(row_frame)
            bottom_sub_frame.pack(fill="x", padx=(18, 0))

            desc_lbl = tk.Label(bottom_sub_frame, text=descriptions.get(key, ""), anchor="w", font=("Arial", 8), fg="#666")
            desc_lbl.pack(side="left", pady=(1, 2))

        # ---------------- 하단: 저장 및 핫스왑 버튼 ----------------
        save_btn = tk.Button(root, text="💾 설정 저장 및 핫스왑(실시간 반영)", bg="#007BFF", fg="white", font=("Arial", 11, "bold"), command=self.save_and_reload)
        save_btn.pack(pady=10, ipadx=15, ipady=5)

    def update_webhook_display(self):
        base = os.getenv("BASE_URL", "http://localhost:8050")
        full_webhook_url = f"{base.rstrip('/')}/api/webhook/ingest"
        self.webhook_entry.config(state="normal")
        self.webhook_entry.delete(0, tk.END)
        self.webhook_entry.insert(0, full_webhook_url)
        self.webhook_entry.config(state="readonly")

    def save_and_reload(self):
        try:
            for key, entry in self.entries.items():
                val = entry.get().strip()
                set_key(ENV_FILE_PATH, key, val)
            
            SettingsManager.reload()
            self.update_webhook_display()
            
            messagebox.showinfo("성공", "✅ 설정이 안전하게 저장되었으며, 임베딩 모델 및 DB 정보가 실시간으로 갱신되었습니다!")
        except Exception as e:
            messagebox.showerror("에러", f"❌ 설정 적용 실패:\n{str(e)}")


def run_tkinter_gui():
    root = tk.Tk()
    app = ConfigUIApp(root)
    root.mainloop()


if __name__ == "__main__":
    ui_thread = threading.Thread(target=run_tkinter_gui, daemon=True)
    ui_thread.start()
    
    print("🚀 [FastAPI 서버] 지식 수신 및 적재 서비스 가동 시작 (포트: 8050)")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)