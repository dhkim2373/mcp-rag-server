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
from tkinter import ttk, messagebox

# 💡 Windows 고해상도(HiDPI) 선명도 개선
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# ============================================================
# ⚙️ 전역 설정 및 동적 로드 관리 클래스
# ============================================================
ENV_FILE_PATH = ".env"
load_dotenv(ENV_FILE_PATH, override=True)

CONFIG_KEYS = [
    "DB_HOST",
    "DB_PORT",    
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",    
    "EMBEDDING_BASE_URL",
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
        cls.DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "의약품스마트검색")
        cls.EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8050")
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
# 🚀 [초고속 튜닝] 대용량 배치(Batch) 루프 프로세서 (한 번에 50개씩 처리)
# ============================================================
def process_batch():
    BATCH_SIZE = 50  # 💡 한 번에 처리할 청크 개수 (GPU 성능에 따라 50~100 조절 가능)
    try:
        with psycopg.connect(**SettingsManager.DB_INFO) as conn:
            with conn.cursor() as cur:
                
                # 1. READY 상태인 데이터를 지정한 BATCH_SIZE만큼 일괄 가져오기
                cur.execute("""
                    SELECT raw_id, user_name, content, source_filename, model_id 
                    FROM tb_raw_document 
                    WHERE status = 'READY' 
                    ORDER BY raw_id ASC 
                    LIMIT %s;
                """, (BATCH_SIZE,))
                rows = cur.fetchall()
                
                if not rows:
                    return
                
                raw_ids = [row[0] for row in rows]
                model_id = rows[0][4]  # 같은 배치 내의 모델 ID 기준
                
                print(f"\n🚀 [초고속 배치 엔진] 연산 시작 - 총 {len(rows)}건 일괄 처리 중... (모델 ID: {model_id})")
                
                # 2. 상태를 PROCESSING으로 일괄 변경
                cur.executemany("UPDATE tb_raw_document SET status = 'PROCESSING' WHERE raw_id = %s;", [(rid,) for rid in raw_ids])
                conn.commit()
                
                # 3. 텍스트 추출 및 정제
                contents = []
                ref_tags = []
                
                # 현재 DB의 다음 chunk_order 미리 조회
                cur.execute("SELECT COALESCE(MAX(chunk_order), 0) FROM tb_document_chunk;")
                next_order = cur.fetchone()[0]
                
                for idx, row in enumerate(rows):
                    raw_id, user_name, content, source_filename, m_id = row
                    final_content = content.strip()
                    contents.append(final_content)
                    
                    ref_match = re.search(r'\|\s*REF:(.*?)(?=\])', final_content)
                    if ref_match:
                        ref_tag = ref_match.group(1).strip()
                    else:
                        ref_tag = f"USER_MEM_{next_order + idx + 1}"
                    ref_tags.append(ref_tag)

                # 4. 💡 [핵심] 50개 텍스트를 한 번에(Batch) Ollama로 보내서 초고속 임베딩 연산 수행
                print(f"   ⚡ Ollama 벡터 임베딩 일괄 생성 요청 중 ({len(contents)}건)...")
                data_vectors = embeddings_engine.embed_documents(contents)
                print(f"   ✅ 벡터 임베딩 생성 완료!")

                # 5. DB Bulk Insert 데이터 구성
                insert_data = []
                for idx, row in enumerate(rows):
                    current_order = next_order + idx + 1
                    insert_data.append((
                        current_order,
                        ref_tags[idx],
                        contents[idx],
                        data_vectors[idx],
                        model_id,
                        0
                    ))

                insert_query = """
                    INSERT INTO tb_document_chunk (chunk_order, reference_number, content, embedding, model_id, is_deleted)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """
                cur.executemany(insert_query, insert_data)

                # 6. 원본 문서 상태를 COMPLETED로 일괄 업데이트
                cur.executemany("UPDATE tb_raw_document SET status = 'COMPLETED', processed_at = NOW() WHERE raw_id = %s;", [(rid,) for rid in raw_ids])
                conn.commit()
                
                print(f"   ➔ 🟢 일괄 RAG 벡터 공간 이관 성공! (총 {len(rows)}건 처리 완료)")

    except Exception as e:
        print(f"   ❌ 배치 파이프라인 에러 발생: {e}")
        # 에러 발생 시 현재 배치의 항목들을 ERROR 상태로 안전하게 마킹
        try:
            if 'raw_ids' in locals() and raw_ids:
                with psycopg.connect(**SettingsManager.DB_INFO) as err_conn:
                    with err_conn.cursor() as err_cur:
                        err_cur.executemany("UPDATE tb_raw_document SET status = 'ERROR' WHERE raw_id = %s;", [(rid,) for rid in raw_ids])
                        err_conn.commit()
        except: 
            pass


# ============================================================
# ⏱️ 백그라운드 스케줄러 태스크
# ============================================================
batch_stop_event = asyncio.Event()

async def run_batch_loop():
    print("⏳ 초고속 배치 지식 DB 스케줄러 가동 중...")
    while not batch_stop_event.is_set():
        try:
            await asyncio.to_thread(process_batch)
        except Exception as e:
            print(f"   ⚠️ 배치 스케줄러 실행 예외: {e}")
        
        try:
            # 대기 시간을 1초로 줄여서 데이터가 들어오면 즉시 처리되도록 반응성 향상
            await asyncio.wait_for(batch_stop_event.wait(), timeout=1.0)
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
        with conn.cursor() as cursor:
            insert_query = """
                INSERT INTO public.tb_raw_document (user_name, content, status, source_filename, model_id) 
                VALUES (%s, %s, 'READY', %s, %s);
            """
            
            bulk_rows = []
            for chunk in payload.chunks:
                chunk_text = re.sub(r'[ \t]+', ' ', chunk.text).strip()
                if not chunk_text:
                    continue
                bulk_rows.append((payload.user_name, chunk_text, filename_val, target_model_id))
                chunk_count += 1
            
            cursor.executemany(insert_query, bulk_rows)
            conn.commit()
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
# 🖥️ 전체 스크롤바가 적용된 Tkinter 설정 UI 클래스
# ============================================================
class ConfigUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚙️ RAG 시스템 환경 설정 및 웹훅 가이드")
        self.root.geometry("660x780")
        self.root.resizable(False, False)
        self.root.configure(bg="#f8f9fa")

        self.entries = {}
        
        self.font_title = ("Malgun Gothic", 11, "bold")
        self.font_bold = ("Malgun Gothic", 9, "bold")
        self.font_regular = ("Malgun Gothic", 9)
        self.font_small = ("Malgun Gothic", 8)

        title_label = tk.Label(root, text="RAG 지식 수신 서버 설정 관리", font=self.font_title, fg="#212529", bg="#f8f9fa")
        title_label.pack(pady=(10, 5))

        container = tk.Frame(root, bg="#f8f9fa")
        container.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(container, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f8f9fa")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfig(self.canvas_window, width=event.width))
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        webhook_frame = tk.LabelFrame(self.scrollable_frame, text=" 🔗 외부 연동 웹훅(Webhook) 안내 ", font=self.font_bold, fg="#0056b3", bg="#ffffff", padx=10, pady=6)
        webhook_frame.pack(fill="x", padx=5, pady=4)

        tk.Label(webhook_frame, text="외부 시스템에서 지식 데이터를 전송할 엔드포인트 주소입니다:", font=self.font_small, anchor="w", fg="#495057", bg="#ffffff").pack(fill="x")
        
        self.webhook_entry = tk.Entry(webhook_frame, font=self.font_bold, bg="#e9ecef", fg="#212529", justify="center", relief="solid", bd=1)
        self.webhook_entry.pack(fill="x", pady=(4, 2), ipady=2)
        self.update_webhook_display()

        viewer_frame = tk.LabelFrame(self.scrollable_frame, text=" 📊 PostgreSQL 테이블 생성가이드 ", font=self.font_bold, fg="#d63384", bg="#ffffff", padx=10, pady=6)
        viewer_frame.pack(fill="x", padx=5, pady=4)

        tk.Label(viewer_frame, text="로컬에 저장된 도움말 파일(embedding_table.html)을 직접 엽니다:", font=self.font_small, anchor="w", fg="#495057", bg="#ffffff").pack(fill="x")
        
        import webbrowser
        def open_local_html_viewer():
            html_path = os.path.abspath("./templates/embedding_table.html")
            if os.path.exists(html_path):
                webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
            else:
                messagebox.showerror("파일 없음", f"❌ 지정된 경로에 뷰어 파일이 없습니다:\n{html_path}")

        open_viewer_btn = tk.Button(viewer_frame, text="🌐 HTML 파일 직접 열기", bg="#d63384", fg="white", activebackground="#b02a6b", activeforeground="white", font=self.font_bold, relief="flat", cursor="hand2", command=open_local_html_viewer)
        open_viewer_btn.pack(fill="x", pady=(4, 2), ipady=2)

        form_outer_frame = tk.LabelFrame(self.scrollable_frame, text=" 🗄️ PostgreSQL DB 및 서비스 환경 변수 설정 ", font=self.font_bold, fg="#198754", bg="#ffffff", padx=10, pady=6)
        form_outer_frame.pack(fill="x", padx=5, pady=4)

        descriptions = {
            "DB_HOST": "• PostgreSQL DB 호스트 주소 (예: localhost)",
            "DB_PORT": "• PostgreSQL 포트 번호 (기본: 5432)",
            "DB_NAME": "• PostgreSQL 데이터베이스 이름",
            "DB_USER": "• 데이터베이스 접속 계정 아이디",
            "DB_PASSWORD": "• 데이터베이스 접속 비밀번호",
            "OLLAMA_BASE_URL": "• 로컬 Ollama AI 서버 주소",
            "EMBEDDING_MODEL": "• 사용할 텍스트 벡터 임베딩 모델 명칭 (예: bge-m3)",
            "EMBEDDING_BASE_URL": "• 기본 도메인/URL",
            "DEFAULT_MODEL_ID": "• 기본 지식 저장소 모델 ID"
        }

        for idx, key in enumerate(CONFIG_KEYS):
            row_frame = tk.Frame(form_outer_frame, bg="#ffffff")
            row_frame.pack(fill="x", pady=2)

            top_sub_frame = tk.Frame(row_frame, bg="#ffffff")
            top_sub_frame.pack(fill="x")

            lbl = tk.Label(top_sub_frame, text=key, width=18, anchor="w", font=self.font_bold, fg="#212529", bg="#ffffff")
            lbl.pack(side="left")

            val = os.getenv(key, "")
            entry = tk.Entry(top_sub_frame, width=42, show=None, font=self.font_regular, fg="#212529", bg="#ffffff", relief="solid", bd=1)
            entry.insert(0, val)
            entry.pack(side="right", padx=2, ipady=1)
            self.entries[key] = entry

            bottom_sub_frame = tk.Frame(row_frame, bg="#ffffff")
            bottom_sub_frame.pack(fill="x", padx=(18, 0))

            desc_lbl = tk.Label(bottom_sub_frame, text=descriptions.get(key, ""), anchor="w", font=self.font_small, fg="#6c757d", bg="#ffffff")
            desc_lbl.pack(side="left", pady=(0, 2))

        save_btn = tk.Button(root, text="💾 설정 저장 및 핫스왑(실시간 반영)", bg="#0d6efd", fg="white", activebackground="#0b5ed7", activeforeground="white", font=self.font_bold, relief="flat", cursor="hand2", command=self.save_and_reload)
        save_btn.pack(pady=10, ipadx=12, ipady=5)

    def update_webhook_display(self):
        base = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8050")
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
    
    def on_closing():
        if messagebox.askokcancel("종료", "RAG 임베딩 설정 관리 창을 닫으시겠습니까?\n(백엔드 서버도 함께 종료됩니다.)"):
            root.destroy()
            os._exit(0) 

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    ui_thread = threading.Thread(target=run_tkinter_gui, daemon=True)
    ui_thread.start()
    
    print("🚀 [FastAPI 서버] 지식 수신 및 적재 서비스 가동 시작 (포트: 8050)")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)