import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from dotenv import load_dotenv, set_key
from model_config_manager import load_model_configs, save_model_configs

# 💡 Windows 고해상도(HiDPI) 선명도 개선
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

ENV_FILE_PATH = ".env"
load_dotenv(ENV_FILE_PATH, override=True)

# 그룹별 키 정의 (💡 RAG 하이퍼파라미터 키 추가)
ENV_KEYS = [
    "RAG_BASE_URL", 
    "GEMINI_API_KEY", 
    "OLLAMA_BASE_URL", 
    "EMBEDDING_MODEL", 
    "QUERY_SPLIT_MODEL",
    "VECTOR_SEARCH_LIMIT",
    "SIMILARITY_THRESHOLD",
    "USE_RERANK",
    "MAX_TARGET_CHUNKS"
]

DB_KEYS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]

# 항목별 상세 설명 딕셔너리
DESCRIPTIONS = {
    "RAG_BASE_URL": "• 서비스 대표 도메인 또는 백엔드 URL 주소",
    "GEMINI_API_KEY": "• 최종 답변을 위한 구글 제미나이 API 인증 키",
    "OLLAMA_BASE_URL": "• 로컬 Ollama AI 서버 주소",
    "EMBEDDING_MODEL": "• 텍스트 벡터 임베딩 모델 명칭 (예: bge-m3)",
    "QUERY_SPLIT_MODEL": "• 질문 의도 분할 및 정제에 사용할 Ollama LLM 모델 명칭",
    "VECTOR_SEARCH_LIMIT": "• 벡터 DB에서 1차로 수집할 후보 청크 개수 (기본: 10)",
    "SIMILARITY_THRESHOLD": "• 반영할 최소 벡터 유사도 컷라인 (예: 0.35)",
    "USE_RERANK": "• 크로스 인코더 리랭커 사용 여부 (True 또는 False)",
    "MAX_TARGET_CHUNKS": "• 리랭크를 통과하여 최종 LLM에 전달할 최대 청크 개수 (기본: 5)",
    "DB_HOST": "• PostgreSQL DB 호스트 주소 (예: localhost)",
    "DB_PORT": "• PostgreSQL 포트 번호 (기본: 5432)",
    "DB_NAME": "• PostgreSQL 데이터베이스 이름",
    "DB_USER": "• 데이터베이스 접속 계정 아이디",
    "DB_PASSWORD": "• 데이터베이스 접속 비밀번호"
}

class AdvancedConfigApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚙️ RAG 시스템 종합 설정 관리")
        self.root.geometry("720x880")
        self.root.resizable(False, False)
        self.root.configure(bg="#f8f9fa")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#f8f9fa", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Malgun Gothic", 9, "bold"), padding=[10, 6])
        style.map("TNotebook.Tab", background=[("selected", "#0d6efd")], foreground=[("selected", "#ffffff")])

        self.entries = {}
        self.model_configs = load_model_configs()

        self.font_bold = ("Malgun Gothic", 9, "bold")
        self.font_regular = ("Malgun Gothic", 9)
        self.font_small = ("Malgun Gothic", 8)

        # 탭 컨테이너
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # 탭 추가
        self.create_scrollable_tab(" 🌍 기본 환경 및 RAG 제어 ", ENV_KEYS)
        self.create_scrollable_tab(" 🗄️ 데이터베이스(DB) ", DB_KEYS)
        
        self.tab_models = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_models, text=" 🤖 모델 & 프롬프트 ")
        self.create_models_tab(self.tab_models)

        # 전체 저장 버튼
        save_btn = tk.Button(root, text="💾 전체 설정 저장 및 반영", bg="#0d6efd", fg="white", 
                             font=("Malgun Gothic", 10, "bold"), relief="flat", cursor="hand2", 
                             command=self.save_all_configs)
        save_btn.pack(fill="x", padx=10, pady=10, ipady=8)

    def create_scrollable_tab(self, name, keys):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name)
        
        canvas = tk.Canvas(frame, borderwidth=0, highlightthickness=0, bg="#ffffff")
        scroll_frame = tk.Frame(canvas, bg="#ffffff", padx=15, pady=10)
        vsbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsbar.set)

        vsbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_frame.bind("<Configure>", lambda event, c=canvas: c.configure(scrollregion=c.bbox("all")))

        for key in keys:
            row = tk.Frame(scroll_frame, bg="#ffffff", pady=4)
            row.pack(fill="x")
            
            top_sub = tk.Frame(row, bg="#ffffff")
            top_sub.pack(fill="x")

            lbl = tk.Label(top_sub, text=key, width=22, anchor="w", font=self.font_bold, fg="#212529", bg="#ffffff")
            lbl.pack(side="left")
            
            entry = tk.Entry(top_sub, width=38, font=self.font_regular, fg="#212529", bg="#fdfdfe", relief="solid", bd=1)
            entry.insert(0, os.getenv(key, ""))
            entry.pack(side="right", ipady=3)
            self.entries[key] = entry
            
            bot_sub = tk.Frame(row, bg="#ffffff")
            bot_sub.pack(fill="x", padx=(22, 0), pady=(2, 0))
            
            desc_lbl = tk.Label(bot_sub, text=DESCRIPTIONS.get(key, ""), anchor="w", font=self.font_small, fg="#6c757d", bg="#ffffff")
            desc_lbl.pack(side="left")

            tk.Frame(scroll_frame, height=1, bg="#e9ecef").pack(fill="x", pady=6)

    def create_models_tab(self, parent):
        main = tk.Frame(parent, bg="#ffffff", padx=15, pady=15)
        main.pack(fill="both", expand=True)

        ctrl = tk.LabelFrame(main, text=" 🤖 모델 ID 관리 ", font=self.font_bold, fg="#0d6efd", bg="#ffffff", padx=10, pady=10)
        ctrl.pack(fill="x", pady=(0, 10))

        sub_ctrl = tk.Frame(ctrl, bg="#ffffff")
        sub_ctrl.pack(fill="x", pady=5)

        tk.Label(sub_ctrl, text="등록된 모델:", font=self.font_bold, bg="#ffffff").pack(side="left", padx=(0, 8))

        self.model_combobox = ttk.Combobox(sub_ctrl, values=list(self.model_configs.keys()), state="readonly", font=self.font_regular, width=22)
        self.model_combobox.pack(side="left", padx=5)
        if self.model_configs:
            self.model_combobox.current(0)
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_select)

        tk.Button(sub_ctrl, text="➕ 신규 추가", bg="#198754", fg="white", font=self.font_small, relief="flat", cursor="hand2", command=self.open_add_model_dialog).pack(side="left", padx=6, ipadx=6, ipady=3)
        tk.Button(sub_ctrl, text="🗑️ 삭제", bg="#dc3545", fg="white", font=self.font_small, relief="flat", cursor="hand2", command=self.delete_current_model).pack(side="left", padx=2, ipadx=6, ipady=3)

        p_frame = tk.LabelFrame(main, text=" 📝 선택된 모델 ID별 최종 출력 프롬프트 설정 ", font=self.font_bold, fg="#198754", bg="#ffffff", padx=10, pady=10)
        p_frame.pack(fill="both", expand=True)

        tk.Label(p_frame, text="해당 모델 ID로 RAG 추론을 수행할 때 적용될 시스템 프롬프트(지침)입니다:", font=self.font_small, fg="#6c757d", bg="#ffffff", anchor="w").pack(fill="x", pady=(0, 5))

        text_container = tk.Frame(p_frame, bg="#ffffff", bd=1, relief="solid")
        text_container.pack(fill="both", expand=True, pady=5)

        self.prompt_text_widget = tk.Text(text_container, height=14, font=self.font_regular, fg="#212529", bg="#fdfdfe", wrap="word", bd=0, padx=8, pady=8)
        self.prompt_text_widget.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.prompt_text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        self.prompt_text_widget.configure(yscrollcommand=scrollbar.set)

        self.update_prompt_text_view()

    def on_model_select(self, event):
        self.update_prompt_text_view()

    def update_prompt_text_view(self):
        selected = self.model_combobox.get()
        if selected in self.model_configs:
            self.prompt_text_widget.delete("1.0", tk.END)
            self.prompt_text_widget.insert("1.0", self.model_configs[selected])

    def open_add_model_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("신규 모델 ID 추가")
        dialog.geometry("360x190")
        dialog.resizable(False, False)
        dialog.grab_set()

        tk.Label(dialog, text="추가할 모델 ID 이름을 입력하세요:", font=self.font_bold, fg="#333333").pack(pady=(20, 10))
        
        entry = tk.Entry(dialog, width=32, font=self.font_regular, bd=1, relief="solid")
        entry.pack(ipady=4, pady=5)
        entry.focus()

        def confirm_add():
            new_id = entry.get().strip()
            if not new_id:
                messagebox.showwarning("경고", "모델 ID를 입력해주세요.", parent=dialog)
                return
            if new_id in self.model_configs:
                messagebox.showwarning("경고", "이미 존재하는 모델 ID입니다.", parent=dialog)
                return
            
            self.model_configs[new_id] = "새로 추가된 모델의 기본 시스템 프롬프트입니다."
            self.model_combobox['values'] = list(self.model_configs.keys())
            self.model_combobox.set(new_id)
            self.update_prompt_text_view()
            dialog.destroy()

        tk.Button(dialog, text="확인", bg="#0d6efd", fg="white", font=self.font_bold, width=12, relief="flat", cursor="hand2", command=confirm_add).pack(pady=15, ipady=3)

    def delete_current_model(self):
        current_model = self.model_combobox.get()
        if len(self.model_configs) <= 1:
            messagebox.showwarning("경고", "최소 1개의 모델 ID는 유지되어야 합니다.", parent=self.root)
            return
        
        if messagebox.askyesno("확인", f"정말로 [{current_model}] 모델을 삭제하시겠습니까?", parent=self.root):
            del self.model_configs[current_model]
            remaining_models = list(self.model_configs.keys())
            self.model_combobox['values'] = remaining_models
            self.model_combobox.set(remaining_models[0])
            self.update_prompt_text_view()
            messagebox.showinfo("완료", "모델이 삭제되었습니다.", parent=self.root)

    def save_all_configs(self):
        try:
            for key, entry in self.entries.items():
                val = entry.get().strip()
                set_key(ENV_FILE_PATH, key, val)
            load_dotenv(ENV_FILE_PATH, override=True)

            current_model = self.model_combobox.get()
            if current_model:
                current_prompt = self.prompt_text_widget.get("1.0", tk.END).strip()
                self.model_configs[current_model] = current_prompt

            save_model_configs(self.model_configs)
            messagebox.showinfo("성공", "✅ 모든 환경 설정과 RAG 파라미터가 성공적으로 저장되었습니다!")
        except Exception as e:
            messagebox.showerror("에러", f"설정 저장 실패:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedConfigApp(root)
    root.mainloop()