import os
import json
import urllib.request
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

ENV_KEYS = [
    "RAG_BASE_URL", 
    "GEMINI_API_KEY", 
    "OLLAMA_BASE_URL", 
    "EMBEDDING_MODEL", 
    "QUERY_SPLIT_MODEL"
]

DB_KEYS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]

DESCRIPTIONS = {
    "RAG_BASE_URL": "• 서비스 대표 도메인 또는 백엔드 URL 주소",
    "GEMINI_API_KEY": "• 최종 답변을 위한 구글 제미나이 API 인증 키",
    "OLLAMA_BASE_URL": "• 로컬 Ollama AI 서버 주소",
    "EMBEDDING_MODEL": "• 텍스트 벡터 임베딩 모델 명칭 (예: bge-m3)",
    "QUERY_SPLIT_MODEL": "• 질문 의도 분할 및 정제에 사용할 Ollama LLM 모델 명칭",
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
        self.root.geometry("720x960")
        self.root.resizable(False, False)
        self.root.configure(bg="#f8f9fa")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#f8f9fa", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Malgun Gothic", 9, "bold"), padding=[10, 6])
        style.map("TNotebook.Tab", background=[("selected", "#0d6efd")], foreground=[("selected", "#ffffff")])

        self.entries = {}
        
        raw_configs = load_model_configs()
        self.model_configs = {}
        for k, v in raw_configs.items():
            if isinstance(v, dict):
                self.model_configs[k] = {
                    "system_prompt": v.get("system_prompt", v.get("prompt", "")),
                    "temperature": float(v.get("temperature", 0.0)),
                    "search_mode": v.get("search_mode", "vector"),
                    "vector_search_limit": int(v.get("vector_search_limit", 10)),
                    "similarity_threshold": float(v.get("similarity_threshold", 0.35)),
                    "use_rerank": str(v.get("use_rerank", "True")),
                    "max_target_chunks": int(v.get("max_target_chunks", 5))
                }
            else:
                self.model_configs[k] = {
                    "system_prompt": str(v),
                    "temperature": 0.0,
                    "search_mode": "vector",
                    "vector_search_limit": 10,
                    "similarity_threshold": 0.35,
                    "use_rerank": "True",
                    "max_target_chunks": 5
                }

        self.font_bold = ("Malgun Gothic", 9, "bold")
        self.font_regular = ("Malgun Gothic", 9)
        self.font_small = ("Malgun Gothic", 8)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.create_scrollable_tab(" 🌍 기본 환경 설정 ", ENV_KEYS)
        self.create_scrollable_tab(" 🗄️ 데이터베이스(DB) ", DB_KEYS)
        
        self.tab_models = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_models, text=" 🤖 모델 & RAG 파라미터 ")
        self.create_models_tab(self.tab_models)

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

        # 1. 모델 선택 바
        ctrl = tk.LabelFrame(main, text=" Model Configuration ", font=self.font_bold, fg="#0d6efd", bg="#ffffff", padx=10, pady=6)
        ctrl.pack(fill="x", pady=(0, 6))

        sub_ctrl1 = tk.Frame(ctrl, bg="#ffffff")
        sub_ctrl1.pack(fill="x", pady=2)

        tk.Label(sub_ctrl1, text="Target Model:", font=self.font_bold, bg="#ffffff", width=14, anchor="w").pack(side="left")
        self.model_combobox = ttk.Combobox(sub_ctrl1, values=list(self.model_configs.keys()), state="readonly", font=self.font_regular, width=22)
        self.model_combobox.pack(side="left", padx=5)
        if self.model_configs:
            self.model_combobox.current(0)
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_select)

        tk.Button(sub_ctrl1, text="➕ Add", bg="#198754", fg="white", font=self.font_small, relief="flat", cursor="hand2", command=self.open_add_model_dialog).pack(side="left", padx=5, ipadx=4, ipady=2)
        tk.Button(sub_ctrl1, text="🗑️ Del", bg="#dc3545", fg="white", font=self.font_small, relief="flat", cursor="hand2", command=self.delete_current_model).pack(side="left", padx=2, ipadx=4, ipady=2)

        # 2. LLM Generation Parameters
        llm_frame = tk.LabelFrame(main, text=" LLM Generation Parameter ", font=self.font_bold, fg="#495057", bg="#ffffff", padx=10, pady=6)
        llm_frame.pack(fill="x", pady=(0, 6))

        row_temp = tk.Frame(llm_frame, bg="#ffffff", pady=2)
        row_temp.pack(fill="x")
        tk.Label(row_temp, text="Temperature:", font=self.font_bold, bg="#ffffff", width=18, anchor="w").pack(side="left")
        self.temp_entry = tk.Entry(row_temp, width=12, font=self.font_regular, relief="solid", bd=1)
        self.temp_entry.pack(side="left", padx=5, ipady=2)
        tk.Label(row_temp, text="(Controls creativity, e.g., 0.0 ~ 1.0)", font=self.font_small, fg="#6c757d", bg="#ffffff").pack(side="left", padx=5)

        # 3. Retrieval Pipeline Parameters
        rag_frame = tk.LabelFrame(main, text=" RAG Retrieval & Filtering Pipeline ", font=self.font_bold, fg="#495057", bg="#ffffff", padx=10, pady=6)
        rag_frame.pack(fill="x", pady=(0, 6))

        # 검색 모드 선택 라디오 버튼 그룹 (수직 배치로 잘림 방지)
        group_mode = tk.LabelFrame(rag_frame, text=" [선택] Search Strategy Mode ", font=self.font_small, fg="#dc3545", bg="#ffffff", padx=8, pady=4)
        group_mode.pack(fill="x", pady=3)

        r_mode = tk.Frame(group_mode, bg="#ffffff", pady=2)
        r_mode.pack(fill="x")
        
        self.search_mode_var = tk.StringVar(value="vector")
        tk.Radiobutton(r_mode, text="Vector Search Only (Dense)", variable=self.search_mode_var, value="vector", 
                       font=self.font_regular, bg="#ffffff", activebackground="#ffffff", cursor="hand2", anchor="w").pack(fill="x", padx=5, pady=1)
        tk.Radiobutton(r_mode, text="Hybrid Search (Vector + Full-Text Search)", variable=self.search_mode_var, value="hybrid", 
                       font=self.font_regular, bg="#ffffff", activebackground="#ffffff", cursor="hand2", anchor="w").pack(fill="x", padx=5, pady=1)

        # 그룹 A: Vector Database Retrieval Group
        group_db = tk.LabelFrame(rag_frame, text=" [1단계] Database Retrieval ", font=self.font_small, fg="#0d6efd", bg="#ffffff", padx=8, pady=4)
        group_db.pack(fill="x", pady=3)

        r1 = tk.Frame(group_db, bg="#ffffff", pady=2)
        r1.pack(fill="x")
        tk.Label(r1, text="Vector Search Limit:", font=self.font_bold, bg="#ffffff", width=22, anchor="w").pack(side="left")
        self.limit_entry = tk.Entry(r1, width=10, font=self.font_regular, relief="solid", bd=1)
        self.limit_entry.pack(side="left", padx=5, ipady=2)
        tk.Label(r1, text="Number of candidate chunks fetched", font=self.font_small, fg="#6c757d", bg="#ffffff").pack(side="left", padx=5)

        r2 = tk.Frame(group_db, bg="#ffffff", pady=2)
        r2.pack(fill="x")
        tk.Label(r2, text="Similarity Threshold:", font=self.font_bold, bg="#ffffff", width=22, anchor="w").pack(side="left")
        self.thresh_entry = tk.Entry(r2, width=10, font=self.font_regular, relief="solid", bd=1)
        self.thresh_entry.pack(side="left", padx=5, ipady=2)
        tk.Label(r2, text="Minimum similarity score cutoff", font=self.font_small, fg="#6c757d", bg="#ffffff").pack(side="left", padx=5)

        # 그룹 B: Reranker & Context Assembly Group
        group_rerank = tk.LabelFrame(rag_frame, text=" [2단계] Cross-Encoder Rerank & Context Assembly ", font=self.font_small, fg="#198754", bg="#ffffff", padx=8, pady=4)
        group_rerank.pack(fill="x", pady=3)

        r3 = tk.Frame(group_rerank, bg="#ffffff", pady=2)
        r3.pack(fill="x")
        tk.Label(r3, text="Use Cross-Encoder Rerank:", font=self.font_bold, bg="#ffffff", width=22, anchor="w").pack(side="left")
        self.rerank_var = tk.BooleanVar(value=True)
        rerank_chk = tk.Checkbutton(r3, text="Enable neural re-ranking for precision", variable=self.rerank_var, font=self.font_regular, bg="#ffffff", activebackground="#ffffff", cursor="hand2")
        rerank_chk.pack(side="left", padx=2)

        r4 = tk.Frame(group_rerank, bg="#ffffff", pady=2)
        r4.pack(fill="x")
        tk.Label(r4, text="Max Target Chunks:", font=self.font_bold, bg="#ffffff", width=22, anchor="w").pack(side="left")
        self.max_chunks_entry = tk.Entry(r4, width=10, font=self.font_regular, relief="solid", bd=1)
        self.max_chunks_entry.pack(side="left", padx=5, ipady=2)
        tk.Label(r4, text="Maximum final chunks delivered to LLM", font=self.font_small, fg="#6c757d", bg="#ffffff").pack(side="left", padx=5)

        # 4. System Prompt Editor
        p_frame = tk.LabelFrame(main, text=" System Prompt Instruction ", font=self.font_bold, fg="#198754", bg="#ffffff", padx=10, pady=6)
        p_frame.pack(fill="both", expand=True)

        text_container = tk.Frame(p_frame, bg="#ffffff", bd=1, relief="solid")
        text_container.pack(fill="both", expand=True, pady=3)

        self.prompt_text_widget = tk.Text(text_container, height=5, font=self.font_regular, fg="#212529", bg="#fdfdfe", wrap="word", bd=0, padx=8, pady=8)
        self.prompt_text_widget.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.prompt_text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        self.prompt_text_widget.configure(yscrollcommand=scrollbar.set)

        self.update_model_view()

    def on_model_select(self, event):
        self.update_model_view()

    def update_model_view(self):
        selected = self.model_combobox.get()
        if selected in self.model_configs:
            conf = self.model_configs[selected]
            self.prompt_text_widget.delete("1.0", tk.END)
            self.prompt_text_widget.insert("1.0", conf.get("system_prompt", ""))
            
            self.temp_entry.delete(0, tk.END)
            self.temp_entry.insert(0, str(conf.get("temperature", 0.0)))
            
            self.search_mode_var.set(conf.get("search_mode", "vector"))

            self.limit_entry.delete(0, tk.END)
            self.limit_entry.insert(0, str(conf.get("vector_search_limit", 10)))
            
            self.thresh_entry.delete(0, tk.END)
            self.thresh_entry.insert(0, str(conf.get("similarity_threshold", 0.35)))
            
            rerank_val = conf.get("use_rerank", "True")
            if isinstance(rerank_val, str):
                is_checked = rerank_val.lower() in ("true", "1", "yes")
            else:
                is_checked = bool(rerank_val)
            self.rerank_var.set(is_checked)
            
            self.max_chunks_entry.delete(0, tk.END)
            self.max_chunks_entry.insert(0, str(conf.get("max_target_chunks", 5)))

    def open_add_model_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Model")
        dialog.geometry("360x190")
        dialog.resizable(False, False)
        dialog.grab_set()

        tk.Label(dialog, text="Enter new model identifier (ID):", font=self.font_bold, fg="#333333").pack(pady=(20, 10))
        entry = tk.Entry(dialog, width=32, font=self.font_regular, bd=1, relief="solid")
        entry.pack(ipady=4, pady=5)
        entry.focus()

        def confirm_add():
            new_id = entry.get().strip()
            if not new_id:
                messagebox.showwarning("Warning", "Model ID cannot be empty.", parent=dialog)
                return
            if new_id in self.model_configs:
                messagebox.showwarning("Warning", "Model ID already exists.", parent=dialog)
                return
            
            self.model_configs[new_id] = {
                "system_prompt": "Default system prompt for the newly added model.",
                "temperature": 0.0,
                "search_mode": "vector",
                "vector_search_limit": 10,
                "similarity_threshold": 0.35,
                "use_rerank": "True",
                "max_target_chunks": 5
            }
            self.model_combobox['values'] = list(self.model_configs.keys())
            self.model_combobox.set(new_id)
            self.update_model_view()
            dialog.destroy()

        tk.Button(dialog, text="Confirm", bg="#0d6efd", fg="white", font=self.font_bold, width=12, relief="flat", cursor="hand2", command=confirm_add).pack(pady=15, ipady=3)

    def delete_current_model(self):
        current_model = self.model_combobox.get()
        if len(self.model_configs) <= 1:
            messagebox.showwarning("Warning", "At least one model configuration must be retained.", parent=self.root)
            return
        
        if messagebox.askyesno("Confirmation", f"Are you sure you want to delete [{current_model}]?", parent=self.root):
            del self.model_configs[current_model]
            remaining_models = list(self.model_configs.keys())
            self.model_combobox['values'] = remaining_models
            self.model_combobox.set(remaining_models[0])
            self.update_model_view()
            messagebox.showinfo("Success", "Model configuration has been deleted.", parent=self.root)

    def save_all_configs(self):
        try:
            for key, entry in self.entries.items():
                val = entry.get().strip()
                set_key(ENV_FILE_PATH, key, val)
            load_dotenv(ENV_FILE_PATH, override=True)

            current_model = self.model_combobox.get()
            if current_model:
                current_prompt = self.prompt_text_widget.get("1.0", tk.END).strip()
                
                try:
                    current_temp = float(self.temp_entry.get().strip() or 0.0)
                except ValueError:
                    current_temp = 0.0

                current_search_mode = self.search_mode_var.get()

                try:
                    current_limit = int(self.limit_entry.get().strip() or 10)
                except ValueError:
                    current_limit = 10

                try:
                    current_thresh = float(self.thresh_entry.get().strip() or 0.35)
                except ValueError:
                    current_thresh = 0.35

                current_rerank = "True" if self.rerank_var.get() else "False"

                try:
                    current_max_chunks = int(self.max_chunks_entry.get().strip() or 5)
                except ValueError:
                    current_max_chunks = 5

                self.model_configs[current_model] = {
                    "system_prompt": current_prompt,
                    "temperature": current_temp,
                    "search_mode": current_search_mode,
                    "vector_search_limit": current_limit,
                    "similarity_threshold": current_thresh,
                    "use_rerank": current_rerank,
                    "max_target_chunks": current_max_chunks
                }

            save_model_configs(self.model_configs)

            try:
                req = urllib.request.Request("http://localhost:8000/reload-config", method="POST")
                with urllib.request.urlopen(req, timeout=3) as response:
                    print("🔄 백엔드 핫스왑 동기화 완료:", response.read().decode())
            except Exception as net_err:
                print(f"⚠️ 백엔드 서버 통신 실패: {net_err}")

            messagebox.showinfo("Success", "✅ All model configurations and parameters have been successfully saved and synchronized!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configurations:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedConfigApp(root)
    root.mainloop()