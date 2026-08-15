import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from dotenv import load_dotenv, set_key
from model_config_manager import load_model_configs, save_model_configs

# 💡 [핵심 추가] Windows 고해상도(HiDPI) 선명도 개선 (글자 흐림 현상 방지)
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

ENV_FILE_PATH = ".env"
load_dotenv(ENV_FILE_PATH, override=True)

CONFIG_KEYS = [
    "RAG_BASE_URL",    
    "GEMINI_API_KEY",
    "DB_HOST",
    "DB_PORT",    
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL"
]

class AdvancedConfigApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚙️ RAG 시스템 종합 설정 및 모델 프롬프트 관리")
        self.root.geometry("700x800")
        self.root.resizable(False, False)

        # 💡 깔끔하고 모던한 스타일 설정 (ttk 테마 커스텀)
        style = ttk.Style()
        style.theme_use("clam")
        
        # 탭 스타일링
        style.configure("TNotebook", background="#f8f9fa", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Malgun Gothic", 9, "bold"), padding=[12, 8])
        style.map("TNotebook.Tab", background=[("selected", "#0d6efd")], foreground=[("selected", "#ffffff")])

        self.entries = {}
        self.model_configs = load_model_configs()

        # 맑은 고딕 가독성 폰트 정의
        self.font_title = ("Malgun Gothic", 11, "bold")
        self.font_bold = ("Malgun Gothic", 9, "bold")
        self.font_regular = ("Malgun Gothic", 9)
        self.font_small = ("Malgun Gothic", 8)

        # 전체 메인 프레임 배경 (은은한 회색 톤)
        main_container = tk.Frame(root, bg="#f8f9fa")
        main_container.pack(fill="both", expand=True)

        # 탭 컨테이너 생성
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=12)

        # 탭 1: 기본 환경 변수 설정
        self.tab_env = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_env, text=" 🗄️ 기본 환경 변수 ")
        self.create_env_tab(self.tab_env)

        # 탭 2: 모델 ID 및 프롬프트 관리
        self.tab_models = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_models, text=" 🤖 모델 ID & 프롬프트 관리 ")
        self.create_models_tab(self.tab_models)

        # 하단 전체 저장 버튼 영역 (시인성이 뛰어난 모던 블루 컬러 적용)
        bottom_frame = tk.Frame(main_container, bg="#f8f9fa", pady=5)
        bottom_frame.pack(fill="x", padx=12, pady=(0, 15))

        save_btn = tk.Button(
            bottom_frame, 
            text="💾 전체 설정 저장 및 반영", 
            bg="#0d6efd", 
            fg="white", 
            activebackground="#0b5ed7",
            activeforeground="white",
            font=("Malgun Gothic", 10, "bold"), 
            relief="flat",
            cursor="hand2",
            command=self.save_all_configs
        )
        save_btn.pack(fill="x", ipady=6)

    def create_env_tab(self, parent):
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0, bg="#ffffff")
        scroll_frame = tk.Frame(canvas, bg="#ffffff", padx=15, pady=10)
        vsbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsbar.set)

        vsbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_frame.bind("<Configure>", lambda event, c=canvas: c.configure(scrollregion=c.bbox("all")))

        descriptions = {
            "RAG_BASE_URL": "• 서비스 대표 도메인 또는 백엔드 URL 주소",
            "GEMINI_API_KEY": "• 구글 제미나이 API 인증 키",
            "DB_HOST": "• PostgreSQL DB 호스트 주소 (예: localhost)",
            "DB_PORT": "• PostgreSQL 포트 번호 (기본: 5432)",
            "DB_NAME": "• 연동할 PostgreSQL 데이터베이스 이름",
            "DB_USER": "• 데이터베이스 접속 계정 아이디",
            "DB_PASSWORD": "• 데이터베이스 접속 비밀번호",
            "OLLAMA_BASE_URL": "• 로컬 Ollama AI 서버 주소",
            "EMBEDDING_MODEL": "• 텍스트 벡터 임베딩 모델 명칭 (예: bge-m3)"
        }

        for key in CONFIG_KEYS:
            row_frame = tk.Frame(scroll_frame, bg="#ffffff", pady=4)
            row_frame.pack(fill="x")

            top_sub = tk.Frame(row_frame, bg="#ffffff")
            top_sub.pack(fill="x")

            lbl = tk.Label(top_sub, text=key, width=18, anchor="w", font=self.font_bold, fg="#212529", bg="#ffffff")
            lbl.pack(side="left")

            val = os.getenv(key, "")
            entry = tk.Entry(top_sub, width=44, font=self.font_regular, fg="#212529", bg="#fdfdfe", relief="solid", bd=1)
            entry.insert(0, val)
            entry.pack(side="right", padx=2, ipady=3)
            self.entries[key] = entry

            bot_sub = tk.Frame(row_frame, bg="#ffffff")
            bot_sub.pack(fill="x", padx=(18, 0), pady=(2, 0))
            
            desc_lbl = tk.Label(bot_sub, text=descriptions.get(key, ""), anchor="w", font=self.font_small, fg="#6c757d", bg="#ffffff")
            desc_lbl.pack(side="left")

            # 항목 간 구분선을 위한 미세한 라인
            separator = tk.Frame(scroll_frame, height=1, bg="#e9ecef")
            separator.pack(fill="x", pady=4)

    def create_models_tab(self, parent):
        main_frame = tk.Frame(parent, bg="#ffffff", padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)

        # 상단: 모델 선택 및 추가/삭제 영역
        top_ctrl_frame = tk.LabelFrame(main_frame, text=" 🤖 모델 ID 관리 ", font=self.font_bold, fg="#0d6efd", bg="#ffffff", padx=12, pady=10)
        top_ctrl_frame.pack(fill="x", pady=(0, 10))

        model_select_sub = tk.Frame(top_ctrl_frame, bg="#ffffff")
        model_select_sub.pack(fill="x", pady=5)

        tk.Label(model_select_sub, text="등록된 모델:", font=self.font_bold, fg="#333333", bg="#ffffff").pack(side="left", padx=(0, 8))
        
        self.model_combobox = ttk.Combobox(model_select_sub, values=list(self.model_configs.keys()), state="readonly", font=self.font_regular, width=22)
        self.model_combobox.pack(side="left", padx=5)
        if self.model_configs:
            self.model_combobox.current(0)
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_select)

        add_model_btn = tk.Button(model_select_sub, text="➕ 신규 추가", bg="#198754", fg="white", activebackground="#157347", activeforeground="white", font=self.font_small, relief="flat", cursor="hand2", command=self.open_add_model_dialog)
        add_model_btn.pack(side="left", padx=6, ipadx=6, ipady=3)

        del_model_btn = tk.Button(model_select_sub, text="🗑️ 삭제", bg="#dc3545", fg="white", activebackground="#bb2d3b", activeforeground="white", font=self.font_small, relief="flat", cursor="hand2", command=self.delete_current_model)
        del_model_btn.pack(side="left", padx=2, ipadx=6, ipady=3)

        # 하단: 선택된 모델의 최종 출력 프롬프트 설정 영역
        prompt_frame = tk.LabelFrame(main_frame, text=" 📝 선택된 모델 ID별 최종 출력 프롬프트 설정 ", font=self.font_bold, fg="#198754", bg="#ffffff", padx=12, pady=10)
        prompt_frame.pack(fill="both", expand=True, pady=5)

        tk.Label(prompt_frame, text="해당 모델 ID로 RAG 추론을 수행할 때 적용될 시스템 프롬프트(지침)입니다:", font=self.font_small, fg="#6c757d", bg="#ffffff", anchor="w").pack(fill="x", pady=(0, 5))

        # 텍스트 위젯 내부에 여백과 가독성 확보
        text_container = tk.Frame(prompt_frame, bg="#ffffff", bd=1, relief="solid")
        text_container.pack(fill="both", expand=True, pady=5)

        self.prompt_text_widget = tk.Text(text_container, height=14, font=self.font_regular, fg="#212529", bg="#fdfdfe", wrap="word", bd=0, padx=8, pady=8)
        self.prompt_text_widget.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.prompt_text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        self.prompt_text_widget.configure(yscrollcommand=scrollbar.set)

        self.update_prompt_text_view()

    def on_model_select(self, event):
        selected_model = self.model_combobox.get()
        self.prompt_text_widget.delete("1.0", tk.END)
        self.prompt_text_widget.insert("1.0", self.model_configs.get(selected_model, ""))

    def update_prompt_text_view(self):
        selected_model = self.model_combobox.get()
        if selected_model in self.model_configs:
            self.prompt_text_widget.delete("1.0", tk.END)
            self.prompt_text_widget.insert("1.0", self.model_configs[selected_model])

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
            messagebox.showinfo("성공", "✅ 모든 환경 설정과 모델별 프롬프트가 성공적으로 저장되었습니다!")
        except Exception as e:
            messagebox.showerror("에러", f"설정 저장 실패:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedConfigApp(root)
    root.mainloop()