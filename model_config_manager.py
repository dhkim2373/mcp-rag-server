import os
import json

PROMPTS_FILE_PATH = "model_prompts.json"

DEFAULT_MODEL_CONFIGS = {
    "3PL지식저장소": "3PL 물류 및 사내 통합 지식 보관소 데이터를 바탕으로 정확한 답변을 도출하는 팩트 검토 AI 보좌관입니다.",
    "의약품스마트검색": "의약품 및 제약 데이터 전문 검색을 수행하며 신뢰도 높은 의약품 정보를 제공하는 전문 AI 보좌관입니다."
}

def load_model_configs():
    """저장된 모델별 프롬프트 설정 파일을 로드 (없으면 기본값 생성)"""
    if os.path.exists(PROMPTS_FILE_PATH):
        try:
            with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return data
        except Exception as e:
            print(f"⚠️ 모델 설정 파일 로드 실패, 기본값 사용: {e}")
    
    save_model_configs(DEFAULT_MODEL_CONFIGS)
    return DEFAULT_MODEL_CONFIGS

def save_model_configs(configs: dict):
    """모델별 프롬프트 설정을 JSON 파일에 저장"""
    try:
        with open(PROMPTS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(configs, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"❌ 모델 설정 파일 저장 실패: {e}")

def get_model_system_instruction(selected_model: str, default_instruction: str) -> str:
    """선택된 모델 ID에 해당하는 커스텀 프롬프트 반환"""
    configs = load_model_configs()
    return configs.get(selected_model, default_instruction)