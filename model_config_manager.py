import os
import json

PROMPTS_FILE_PATH = "model_prompts.json"

DEFAULT_MODEL_CONFIGS = {
    "3PL지식저장소": {
        "system_prompt": "3PL 물류 및 사내 통합 지식 보관소 데이터를 바탕으로 정확한 답변을 도출하는 팩트 검토 AI 보좌관입니다.\n\n[답변 가이드라인]\n1. 🔗 [RAG 우선 원칙] 아래 제공되는 [참조 지식 컨텍스트]에 명확히 명시된 팩트만을 기반으로 자연스럽고 읽기 쉽게 답변해야 합니다.\n2. 💡 [멀티모달 적극 활용] 내용 중 이미지가 있다면 적극적으로 보여주도록 합니다.\n3. 🚨 [최신성 및 정확성] 사내 데이터와 일반 지식을 조화롭게 융합하되, 팩트에 기반하여 신뢰도 높은 정보를 제공하세요.",
        "temperature": 0.0,
        "search_mode": "vector",
        "vector_search_limit": 10,
        "similarity_threshold": 0.5,
        "use_rerank": "True",
        "max_target_chunks": 5
    },
    "의약품스마트검색": {
        "system_prompt": "의약품 및 제약 데이터 전문 검색을 수행하며 신뢰도 높은 의약품 정보를 제공하는 전문 AI 보좌관입니다. 제품을 검색했으면 특장점과 복용지도를 같이 볼 수 있도록 제공합니다.\n\n[답변 가이드라인]\n1. 🔗 [RAG 우선 원칙] 아래 제공되는 [참조 지식 컨텍스트]에 사내 데이터(제품 정보, 복용지도 등)가 존재한다면, 해당 내용을 최우선으로 반영하여 상세히 설명하세요. \n2. 💡 [사전 지식 활용 허용] 만약 컨텍스트에 없는 일반 의약품 성분 효능, 작용 기전, 복약 상식 등에 대한 질문이 들어올 경우, 사용자가 풍부한 정보를 얻을 수 있도록 당신의 사전 지식을 적극적으로 동원하여 친절하고 정확하게 설명해 주세요.\n3. 🚨 [최신성 및 정확성] 사내 데이터와 일반 지식을 조화롭게 융합하되, 팩트에 기반하여 신뢰도 높은 정보를 제공하세요.",
        "temperature": 0.2,
        "search_mode": "vector",
        "vector_search_limit": 10,
        "similarity_threshold": 0.5,
        "use_rerank": "False",
        "max_target_chunks": 10
    }
}

def load_model_configs():
    """저장된 모델별 설정 파일을 안전하게 로드하고, 손상되거나 누락된 항목은 기본값으로 자동 방어"""
    try:
        if os.path.exists(PROMPTS_FILE_PATH):
            with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("파일 내용이 비어 있습니다.")
                data = json.loads(content)
                
                if isinstance(data, dict) and data:
                    migrated_data = {}
                    for k, v in data.items():
                        if isinstance(v, dict):
                            migrated_data[k] = {
                                "system_prompt": str(v.get("system_prompt", v.get("prompt", "시스템 프롬프트가 없습니다."))),
                                "temperature": float(v.get("temperature", 0.0)),
                                "search_mode": str(v.get("search_mode", "vector")),
                                "vector_search_limit": int(v.get("vector_search_limit", 10)),
                                "similarity_threshold": float(v.get("similarity_threshold", 0.35)),
                                "use_rerank": str(v.get("use_rerank", "True")),
                                "max_target_chunks": int(v.get("max_target_chunks", 5))
                            }
                        else:
                            # 구버전 형태(문자열)인 경우 대응
                            migrated_data[k] = {
                                "system_prompt": str(v),
                                "temperature": 0.0,
                                "search_mode": "vector",
                                "vector_search_limit": 10,
                                "similarity_threshold": 0.35,
                                "use_rerank": "True",
                                "max_target_chunks": 5
                            }
                    return migrated_data
    except Exception as e:
        print(f"⚠️ 모델 설정 파일 로드 중 오류 발생 (기본값으로 초기화합니다): {e}")
    
    # 예외 발생 시 안전하게 기본값 파일 생성 후 반환
    save_model_configs(DEFAULT_MODEL_CONFIGS)
    return DEFAULT_MODEL_CONFIGS

def save_model_configs(configs: dict):
    """모델별 설정 정보를 안전하게 JSON 파일에 저장"""
    try:
        with open(PROMPTS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(configs, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"❌ 모델 설정 파일 저장 실패: {e}")

def get_model_system_instruction(selected_model: str, default_instruction: str) -> str:
    """선택된 모델 ID에 해당하는 커스텀 시스템 프롬프트 반환"""
    try:
        configs = load_model_configs()
        conf = configs.get(selected_model)
        if isinstance(conf, dict):
            return conf.get("system_prompt", default_instruction)
        elif isinstance(conf, str):
            return conf
    except Exception:
        pass
    return default_instruction