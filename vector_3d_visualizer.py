import os
import re
import json
import psycopg
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ==========================================================
# 1. 환경 설정 및 데이터베이스 정보
# ==========================================================
MY_DATABASE_INFO = {
    "host": "localhost",
    "dbname": "redbombz",
    "user": "redbombz",
    "password": "a11223344*",
    "port": 5432
}

# ==========================================================
# 2. 로컬 Ollama 모델 및 임베딩 엔진 세팅
# ==========================================================
print("⚙️ 로컬 Ollama 엔진 및 bge-m3 임베딩 엔진 로드 중...")
llm = ChatOllama(base_url="http://localhost:11434", model="exaone3.5:7.8b", temperature=0)
embeddings_engine = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3")

query_splitter_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 사용자의 복합 질문을 분석하여, 검색에 필요한 독립된 단일 의도의 문장들로 분할하는 분석 엔진입니다.
    질문 내에 여러 정보(A정보, B정보 등)를 요구하는 맥락이 있다면 이를 명확한 개별 문장으로 나누어 JSON 배열로만 출력하세요.
    앞뒤 설명이나 마크다운 기호는 절대 붙이지 마세요.
    
    출력 양식 예시:
    {{
      "sub_queries": [
        "첫 번째 분할된 질문 문장",
        "두 번째 분할된 질문 문장"
      ]
    }}
    """),
    ("user", "분석할 사용자 질문:\n\n{user_query}")
])

# ==========================================================
# 3. 데이터베이스 데이터 수집 함수
# ==========================================================
def fetch_embedded_data():
    conn = None
    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        cur = conn.cursor()
        # RAG 검색 실시간 필터 조건인 is_deleted = 0 동기화 반영
        query = """
            SELECT chunk_id, reference_number, content, embedding 
            FROM tb_document_chunk 
            WHERE embedding IS NOT NULL AND is_deleted = 0
            ORDER BY chunk_id ASC;
        """
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        return rows
    except Exception as e:
        print(f"❌ DB 로드 실패: {e}")
        return []
    finally:
        if conn: conn.close()

# ==========================================================
# 4. 메인 분석 및 시각화 파이프라인
# ==========================================================
def main():
    print("\n" + "="*60)
    user_query = input("🔍 분석 및 3D 지식 공간에 투사할 복합 질문을 입력하세요:\n👉 ")
    if not user_query.strip():
        print("❌ 질문이 입력되지 않아 종료합니다.")
        return

    # [Step 2] LLM을 활용한 질문 분할
    print("\n🤖 EXAONE 3.5 기반 질문 분할 분석 중...")
    sub_queries = [user_query]
    try:
        chain = query_splitter_prompt | llm
        response = chain.invoke({"user_query": user_query})
        res_text = response.content.strip()
        
        json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            sub_queries = parsed.get("sub_queries", [user_query])
            
        print(f"➔ 🟢 질문이 {len(sub_queries)}개의 세부 의도로 분할되었습니다:")
        for idx, q in enumerate(sub_queries, start=1):
            print(f"   [{idx}] {q}")
    except Exception as e:
        print(f"   ⚠️ 질문 분할 실패 ({e}). 원본 질문 전체를 하나로 투사합니다.")

    # [Step 3] 분할된 질문들 bge-m3로 실시간 임베딩
    print("\n🔄 분할된 질문들의 bge-m3 벡터 추출 중...")
    query_vectors = []
    for q in sub_queries:
        query_vectors.append(embeddings_engine.embed_query(q.strip()))

    # [Step 4] 실제 pgvector 조회를 날려 질문별 Top 5에 픽업된 chunk_id 수집
    print("\n🎯 pgvector 시뮬레이션 돌려 실제 매칭될 상위 지식 추적 중...")
    retrieved_chunk_ids = set()
    try:
        conn = psycopg.connect(**MY_DATABASE_INFO)
        cur = conn.cursor()
        for q_vec in query_vectors:
            # 실제 RAG 백엔드 스캔 사양과 100% 동일하게 매칭 (LIMIT 5 장치 반영)
            search_query = """
                SELECT chunk_id FROM tb_document_chunk
                WHERE is_deleted = 0
                ORDER BY embedding <=> %s::vector ASC, chunk_id DESC
                LIMIT 5;
            """
            cur.execute(search_query, (q_vec,))
            for res in cur.fetchall():
                retrieved_chunk_ids.add(res[0])
        cur.close(); conn.close()
        print(f"   ➔ 🎯 현재 질문 분석 결과, 총 {len(retrieved_chunk_ids)}개의 지식 조각이 매칭 선택됩니다.")
    except Exception as db_err:
        print(f"   ❌ 유사도 스캔 중 오류 발생: {db_err}")

    # [Step 5] DB에서 기존 지식 데이터 로드
    print("\n📂 DB에서 통합 지식 자산 임베딩 테이블 로드 중...")
    raw_data = fetch_embedded_data()
    total_chunks = len(raw_data)
    print(f"📊 로드 완료: 기존 지식 파편 {total_chunks}개 확보.")

    if total_chunks < 4:
        print("⚠️ 데이터가 부족하여 3D 임베딩 벡터 공간을 구성할 수 없습니다.")
        return

    chunk_ids = []
    ref_numbers = []
    contents = []
    all_vectors = []
    data_types = []  # 상태 구조 정의 라벨 데이터셋

    # 1) 기존 데이터 파싱 및 픽업 여부 분류 처리
    for row in raw_data:
        chunk_id, ref_num, content, embedding_str = row
        if isinstance(embedding_str, str):
            clean_str = embedding_str.replace('[', '').replace(']', '')
            vector_array = np.fromstring(clean_str, sep=',')
        else:
            vector_array = np.array(embedding_str, dtype=np.float32)

        chunk_ids.append(chunk_id)
        ref_numbers.append(ref_num if ref_num else "N/A")
        contents.append(content[:80] + "..." if len(content) > 80 else content)
        all_vectors.append(vector_array)
        
        # 🎯 [단어 정제 반영] 픽업된 아이디 군집 분류 용어 통일
        if chunk_id in retrieved_chunk_ids:
            data_types.append("선택된 정보")
        else:
            data_types.append("저장된 정보")

    # 2) 사용자 세부 질문 벡터 결합
    for idx, q_vec in enumerate(query_vectors, start=1):
        chunk_ids.append(9990 + idx)
        ref_numbers.append(f"분석 질문 #{idx}")
        contents.append(sub_queries[idx-1])
        all_vectors.append(np.array(q_vec, dtype=np.float32))
        data_types.append("사용자 질문")

    X = np.array(all_vectors)

    # [Step 6] t-SNE 통합 차원 축소
    target_perplexity = min(30, max(2, len(X) // 3))
    print(f"\n🤖 t-SNE 통합 공간 압축 연산 시동... (총 요소: {len(X)}개)")
    tsne = TSNE(n_components=3, random_state=42, perplexity=target_perplexity, max_iter=1000)
    X_3d = tsne.fit_transform(X)
    print("➔ 🟢 3차원 공간 좌표 동기화 축소 성공.")

    # [Step 7] Plotly 레이어 3중 다이내믹 분리 렌더링
    print("🎨 3D 인터랙티브 디버거 공간 렌더링 시작...")
    
    unselected_sop_idx = [i for i, t in enumerate(data_types) if t == "저장된 정보"]
    selected_sop_idx = [i for i, t in enumerate(data_types) if t == "선택된 정보"]
    query_idx = [i for i, t in enumerate(data_types) if t == "사용자 질문"]

    # 1) 일반 저장된 지식 자산들 (차분한 아쿠아 블루 계열, 크기 축소 및 투명화로 배경 배치)
    trace_base_sop = go.Scatter3d(
        x=X_3d[unselected_sop_idx, 0], y=X_3d[unselected_sop_idx, 1], z=X_3d[unselected_sop_idx, 2],
        mode='markers',
        marker=dict(size=3.5, color='rgb(160, 195, 210)', opacity=0.35, line=dict(color='white', width=0.3)),
        name='보관된 사내 지식 자산',
        hoverinfo='text',
        hovertext=[f"<b>[보관 지식]</b><br>ID: {chunk_ids[i]}<br>태그: {ref_numbers[i]}<br>내용: {contents[i]}" for i in unselected_sop_idx]
    )

    # 2) 🎯 [크기 튜닝 및 용어 정제] 코사인 유사도로 픽업된 매칭 데이터 (기존 8 ➔ 5.5로 슬림화)
    trace_selected_sop = go.Scatter3d(
        x=X_3d[selected_sop_idx, 0], y=X_3d[selected_sop_idx, 1], z=X_3d[selected_sop_idx, 2],
        mode='markers+text',
        marker=dict(size=5.5, color='rgb(255, 140, 0)', opacity=0.95, line=dict(color='white', width=0.8)),
        text=np.array(ref_numbers)[selected_sop_idx], textposition="top center",
        textfont=dict(color='rgb(210, 85, 0)', size=10, family="Arial"),
        name='🎯 RAG 검색 매칭 지식 (Top 5)',
        hoverinfo='text',
        hovertext=[f"<b>🔥 [실시간 매칭 픽업 데이터]</b><br>ID: {chunk_ids[i]}<br>태그: {ref_numbers[i]}<br>내용: {contents[i]}" for i in selected_sop_idx]
    )

    # 3) 입력한 사용자의 분석 질문 조각 마커 (가장 강력한 빨간색 다이아몬드 지표 고정)
    trace_query = go.Scatter3d(
        x=X_3d[query_idx, 0], y=X_3d[query_idx, 1], z=X_3d[query_idx, 2],
        mode='markers+text',
        marker=dict(size=12, color='rgb(240, 0, 50)', symbol='diamond', opacity=1.0, line=dict(color='black', width=1.2)),
        text=np.array(ref_numbers)[query_idx], textposition="top center",
        textfont=dict(color='rgb(240, 0, 50)', size=11, family="Arial Black"),
        name='🚨 입력한 분석 질문 조각',
        hoverinfo='text',
        hovertext=[f"<b>🚨 [추적 질문]</b><br>라벨: {ref_numbers[i]}<br>의도: {contents[i]}" for i in query_idx]
    )

    layout = go.Layout(
        title=dict(text=f"🔴 bge-m3 기반 하이브리드 지식 추적형 3D 벡터 디버거 🔴<br><sup>대상 질문: {user_query[:50]}...</sup>", x=0.5, y=0.95),
        scene=dict(
            xaxis=dict(title='벡터 공간 X축', backgroundcolor="rgb(250, 250, 250)", gridcolor="rgba(0,0,0,0.08)"),
            yaxis=dict(title='벡터 공간 Y축', backgroundcolor="rgb(250, 250, 250)", gridcolor="rgba(0,0,0,0.08)"),
            zaxis=dict(title='벡터 공간 Z축', backgroundcolor="rgb(250, 250, 250)", gridcolor="rgba(0,0,0,0.08)")
        ),
        legend=dict(x=0.05, y=0.95, bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=60)
    )

    fig = go.Figure(data=[trace_base_sop, trace_selected_sop, trace_query], layout=layout)
    print("🚀 기본 웹 브라우저로 최적화된 하이브리드 3D 디버깅 뷰를 오픈합니다!")
    fig.show()

if __name__ == "__main__":
    main()