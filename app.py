import streamlit as st
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"


# ---------- ① client 만들기 : 키는 st.secrets 에서 읽음 (코드에 적지 않음) ----------
@st.cache_resource                          # 처음 한 번만 만들고, 다시 실행될 때는 만들어 둔 것을 그대로 씀
def get_client():
    """Gemini client 를 만들어 돌려준다"""
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=120_000,                                   # 2분 안에 답이 없으면 멈춤
            retry_options=types.HttpRetryOptions(http_status_codes=[500, 503]),   # 429 는 말없이 기다리지 않게
        ),
    )


# ---------- 키가 있는지 먼저 확인 ----------
try:
    client = get_client()
except Exception:
    st.error("🔑 GEMINI_API_KEY 를 찾지 못했습니다 — Manage app → ⋮ → Settings → Secrets 칸을 확인하세요.")
    st.stop()                               # 여기서 화면 그리기를 멈춤

# ---------- ② 화면 + 요청 ----------
st.title("📝 AI 용어 설명기")
st.caption(f"모델 : {MODEL_NAME}")

style = st.selectbox("말투", ["초등학생도 알아듣게 쉽게", "전공 수업처럼 정확하게", "재미있는 비유를 들어"])
word = st.text_input("궁금한 용어", placeholder="예: API · 토큰 · 시스템 프롬프트")

if st.button("설명 듣기", disabled=(word.strip() == "")):        # 용어를 넣기 전에는 버튼이 꺼져 있음
    with st.spinner("Gemini 가 답을 쓰는 중..."):                 # 기다리는 동안 빙글빙글 표시
        try:
            reply = client.interactions.create(
                model=MODEL_NAME,
                store=False,
                system_instruction=f"너는 AI 용어 설명 도우미야. {style} 3문장 이내로 설명한다.",
                input=word,
            )
        except Exception as error:
            st.error(error)                       # 우선 오류를 그대로 보여 줌 (2-3 에서 한글로 바꿈)
        else:
            st.markdown(reply.output_text)                       # ** 굵게 같은 마크다운도 그대로 보여 줌
            st.caption(f"이번 요청 토큰 : {reply.usage.total_tokens}")
