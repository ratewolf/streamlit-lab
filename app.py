import streamlit as st                      # Streamlit 을 st 라는 짧은 이름으로 씀

st.title("🎈 나의 첫 Streamlit 앱")          # 큰 제목
st.write("Streamlit 은 파이썬 코드만으로 웹 화면을 만듭니다.")   # 글 · 숫자 · 표 무엇이든 찍어 줌

nickname = st.text_input("별명을 입력하세요")  # 입력칸 — 입력한 글이 nickname 에 들어옴

if st.button("인사하기"):                      # 버튼 — 누른 그 순간에만 True
    st.success(f"{nickname} 님, 반가워요!")    # 초록색 알림 상자
