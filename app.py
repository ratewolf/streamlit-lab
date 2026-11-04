import streamlit as st

st.title("🍱 점심 메뉴 고르기")

# ---------- 사이드바 : 화면 왼쪽 접히는 칸 ----------
st.sidebar.header("조건")
budget = st.sidebar.slider("예산 (원)", min_value=5000, max_value=15000, value=10000, step=1000)
spicy = st.sidebar.checkbox("매운 음식 괜찮음")

# ---------- 본문 ----------
category = st.selectbox("종류", ["한식", "분식", "중식", "양식"])
menus = {
    "한식": [("김치찌개", 8000, True), ("비빔밥", 9000, False), ("제육볶음", 9500, True)],
    "분식": [("떡볶이", 5000, True), ("김밥", 5000, False), ("라면", 6000, True)],
    "중식": [("짜장면", 7000, False), ("짬뽕", 8000, True), ("볶음밥", 8000, False)],
    "양식": [("토마토 파스타", 12000, False), ("돈가스", 10000, False), ("매운 크림 파스타", 13000, True)],
}

# 조건에 맞는 메뉴만 남김 — 예산 이하이고, 매운 음식이 싫으면 매운 메뉴는 뺌
choices = [name for name, price, is_spicy in menus[category] if price <= budget and (spicy or not is_spicy)]

left, right = st.columns(2)                 # 화면을 두 칸으로 나눔
left.metric("조건에 맞는 메뉴", f"{len(choices)}개")
right.metric("예산", f"{budget:,}원")

if choices:
    st.write("추천 :", " · ".join(choices))
else:
    st.warning("조건에 맞는 메뉴가 없습니다. 예산을 올려 보세요.")
