import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Умный Город — Классификатор Жалоб", layout="wide")
st.title("Классификатор обращений граждан")

# Вкладки
tab1, tab2, tab3 = st.tabs(["Предсказание", "Статистика датасета", "О проекте"])

with tab1:
    st.subheader("Введите текст жалобы")
    text_input = st.text_area("Текст обращения:", height=150, 
                              placeholder="Например: в подвале сильно воняет, судя по всему, там прорвало трубы!!!")
    
    if st.button("Определить категорию", type="primary"):
        if text_input.strip():
            with st.spinner("Модель думает..."):
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": text_input}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Категория:** {result['category']}")
                    st.info(f"**Уверенность:** {result['confidence']*100:.1f}%")
                    
                    # Вероятности
                    probs = pd.DataFrame(result['probabilities'].items(), columns=['Категория', 'Вероятность'])
                    probs = probs.sort_values('Вероятность', ascending=True)
                    fig = px.bar(probs, x='Вероятность', y='Категория', orientation='h', 
                                title="Вероятности по категориям")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Ошибка соединения с API")
        else:
            st.warning("Введите текст!")

with tab2:
    st.subheader("Статистика датасета")
    df = pd.read_csv("combined_dataset.csv")
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего обращений", len(df))
    col2.metric("Средняя длина текста", f"{df['clean_text_length'].mean():.0f} символов")
    col3.metric("Количество категорий", df['category'].nunique())
    
    st.plotly_chart(px.pie(df, names='category', title="Распределение по категориям"), 
                    use_container_width=True)

with tab3:
    st.markdown("""
    ### О проекте
    - **Модель:** LightGBM + TF-IDF
    - **Данные:** 62 438 обращений граждан
    - **API:** FastAPI
    - **Приложение:** Streamlit
    """)

# Запуск: streamlit run app.py