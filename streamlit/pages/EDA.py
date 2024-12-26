import streamlit as st
from pages.Dataset import get_data_from_backend

st.set_page_config(
    page_title="Аналитика и EDA",# Название в меню
    page_icon="",# Иконка в меню
    layout="wide"
)

st.title("ℹ️ EDA Page")
st.write("Это страница 'Аналитики и EDA'.")
@st.cache_data
def get_cached_data():
    return get_data_from_backend()

df = get_cached_data()
if df is not None:
    st.write("Данные с сервера:", df)