import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pickle

FASTAPI_HOST = "http://127.0.0.1:8000/"
headers = {'Content-Type': 'application/octet-stream'}
st.session_state.df = pd.DataFrame()

st.set_page_config(
    page_title="Датасет",   # Название в меню
    page_icon="",           # Иконка в меню
    layout="wide"
)
st.title("Dataset Page")
st.write("Добро пожаловать на страницу где расположена информация по текущему датасету!")

# Функция для отправки CSV файла на сервер
def send_csv_to_backend(data_frame):
    api_url = FASTAPI_HOST + "upload_dataframe"
    df_serialized = pickle.dumps(data_frame)
    try:
        response = requests.post(api_url, data=df_serialized, headers=headers)

        if response.status_code == 200:
            st.success('Файл успешно загружен в API')
            st.session_state.df = data_frame
            st.json(response.json())
        else:
            st.error(f"Ошибка при загрузке файла: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения с API: {e}")

st.write("Загрузка CSV файла на сервер")

uploaded_file = st.file_uploader("Выберите CSV файл для отправки", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр данных из файла:")
    st.write("Первые 5 строк вашего файла:")
    st.write(df.head())
    if st.button("Отправить файл на сервер"):
            send_csv_to_backend(df)

if not st.session_state.df.empty:
    st.dataframe(st.session_state.df)
    df = st.session_state.df
    st.write("Данные с сервера:", df)

    # Заголовок
    st.title("Текущий Датасет")

    # Отображение данных
    st.write("Датасет:")
    st.dataframe(df)

    # Информация о датасете
    st.write("Информация о датасете:")
    st.write(df.info())

    # Статистика
    st.write("Описательная статистика:")
    st.write(df.describe())

    # Пропущенные значения
    st.write("Пропущенные значения в датасете:")
    st.write(df.isnull().sum())