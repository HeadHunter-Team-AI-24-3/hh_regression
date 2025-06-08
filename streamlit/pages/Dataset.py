import logging
import os

import pandas as pd
import streamlit as st
from utils import get_dataFrame, send_csv_to_backend, set_logo_md, FASTAPI_HOST, headers

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Датасет", page_icon="", layout="wide")  # Название в меню  # Иконка в меню

set_logo_md()

st.sidebar.header("Меню приложений")

if "df" not in st.session_state:
    get_dataFrame()

logger.info("Dataset page successfully opened")

st.title("Dataset Page")
st.header("Добро пожаловать на страницу где расположена информация по текущему датасету!")

st.subheader("Если у вас нет вашего датасета")
st.write(
    "Если у вас нет датасета, но вы хотите посмотреть работу нашего приложения, то можете воспользоваться нашим датасетом, который был сокращен до размера < 200 Mb."
)
if st.button("Использовать наш датасет"):
    file_path = os.path.abspath(".") + "/base_datassets/final_data_converted.csv"

    df = pd.read_csv(file_path)
    send_csv_to_backend(df)

st.subheader("Загрузка CSV файла на сервер")

uploaded_file = st.file_uploader("Выберите CSV файл для отправки", type=["csv"])

if uploaded_file is not None:
    logger.info("Try to upload file success")
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр данных из файла:")
    st.write("Первые 5 строк вашего файла:")
    st.write(df.head())
    if st.button("Отправить файл на сервер"):
        send_csv_to_backend(df)

if not st.session_state.df.empty:
    df = st.session_state.df

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
