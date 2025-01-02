import logging
import base64
import os
import streamlit as st
import requests
import pickle

logger = logging.getLogger(__name__)

# Если запускаете через Docker то раскоментируйте нижню строку и закоментируйте строк №10
# FASTAPI_HOST = "http://fastapi:8000/"
FASTAPI_HOST = "http://127.0.0.1:8000/"
headers = {'Content-Type': 'application/octet-stream', 'User-Agent':'*'}

def set_logo_md():
    # Функция для преобразования изображения в base64
    def svg_to_base64(svg_file_path):
        with open(svg_file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Путь к вашему SVG файлу
    svg_path = os.path.join(os.getcwd(), '..', 'assets', 'README', 'Logo.svg')

    # Преобразуем SVG в base64
    svg_base64 = svg_to_base64(svg_path)

    # Создаем URL для base64 изображения
    svg_url = f"data:image/svg+xml;base64,{svg_base64}"

    # Вставляем CSS стили с помощью st.markdown
    st.markdown(f"""
        <style>
            /* Находим элемент с data-testid="stSidebarHeader" */
            [data-testid="stSidebarHeader"]::before {{
                content: '';
                display: block;
                width: 100px;  /* Установите желаемую ширину */
                height: 100px; /* Установите желаемую высоту */
                background-image: url('{svg_url}');
                background-size: contain;
                background-repeat: no-repeat;
            }}
        </style>
    """, unsafe_allow_html=True)

def send_csv_to_backend(data_frame):
    logger.info("Call send_csv_to_backend function")
    api_url = FASTAPI_HOST + "upload_dataframe"
    df_serialized = pickle.dumps(data_frame)
    try:
        logger.info("Call to upload_dataframe")
        response = requests.post(api_url, data=df_serialized, headers=headers)

        if response.status_code == 200:
            logger.info("Dataframe successfully uploaded")
            st.success('Файл успешно загружен в API')
            st.session_state.df = data_frame
            st.json(response.json())
        else:
            st.error(f"Ошибка при загрузке файла: {response.status_code}, {response.text}")
            logger.error(f"An error was received when uplodaing the dataframe: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения с API: {e}")
        logger.error(f"An error was received when connecting to API: {e}")

def get_dataFrame ():
    api_url = FASTAPI_HOST + "get_dataframe"
    try:
        logger.info("Call to get_dataframe api method")
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            serialized_data = bytes.fromhex(response.json()["df"])
            df = pickle.loads(serialized_data)
            st.toast("Датафрейм успешно получен с сервера")
            st.session_state.df = df
            logger.info("DataFrame successfully received")
        else:
            st.toast(f"Ошибка загрузки данных: {response.status_code}")
            logger.error(f"An error was received when receiving the dataframe: {response.status_code}")
            st.session_state.df = pd.DataFrame()

    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения с API: {e}")
        logger.error(f"An error was received when connecting to API: {e}")
        st.session_state.df = pd.DataFrame()