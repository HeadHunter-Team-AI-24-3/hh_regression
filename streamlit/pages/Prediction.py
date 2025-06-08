import logging
import pickle

import pandas as pd
import requests
import streamlit as st
from utils import FASTAPI_HOST

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Предсказать",
    page_icon="",
    layout="wide",
)
st.title("Страница для выполнения предсказаний")
st.write("На этой странице можно выполнять предсказание с использованием обученных моделей")


def predict(model_id, input_data):
    logger.info(f"Prediction for the model {model_id}")
    df = pd.read_csv(input_data)
    df_serialized = pickle.dumps(df)
    response = requests.post(f"{FASTAPI_HOST}/predict/{model_id}", data=df_serialized)
    if response.status_code == 200:
        logger.info(f"Predictions were completed successfully")
        predictions = response.json().get("predictions", [])
        st.success(f"Предсказания для модели {model_id} выполнены успешно")
        st.write("Предсказания: ")
        st.write(predictions)
    else:
        logger.error(f"Error when making a prediction")
        st.error(f"Ошибка: {response.status_code}, {response.text}")


container = st.container(border=True)
model_id = container.text_input("ID модели")
uploaded_file = container.file_uploader("Загрузите CSV файл для предсказания", type=["csv"])
if container.button("Выполнить предсказание"):
    if not model_id:
        container.error("Укажите ID модели")
    elif not uploaded_file:
        container.error("Загрузите файл с данными")
    else:
        predict(model_id, uploaded_file)
