import json
import logging
import requests
import streamlit as st
from utils import FASTAPI_HOST

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Модель", page_icon="", layout="wide")

st.title("Страница модели")
tab1, tab2, tab3 = st.tabs(["✏️ Создание", "📁 Информация", "❌ Удаление"])


def train_model(model_id, model_name, hyperparameters):
    logger.info(f"Starting model training")
    param = {
        "model_id": model_id,
        "model_name": model_name,
        "hyperparameters": hyperparameters,
    }
    response = requests.post(f"{FASTAPI_HOST}/train_model", json=param)
    if response.status_code == 200:
        logger.info(f"The model has been successfully trained")
        st.success(f"Модель {model_name} успешно обучена!")
        st.json(response.json())
    else:
        logger.error(f"Error when training the model ")
        st.error(f"Ошибка: {response.status_code}, {response.text}")


def get_model_info(model_id):
    logger.info(f"Getting information about the model {model_id}")
    response = requests.get(f"{FASTAPI_HOST}/get_model_info/{model_id}")
    if response.status_code == 200:
        logger.info(f"Information about the model was successfully received")
        st.json(response.json())
    else:
        logger.error(f"Error when getting information about the model")
        st.error(f"Ошибка: {response.status_code}, {response.text}")


def list_all_models():
    logger.info("Requesting a list of all models")
    response = requests.get(f"{FASTAPI_HOST}/get_models_info")
    if response.status_code == 200:
        logger.info("The list of all models was successfully received")
        models = response.json()
        st.subheader("Список всех моделей:")
        for model in models:
            st.json(model)
            st.write("---")
    else:
        logger.error("An error occurred when getting the list of models.")
        st.error(f"Ошибка: {response.status_code}, {response.text}")


def delete_model(model_id):
    logger.info(f"Deleting a model {model_id}")
    response = requests.delete(f"{FASTAPI_HOST}/models/{model_id}")
    if response.status_code == 200:
        logger.info(f"Model successfully deleted")
        st.success(f"Модель {model_id} успешно удалена!")
    else:
        logger.error(f"Error deleting the model")
        st.error(f"Ошибка: {response.status_code}, {response.text}")


def delete_all_models():
    logger.info(f"Deleting all models")
    response = requests.delete(f"{FASTAPI_HOST}/models")
    if response.status_code == 200:
        logger.info(f"All models have been successfully deleted")
        st.success("Все модели успешно удалены!")
    else:
        logger.info(f"An error occurred when deleting models")
        st.error(f"Ошибка: {response.status_code}, {response.text}")


with tab1:
    st.write("Создание и редактирования моделей")
    container = st.container(border=True)
    model_id = container.text_input("ID модели", key=1)
    model_name = container.text_input("Название модели", key=2)
    hyperparameters = container.text_area(
        "Гиперпараметры", value='{"iterations": 100, "learning_rate": 0.1, "depth": 6}'
    )
    if container.button("Обучить"):
        train_model(model_id, model_name, json.loads(hyperparameters))

with tab2:
    st.write("Просмотр информации о модели")
    option = st.selectbox(
        "Информацию о скольки моделях хотите посмотреть?",
        ("Об одной", "О всех"),
        index=None,
        placeholder="Select",
    )

    if option == "Об одной":
        model_id_inf = st.text_input("ID модели", key=3)
        if st.button("Просмотр информации"):
            get_model_info(model_id_inf)
    elif option == "О всех":
        if st.button("Просмотр"):
            list_all_models()

with tab3:
    option = st.selectbox(
        "Сколько моделей хотите удалить?",
        ("Одну", "Все"),
        index=None,
        placeholder="Select",
    )

    if option == "Одну":
        model_id = st.text_input("Model ID")
        if st.button("Удалить модель"):
            delete_model(model_id)
    elif option == "Все":
        if st.button("Удалить все модели"):
            delete_all_models()
