import logging
import plotly.graph_objects as go
import requests
import streamlit as st
from pages.Dataset import FASTAPI_HOST

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Модели", page_icon="", layout="wide")

st.title("Управление моделями")
tab1, tab2 = st.tabs(["📈 Кривые обучения", "📈 Сравнение кривых"])


def get_learning_curves(model_id):
    logger.info(f"Requesting learning curves for the model")
    response = requests.get(f"{FASTAPI_HOST}/get_learning_curves/{model_id}")
    if response.status_code == 200:
        learning_curves = response.json().get("learning_curves", {})
        logger.info(f"Learning curves have been successfully obtained")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=learning_curves["iterations"],
                y=learning_curves["train_rmse"],
                mode="lines",
                name="Train RMSE",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=learning_curves["iterations"],
                y=learning_curves["test_rmse"],
                mode="lines",
                name="Test RMSE",
            )
        )
        st.plotly_chart(fig)
    else:
        logger.error(f"Error in obtaining learning curves")
        st.error(f"Ошибка: {response.status_code}, {response.text}")


def compare_learning_curves(ids):
    logger.info(f"Comparison of learning curves for models {ids}")
    response = requests.post(f"{FASTAPI_HOST}/compare_learning_curves/", json={"model_ids": ids})
    if response.status_code == 200:
        data = response.json()
        learning_curves = data.get("learning_curves_comparison", {})
        logger.info(f"Learning curves have been successfully obtained")
        fig = go.Figure()
        for model_id, curves in learning_curves.items():
            iterations = curves["iterations"]
            train_rmse = curves["train_rmse"]
            test_rmse = curves["test_rmse"]
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=train_rmse,
                    mode="lines",
                    name=f"{model_id} Train RMSE",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=test_rmse,
                    mode="lines",
                    name=f"{model_id} Test RMSE",
                )
            )

            fig.update_layout(
                title="Сравнение кривых обучения",
                xaxis_title="Количесво итераций",
                yaxis_title="RMSE",
                legend_title="Модели",
            )

        st.plotly_chart(fig, use_container_width=True)

    else:
        logger.error(f"Error in obtaining learning curves")
        st.error(f"Ошибка: {response.status_code}, {response.text}")


with tab1:
    st.write("Посмотреть кривые обучения")
    container = st.container(border=True)
    model_id_cur = container.text_input("ID модели", key=7)
    if container.button("Посмотреть кривые обучения"):
        get_learning_curves(model_id_cur)


with tab2:
    st.write("На этой странице вы можете сравнивать кривые обучения")
    container = st.container(border=True)
    model_ids = container.text_area("Введите id моделей через запятую:", value="1, 2")
    model_ids = [model_id.strip() for model_id in model_ids.split(",")]
    if container.button("Сравнить"):
        compare_learning_curves(model_ids)
