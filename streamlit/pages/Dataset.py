import requests
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Датасет",   # Название в меню
    page_icon="",           # Иконка в меню
    layout="wide"
)
st.title("Dataset Page")
st.write("Добро пожаловать на страницу где расположена информация по текущему датасету!")


# Функция для получения данных с бэкенда
@st.cache_data
def get_data_from_backend():
    # url = "https://your-backend-api.com/get-data"
    # response = requests.get(url)
    #
    # if response.status_code == 200:
    #     data = response.json()
    #     df = pd.DataFrame(data)
    #     return df
    # else:
    #     st.error("Ошибка при получении данных с сервера.")
    #     return None

    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Age": [25, 30, 35, 40, 45],
        "Salary": [50000, 55000, 60000, 65000, 70000]
    }
    df = pd.DataFrame(data)
    return df

# Запрашиваем данные при открытии страницы
df = get_data_from_backend()


# Функция для отправки CSV файла на сервер
def send_csv_to_backend(csv_file):
    st.success('Файл отправлен')
    # url = "https://your-backend-api.com/upload"
    # files = {"file": csv_file}
    #
    # response = requests.post(url, files=files)
    #
    # if response.status_code == 200:
    #     st.success("CSV файл успешно отправлен!")
    # else:
    #     st.error(f"Ошибка при отправке файла. Код ошибки: {response.status_code}")

st.write("Загрузка CSV файла на сервер")

uploaded_file = st.file_uploader("Выберите CSV файл для отправки", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр данных из файла:")
    st.dataframe(df)
    if st.button("Отправить файл на сервер"):
        send_csv_to_backend(uploaded_file)

if df is not None:
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