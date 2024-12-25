import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

FASTAPI_HOST = "http://localhost:8000"

def main():
    st.title('Приложение: Анализ Погоды')

    st.subheader('Загрузите файл')
    uploaded_file = st.file_uploader('Загрузите файл CSV c историческими данными', type='csv')

    if uploaded_file is not None:
        api_url = FASTAPI_HOST + "/dataset"

        try:
            response = requests.post(api_url, files={'file': uploaded_file})
            
            if response.status_code == 200:
                st.success('Файл успешно загружен в API')
            else:
                st.error(f'Ошибка при загрузке файла: {response.status_code}')
        except requests.exceptions.RequestException as e:
            st.error(f'Ошибка соединения с API: {e}')


if __name__ == '__main__':
    main()