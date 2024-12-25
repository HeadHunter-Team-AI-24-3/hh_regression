import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pickle

FASTAPI_HOST = "http://127.0.0.1:8000/"
headers = {'Content-Type': 'application/octet-stream'}

def main():
    st.title('Приложение: Предсказание ЗП на сайте HH.')

    st.subheader('Загрузите файл')
    uploaded_file = st.file_uploader('Загрузите файл CSV c историческими данными', type='csv')

    if uploaded_file is not None:
        api_url = FASTAPI_HOST + "upload_dataframe"

        df = pd.read_csv(uploaded_file)

        st.write("Первые 5 строк вашего файла:")
        st.write(df.head()) 

        df_serialized = pickle.dumps(df)

        try:
            response = requests.post(api_url, data=df_serialized, headers=headers)
            
            if response.status_code == 200:
                st.success('Файл успешно загружен в API')
                st.json(response.json())
            else:
                st.error(f"Ошибка при загрузке файла: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка соединения с API: {e}")

if __name__ == '__main__':
    main()