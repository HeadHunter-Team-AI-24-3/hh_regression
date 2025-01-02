import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy import stats
from scipy.stats import kstest, kruskal, ttest_ind
import altair as alt
import requests
import pickle
import os
import logging
from utils import *

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Аналитика и EDA",# Название в меню
    page_icon="",# Иконка в меню
    layout="wide"
)

set_logo_md()

# Если запускаете через Docker то раскоментируйте нижню строку и закоментируйте строк №10
# FASTAPI_HOST = "http://fastapi:8000/"
FASTAPI_HOST = "http://127.0.0.1:8000/"
headers = {'Content-Type': 'application/octet-stream', 'User-Agent':'*'}

if 'df' not in st.session_state:
    get_dataFrame()

logger.info("EDA page successfully opened")

st.title("ℹ️ EDA Page")
st.write("Это страница 'Аналитики и EDA'.")

if not st.session_state.df.empty:
    logger.info("Datasset isn't empty on server")
    init_df = st.session_state.df
    st.write("Датасет")
    st.write(init_df.head())

    if "salary" in init_df.columns:
        min_salary, max_salary = st.sidebar.slider(
            "Выберите диапазон зарплат:",
            int(init_df['salary'].min()),
            int(init_df['salary'].max()),
            (int(init_df['salary'].min()), int(init_df['salary'].max()))
        )

        df = init_df[(init_df['salary'] >= min_salary) & (init_df['salary'] <= max_salary)]
    else:
        df = init_df

    # Настройки графиков
    color_palette = st.sidebar.selectbox(
        "Выберите цветовую схему",
        options=["deep", "muted", "bright", "pastel", "dark", "colorblind"]
    )
    plt_color = st.sidebar.color_picker("Выберите цвет для графиков (основной)", "#3498db")
    plt_color_secondary = st.sidebar.color_picker("Выберите цвет для графиков (второстепенный)", "#FF8000")
    fig_width = st.sidebar.slider("Ширина графика", min_value=5, max_value=20, value=10)
    fig_height = st.sidebar.slider("Высота графика", min_value=3, max_value=15, value=6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    st.header("Постройте свой график")
    st.write("Выберите свой столбец и тип графика и постройте свой уникальный график")
    # Выбор колонок для графика
    x_col = st.selectbox("Выберите колонку для оси X", df.columns, index=0)
    y_col = st.selectbox("Выберите колонку для оси Y", df.columns, index=1)

    # Выбор типа графика
    chart_type = st.selectbox("Выберите тип графика", ["Линейный", "Столбчатый", "Диаграмма рассеяния"])

    # Настройки цвета графика
    color = st.color_picker("Выберите цвет графика", "#3498db")

    if x_col and y_col:
        # Построение графика
        fig_your_plot, ax_your_plot = plt.subplots(figsize=(fig_width, fig_height))

        if chart_type == "Линейный":
            ax_your_plot.plot(df[x_col], df[y_col], color=color, label=f"{y_col} vs {x_col}")
        elif chart_type == "Столбчатый":
            ax_your_plot.bar(df[x_col], df[y_col], color=color, label=f"{y_col} vs {x_col}")
        elif chart_type == "Диаграмма рассеяния":
            ax_your_plot.scatter(df[x_col], df[y_col], color=color, label=f"{y_col} vs {x_col}")

        # Настройки графика
        ax_your_plot.set_title(f"{chart_type} график")
        ax_your_plot.set_xlabel(x_col)
        ax_your_plot.set_ylabel(y_col)
        ax_your_plot.legend()

        # Отображение графика
        st.pyplot(fig_your_plot)

        st.subheader("Диаграмма разброса (Plotly)")
        color = st.color_picker("Выберите цвет точек", "#636EFA")  # Цвет для точек
        fig_scatter = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        st.plotly_chart(fig_scatter)

        st.subheader("Линейный график (Altair)")
        chart_type = st.radio("Тип линейного графика", ["Обычный", "Шаговый", "Область"])
        if chart_type == "Обычный":
            line_chart = alt.Chart(df).mark_line().encode(
                x=x_col,
                y=y_col,
                tooltip=[x_col, y_col]
            )
        elif chart_type == "Шаговый":
            line_chart = alt.Chart(df).mark_line(interpolate='step-after').encode(
                x=x_col,
                y=y_col,
                tooltip=[x_col, y_col]
            )
        else:  # Область
            line_chart = alt.Chart(df).mark_area().encode(
                x=x_col,
                y=y_col,
                tooltip=[x_col, y_col]
            )
        st.altair_chart(line_chart, use_container_width=True)

        st.subheader("Гистограмма (Plotly)")
        bins = st.slider("Количество интервалов (бинов)", min_value=5, max_value=50, value=20)
        fig_hist = px.histogram(df, x=x_col, nbins=bins, color_discrete_sequence=[color])
        st.plotly_chart(fig_hist)

        st.subheader("Парные диаграммы (Plotly)")
        selected_columns = st.multiselect("Выберите колонки для анализа", df.columns, default=df.columns[:3])
        if selected_columns:
            fig_pair = px.scatter_matrix(df, dimensions=selected_columns, color_discrete_sequence=[color])
            st.plotly_chart(fig_pair)

        st.subheader("Коробчатая диаграмма (Plotly)")
        fig_box = px.box(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        st.plotly_chart(fig_box)

    if "salary" not in df.columns:
        st.error("В датасете нет столбца 'salary'. Пожалуйста, загрузите корректный файл. Или воспользуйтесь построением своего графика из вашего датасета")
    else:
        st.subheader("Постройте свой график, но с нашим датасетом")
        chart_type = st.selectbox(
            "Выберите тип графика",
            options=["Scatterplot", "Boxplot", "Barplot"]
        )
        columns = [col for col in df.columns if col != "Зарплата"]
        selected_column = st.selectbox(
            "Выберите столбец для анализа",
            options=columns
        )
        st.subheader(f"График: {chart_type} - Зарплата vs {selected_column}")

        if chart_type == "Scatterplot":
            sns.scatterplot(data=df, x=selected_column, y="salary", palette=color_palette, ax=ax)
        elif chart_type == "Boxplot":
            sns.boxplot(data=df, x=selected_column, y="salary", palette=color_palette, ax=ax)
        elif chart_type == "Barplot":
            sns.barplot(data=df, x=selected_column, y="salary", palette=color_palette, ax=ax)

        st.pyplot(fig)

    if 'salary' in df:
        st.header("Столбец salary")

        fig, ax = plt.subplots(1, 3, figsize=(fig_width, fig_height))

        Q1 = df['salary'].quantile(0.25)
        Q3 = df['salary'].quantile(0.75)
        IQR = Q3 - Q1

        df_iqr = df[(df['salary'] >= Q1 - 1.5 * IQR) & (df['salary'] <= Q3 + 1.5 * IQR)]

        # Все зарплаты
        sns.histplot(df['salary'], bins=30, kde=True, ax=ax[0], palette=color_palette)
        ax[0].grid(True)
        ax[0].set_title('Все зарплаты')
        ax[0].set_xlabel('Зарплата')
        ax[0].set_ylabel('Частота')

        # Зарплаты < 1 000 000
        sns.histplot(df.loc[df['salary'] < 1000000, 'salary'], bins=30, kde=True, ax=ax[1], palette=color_palette)
        ax[1].grid(True)
        ax[1].set_title('Зарплаты < 1,000,000')
        ax[1].set_xlabel('Зарплата')
        ax[1].set_ylabel('Частота')

        # Зарплаты, обрезанные по IQR
        #plt.subplot(1, 3, 3)
        sns.histplot(df_iqr['salary'], bins=30, kde=True, ax=ax[2], palette=color_palette)
        ax[2].grid(True)
        ax[2].set_title('Зарплаты, обрезанные по IQR')
        ax[2].set_xlabel('Зарплата')
        ax[2].set_ylabel('Частота')

        st.pyplot(fig)

        salary = df['salary']
        mean = np.mean(salary)
        std = np.std(salary)
        stat, p_value = kstest(salary, 'norm', args=(mean, std))
        st.write(f'K-S Test Статистика: {stat}, p-value: {p_value}')

        st.write("Интерпретация результата")
        alpha = 0.05
        if p_value > alpha:
            st.write('Убедительных доказательств против гипотезы о нормальности нет (не отвергаем H0)')
        else:
            st.write('Гипотеза о нормальности отвергается (отвергаем H0)')

    if 'premium' in df:
        st.header("Столбец premium")
        import matplotlib.ticker as ticker
        st.subheader("Общая информация")
        st.write("Количество пропусков")
        st.write(df['premium'].isnull().sum())
        st.write("Уникальные значения")
        st.write(df['premium'].unique())
        premium_counts = df['premium'].value_counts()
        st.write(premium_counts)

        premium_salary_stats = df.groupby('premium')['salary'].describe()
        st.write(premium_salary_stats)

        average_salary_premium = df.groupby('premium')['salary'].mean()
        st.write(average_salary_premium)

        mean_salary_by_premium = df.groupby('premium')['salary'].mean().reset_index()

        overall_mean_salary = df['salary'].mean()

        mean_salary_by_premium['difference_from_overall'] = mean_salary_by_premium['salary'] - overall_mean_salary

        st.write(mean_salary_by_premium)

        fig_premium, ax_premium = plt.subplots(figsize=(fig_width, fig_height))

        # Создание столбчатой диаграммы для сравнения средних зарплат
        sns.barplot(x='premium', y='salary', data=mean_salary_by_premium, palette=color_palette, ax=ax_premium)

        ax_premium.set_title('Сравнение средних зарплат по значению premium')
        ax_premium.set_xlabel('Наличие премиума')
        ax_premium.set_ylabel('Средняя зарплата')
        ax_premium.axhline(y=overall_mean_salary, color='red', linestyle='--', label='Средняя зарплата всего датасета')
        ax_premium.legend()

        st.pyplot(fig_premium)

        st.subheader("Результаты")
        premium_groups = [group['salary'].values for name, group in df.groupby('premium')]
        f_statistic, p_value = stats.f_oneway(*premium_groups)

        st.write(f'F-статистика: {f_statistic}, p-значение: {p_value}')

    if 'has_test' in df:
        st.title("Столбец has_test")
        st.subheader("Общая информация")
        st.write("Количество пропусков")
        st.write(df['has_test'].isna().any())
        st.write("Уникальные значения")
        st.write(df['has_test'].unique())

        average_salary_true = df[df['has_test'] == True]['salary'].mean()
        count_true = df[df['has_test'] == True].shape[0]

        average_salary_false = df[df['has_test'] == False]['salary'].mean()
        count_false = df[df['has_test'] == False].shape[0]

        st.write(f"Средняя зарплата (has_test=True): {average_salary_true:.2f}, Количество строк: {count_true}")
        st.write(f"Средняя зарплата (has_test=False): {average_salary_false:.2f}, Количество строк: {count_false}")

        fig_has_test, ax_has_test = plt.subplots(1,2, figsize=(fig_width, fig_height))

        # Средняя зарплата по значениям столбца has_test
        sns.barplot(x='has_test', y='salary', data=df, ci=None, ax=ax_has_test[0], palette=color_palette)
        ax_has_test[0].set_title('Средняя зарплата по наличию теста (has_test)')
        ax_has_test[0].set_xlabel('Наличие теста')
        ax_has_test[0].set_ylabel('Средняя зарплата')
        ax_has_test[0].set_xticks([0, 1], ['Нет', 'Да'])
        ax_has_test[0].grid(axis='y')

        sns.countplot(x='has_test', data=df, ax=ax_has_test[1], palette=color_palette)
        ax_has_test[1].set_title('Количество строк по наличию теста (has_test)')
        ax_has_test[1].set_xlabel('Наличие теста')
        ax_has_test[1].set_ylabel('Количество строк')
        ax_has_test[1].set_xticks([0, 1], ['Нет', 'Да'])
        ax_has_test[1].grid(axis='y')

        st.subheader("Графики")
        st.pyplot(fig_has_test)

        st.subheader("Результаты")
        test_groups = [group['salary'].values for name, group in df.groupby('has_test')]
        f_statistic, p_value = stats.f_oneway(*test_groups)

        st.write(f'F-статистика: {f_statistic}, p-значение: {p_value}')

    if 'area_name' in df:
        st.title("Столбец area_name")
        st.subheader("Общая информация")
        st.write("Количество пропусков")
        st.write(df['area_name'].isna().sum())

        st.write("Уникальные населённые пункты:")
        st.write(df['area_name'].unique())
        with st.expander("Количество записей по каждому населенному пункту:"):
            st.write(df['area_name'].value_counts())

        salary_summary = df.groupby('area_name')['salary'].agg(['mean', 'median', 'min', 'max', 'std'])
        st.write("Статистика по зарплатам:")
        st.write(salary_summary)

        city_counts = df['area_name'].value_counts()

        cities_with_enough_data = city_counts[city_counts > 2000].index

        filtered_df = df[df['area_name'].isin(cities_with_enough_data)]

        mean_salary_by_city = filtered_df.groupby('area_name')['salary'].mean().sort_values()

        fig_area_name1, ax_area_name1 = plt.subplots(figsize=(fig_width, fig_height))
        mean_salary_by_city.plot(kind='barh', color=plt_color, ax=ax_area_name1)
        ax_area_name1.set_title('Средняя зарплата по населенным пунктам (более 2000 записей)')
        ax_area_name1.set_xlabel('Средняя зарплата')
        ax_area_name1.set_ylabel('Населённый пункт')
        st.pyplot(fig_area_name1)

        average_salary_by_city = filtered_df.groupby('area_name')['salary'].mean()

        overall_average_salary = df['salary'].mean()

        deviation = average_salary_by_city - overall_average_salary

        fig_area_name2, ax_area_name2 = plt.subplots(figsize=(fig_width, fig_height))
        deviation.sort_values().plot(kind='bar', color=plt_color, ax=ax_area_name2)
        ax_area_name2.axhline(0, color='red', linewidth=1, linestyle='--')  # линия нуля
        ax_area_name2.set_title('Отклонение средней зарплаты по городам (более 2000 записей) от общей средней зарплаты')
        ax_area_name2.set_xlabel('Город')
        ax_area_name2.set_ylabel('Отклонение средней зарплаты')
        st.pyplot(fig_area_name2)

        st.subheader("Результаты")
        group1 = df[df['area_name'] == 'Москва']['salary']
        group2 = df[df['area_name'] == 'Санкт-Петербург']['salary']
        t_stat, p_value = stats.ttest_ind(group1, group2)
        st.write(f"T-статистика: {t_stat}, p-значение: {p_value}")

    if 'address_metro_station_name' in df:
        st.title("Столбец address_metro_station_name")
        conditions = {
            'Есть метро': df['address_metro_station_name'] != 'Нет метро',
            'Не указана': df['address_metro_station_name'] == 'Не указана',
            'Нет метро': df['address_metro_station_name'] == 'Нет метро'
        }

        summary = {
            'Количество': [],
            'Средняя зарплата': []
        }

        for key, condition in conditions.items():
            count = len(df[condition])
            average_salary = df[condition]['salary'].mean() if count > 0 else 0
            summary['Количество'].append(count)
            summary['Средняя зарплата'].append(average_salary)

        summary_df = pd.DataFrame(summary, index=conditions.keys())

        with st.expander("Посмотреть DF"):
            st.write(summary_df)

        fig_metro_stations1, ax_metro_stations1 = plt.subplots(figsize=(fig_width, fig_height))
        summary_df[['Количество']].plot(kind='bar', legend=False, color=plt_color, ax=ax_metro_stations1)
        ax_metro_stations1.set_title('Количество станций по состоянию')
        ax_metro_stations1.set_xlabel('Состояние')
        ax_metro_stations1.set_ylabel('Количество')
        fig_metro_stations1.tight_layout()
        st.pyplot(fig_metro_stations1)

        fig_metro_stations2, ax_metro_stations2 = plt.subplots(figsize=(fig_width, fig_height))
        summary_df[['Средняя зарплата']].plot(kind='bar', legend=False, color=plt_color, ax=ax_metro_stations2)
        ax_metro_stations2.set_title('Средняя зарплата по состоянию станций')
        ax_metro_stations2.set_xlabel('Состояние')
        ax_metro_stations2.set_ylabel('Средняя зарплата')
        fig_metro_stations2.tight_layout()
        st.pyplot(fig_metro_stations2)

        filtered_df = df[(df['area_name'] == 'Москва') &
                         (df['address_metro_station_name'] != 'Нет метро') &
                         (df['address_metro_station_name'] != 'Станция не указана')]

        with st.expander("Посмотреть DF по Москве"):
            st.write(filtered_df)

        station_counts = filtered_df['address_metro_station_name'].value_counts().nlargest(5)
        average_salaries = filtered_df.groupby('address_metro_station_name')['salary'].mean()

        top_stations = pd.DataFrame({
            'Количество': station_counts,
            'Средняя зарплата': average_salaries[station_counts.index]
        })

        with st.expander("Посмотреть топ станций"):
            st.write(top_stations)

        fig_metro_stations3, ax_metro_stations3 = plt.subplots(figsize=(fig_width, fig_height))

        top_stations['Количество'].plot(kind='bar', color=plt_color, position=0, label='Количество', ax=ax_metro_stations3)
        ax_metro_stations3.set_ylabel('Количество', color=plt_color)
        ax_metro_stations3.tick_params(axis='y', labelcolor=plt_color)

        ax2 = ax_metro_stations3.twinx()
        top_stations['Средняя зарплата'].plot(kind='bar', ax=ax2, color=plt_color_secondary, position=1, label='Средняя зарплата')
        ax2.set_ylabel('Средняя зарплата', color=plt_color_secondary)
        ax2.tick_params(axis='y', labelcolor=plt_color_secondary)

        ax_metro_stations3.set_title('Топ-5 популярных станций метро в Москве')
        ax_metro_stations3.set_xlabel('Станция метро')
        ax_metro_stations3.set_xticklabels(top_stations.index, rotation=45)
        fig_metro_stations3.tight_layout()
        st.pyplot(fig_metro_stations3)

        filtered_df_all = df[(df['address_metro_station_name'] != 'Нет метро') & (
                    df['address_metro_station_name'] != 'Станция не указана')]
        top_stations_all = filtered_df_all['address_metro_station_name'].value_counts().nlargest(5)

        filtered_df_moscow = df[(df['area_name'] == 'Москва') & (df['address_city'] == 'Москва') &
                                (df['address_metro_station_name'] != 'Нет метро') &
                                (df['address_metro_station_name'] != 'Станция не указана')]
        top_salary_stations_moscow = filtered_df_moscow.groupby('address_metro_station_name')['salary'].mean().nlargest(
            5)

        with st.expander("Топ-5 самых популярных станций по количеству (все города):"):
            st.write(top_stations_all)
        with st.expander("Топ-5 станций по средней зарплате (только Москва):"):
            st.write(top_salary_stations_moscow)

        fig_metro_stations4, ax_metro_stations4 = plt.subplots(1, 2, figsize=(fig_width, fig_height))

        top_stations_all.plot(kind='bar', ax=ax_metro_stations4[0], color=plt_color)
        ax_metro_stations4[0].set_title('Топ-5 самых популярных станций (все города)')
        ax_metro_stations4[0].set_xlabel('Станция метро')
        ax_metro_stations4[0].set_ylabel('Количество упоминаний')

        top_salary_stations_moscow.plot(kind='bar', ax=ax_metro_stations4[1], color=plt_color_secondary)
        ax_metro_stations4[1].set_title('Топ-5 станций по средней зарплате (только Москва)')
        ax_metro_stations4[1].set_xlabel('Станция метро')
        ax_metro_stations4[1].set_ylabel('Средняя зарплата')

        fig_metro_stations4.tight_layout()
        st.pyplot(fig_metro_stations4)

        st.subheader("Результаты")
        street_groups = [group['salary'].values for name, group in df.groupby('address_metro_station_name')]
        f_statistic, p_value = stats.f_oneway(*street_groups)

        st.write(f'F-статистика: {f_statistic}, p-значение: {p_value}')

    if 'address_metro_line_name' in df:
        st.header("Столбец address_metro_line_name")
        conditions = {
            'Есть метро': df['address_metro_line_name'] != 'Нет метро',
            'Не указана': df['address_metro_line_name'] == 'Линия не указана',
            'Нет метро': df['address_metro_line_name'] == 'Нет метро'
        }

        summary = {
            'Количество': [],
            'Средняя зарплата': []
        }

        for key, condition in conditions.items():
            count = len(df[condition])
            average_salary = df[condition]['salary'].mean() if count > 0 else 0
            summary['Количество'].append(count)
            summary['Средняя зарплата'].append(average_salary)

        summary_df_lines = pd.DataFrame(summary, index=conditions.keys())

        with st.expander("Посмотреть DF"):
            st.write(summary_df_lines)

        fig_metro_lines1, ax_metro_lines1 = plt.subplots(figsize=(fig_width, fig_height))
        summary_df_lines[['Количество']].plot(kind='bar', legend=False, color=plt_color, ax=ax_metro_lines1)
        ax_metro_lines1.set_title('Количество станций по состоянию')
        ax_metro_lines1.set_xlabel('Состояние')
        ax_metro_lines1.set_ylabel('Количество')
        fig_metro_lines1.tight_layout()
        st.pyplot(fig_metro_lines1)

        fig_metro_lines2, ax_metro_lines2 = plt.subplots(figsize=(fig_width, fig_height))
        summary_df_lines[['Средняя зарплата']].plot(kind='bar', legend=False, color=plt_color, ax=ax_metro_lines2)
        ax_metro_lines2.set_title('Средняя зарплата по состоянию станций')
        ax_metro_lines2.set_xlabel('Состояние')
        ax_metro_lines2.set_ylabel('Средняя зарплата')
        fig_metro_lines2.tight_layout()
        st.pyplot(fig_metro_lines2)

        filtered_df_lines = df[(df['area_name'] == 'Москва') &
                         (df['address_metro_line_name'] != 'Нет метро') &
                         (df['address_metro_line_name'] != 'Линия не указана')]

        lines_counts = filtered_df_lines['address_metro_line_name'].value_counts().nlargest(5)
        average_salaries = filtered_df_lines.groupby('address_metro_line_name')['salary'].mean()

        top_lines = pd.DataFrame({
            'Количество': lines_counts,
            'Средняя зарплата': average_salaries[lines_counts.index]
        })

        st.write(top_lines)

        fig_metro_lines3, ax_metro_lines3 = plt.subplots(figsize=(fig_width, fig_height))

        top_lines['Количество'].plot(kind='bar', ax=ax_metro_lines3, color=plt_color, position=0, width=0.4, label='Количество')
        ax_metro_lines3.set_ylabel('Количество', color=plt_color)
        ax_metro_lines3.tick_params(axis='y', labelcolor=plt_color)

        ax2_lines = ax_metro_lines3.twinx()
        top_lines['Средняя зарплата'].plot(kind='bar', ax=ax2_lines, color=plt_color_secondary, position=1, width=0.4,
                                              label='Средняя зарплата')
        ax2_lines.set_ylabel('Средняя зарплата', color=plt_color_secondary)
        ax2_lines.tick_params(axis='y', labelcolor=plt_color_secondary)

        ax_metro_lines3.set_title('Топ-5 популярных линий метро в Москве')
        ax_metro_lines3.set_xlabel('Станция метро')
        ax_metro_lines3.set_xticklabels(top_lines.index, rotation=45)
        fig_metro_lines3.tight_layout()
        st.pyplot(fig_metro_lines3)

        filtered_df_all = df[
            (df['address_metro_line_name'] != 'Нет метро') & (df['address_metro_line_name'] != 'Линия не указана')]
        top_stations_all = filtered_df_all['address_metro_line_name'].value_counts().nlargest(5)

        filtered_df_moscow = df[(df['area_name'] == 'Москва') & (df['address_city'] == 'Москва') &
                                (df['address_metro_line_name'] != 'Нет метро') &
                                (df['address_metro_line_name'] != 'Линия не указана')]
        top_salary_stations_moscow = filtered_df_moscow.groupby('address_metro_line_name')['salary'].mean().nlargest(5)

        with st.expander("Топ-5 самых популярных линий по количеству (все города):"):
            st.write(top_stations_all)
        with st.expander("Топ-5 линий по средней зарплате (только Москва):"):
            st.write(top_salary_stations_moscow)

        fig_metro_lines4, ax_metro_lines4 = plt.subplots(1, 2, figsize=(fig_width, fig_height))

        top_stations_all.plot(kind='bar', ax=ax_metro_lines4[0], color=plt_color)
        ax_metro_lines4[0].set_title('Топ-5 самых популярных линий (все города)')
        ax_metro_lines4[0].set_xlabel('Линия метро')
        ax_metro_lines4[0].set_ylabel('Количество упоминаний')

        top_salary_stations_moscow.plot(kind='bar', ax=ax_metro_lines4[1], color=plt_color_secondary)
        ax_metro_lines4[1].set_title('Топ-5 линий по средней зарплате (только Москва)')
        ax_metro_lines4[1].set_xlabel('Линия метро')
        ax_metro_lines4[1].set_ylabel('Средняя зарплата')

        fig_metro_lines4.tight_layout()
        st.pyplot(fig_metro_lines4)

        street_groups = [group['salary'].values for name, group in df.groupby('address_metro_line_name')]
        f_statistic, p_value = stats.f_oneway(*street_groups)

        st.write(f'F-статистика: {f_statistic}, p-значение: {p_value}')

    if 'employer_accredited_it_employer' in df:
        st.header("Столбец employer_accredited_it_employer")
        employer_accredited_it_employer_palette = {True: 'green', False: 'red', 'Unknown': 'black'}
        fig_accredited_it, ax_accredited_it = plt.subplots(figsize=(fig_width, fig_height))

        sns.barplot(
            x=df['employer_accredited_it_employer'].value_counts(normalize=True).index,
            y=df['employer_accredited_it_employer'].value_counts(normalize=True),
            palette=color_palette,
            hue=df['employer_accredited_it_employer'].value_counts(normalize=True).index,
            legend=False,
            ax=ax_accredited_it
        )
        ax_accredited_it.set_title('Распределение аккредитованных работодателей', fontsize=14)
        ax_accredited_it.set_xlabel('Аккредитация работодателя')
        ax_accredited_it.set_ylabel('Статистика')

        for p in ax_accredited_it.patches:
            percentage = '{:.2f}%'.format(100 * p.get_height())
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax_accredited_it.annotate(percentage, (x, y), ha='center', va='bottom')

        st.pyplot(fig_accredited_it)

        st.subheader("Проверим группы на нормальность и гомогенность дисперсий, при положительном результате сделаем ANOVA тестирование")

        groupA = df.loc[df['employer_accredited_it_employer'] == "False", 'salary']
        groupB = df.loc[df['employer_accredited_it_employer'] == "True", 'salary']
        groupC = df.loc[df['employer_accredited_it_employer'] == 'Unknown', 'salary']


        def check_normality(group, name):
            stat, p_value = stats.shapiro(group)
            st.write(f"Группа {name} - Тест Шапиро-Уилка: Статистика = {stat:.4f}, p-значение = {p_value:.4f}")
            if p_value > 0.05:
                st.write(f"Группа {name}: Данные распределены нормально.")
            else:
                st.write(f"Группа {name}: Данные не распределены нормально.")


        check_normality(groupA, 'A')
        check_normality(groupB, 'B')
        check_normality(groupC, 'C')

        stat, p_value = stats.levene(groupA, groupB, groupC)
        st.write(f"\nТест Левена: Статистика = {stat:.4f}, p-значение = {p_value:.4f}")
        if p_value > 0.05:
            st.write("Дисперсии считать одинаковыми.")
        else:
            st.write("Дисперсии считать неодинаковыми.")

        st.subheader("Посмотрим на Крускала - Уоллиса - непараметрический тест для сравнения различий двух и более выборок.")

        h_stat, p_value = stats.kruskal(groupA, groupB, groupC)
        st.write(f'Kruskal-Wallis H-statistic: {h_stat}, P-value: {p_value}')

        if p_value < 0.05:
            st.write('Есть статистически значимые различия между группами')
        else:
            st.write('Нет статистически значимых различий между группами.')

        percentile_99 = df['salary'].quantile(0.999)  # Для наглядной визуализации возьмём salary < 99.9% выборки
        filtered_df = df[df['salary'] < percentile_99]

        with st.expander("Посмотреть отфильтрованный DF"):
            st.write(filtered_df)

        filtered_df['employer_accredited_it_employer_numeric'] = filtered_df['employer_accredited_it_employer'].map(
            {True: 1, False: 0, 'Unknown': -1})

        fig_accredited_it2, ax_accredited_it2 = plt.subplots(figsize=(fig_width, fig_height))
        sns.boxplot(x='employer_accredited_it_employer', y='salary', data=filtered_df, palette=color_palette, ax=ax_accredited_it2)
        ax_accredited_it2.set_title('Распределение заработной платы в зависимости от аккредитации работодателя')
        ax_accredited_it2.set_xlabel('Аккредитация работодателя')
        ax_accredited_it2.set_ylabel('Заработная плата')
        st.pyplot(fig_accredited_it2)

        means = [
            np.mean(groupA),
            np.mean(groupB),
            np.mean(groupC)
        ]

        std_errors = [
            stats.sem(groupA),
            stats.sem(groupB),
            stats.sem(groupC)
        ]

        groups = ['True', 'False', 'Unknown']

        fig_accredited_it3, ax_accredited_it3 = plt.subplots(figsize=(fig_width, fig_height))
        plt.bar(groups, means, yerr=std_errors, capsize=5, color=['green', 'red', 'black'])
        ax_accredited_it3.set_title('Средняя заработная плата по аккредитации работодателя с доверительными интервалами')
        ax_accredited_it3.set_xlabel('Аккредитация работодателя')
        ax_accredited_it3.set_ylabel('Средняя заработная плата')
        st.pyplot(fig_accredited_it3)

        st.subheader("Результаты")
        st.write("С аккредитацией: медиана равна 96.242 рублей, 50% данных находится в диапазоне от ~60к до ~ 180к, максимальное значение усов доходит до ~360к. \nЕсть выбросы; Без аккредитации: медиана равна 70.000 рублей, 50% данных находится в диапазоне от ~50к до ~ 110к, максимальное значение усов доходит до ~200к. \nЕсть выбросы; Без уточения аккредитации: медиана равна 110.000 рублей, 50% данных находится в диапазоне от ~80к до ~ 160к, максимальное значение усов доходит до ~360к. Есть выбросы. \nТест Крускала-Уоллиса доказал, что есть статистически значимые различия между тремя группами, также доверительные интервалы средних значений различны. Зависимость через корреляцию Phik не была выявлена.")

    if 'schedule_name' in df:
        st.header("Столбец schedule_name")
        count_series = df['schedule_name'].value_counts()
        percentage = count_series / count_series.sum() * 100
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
        explode = (0.1, 0, 0, 0, 0)

        # Переименуем индекс для лучшего отображения на графике
        percentage.index = [f'{i} ({p:.1f}%)' for i, p in zip(percentage.index, percentage)]

        fig_schedule_name1, ax_schedule_name1 = plt.subplots(figsize=(fig_width, fig_height))
        ax_schedule_name1.pie(percentage, labels=percentage.index, colors=colors, autopct='%1.1f%%', startangle=140,
                explode=explode, shadow=True)
        ax_schedule_name1.set_title('Процентное соотношение schedule_name', fontsize=14)
        ax_schedule_name1.axis('equal')

        st.pyplot(fig_schedule_name1)

        mean_salaries = df.groupby('schedule_name')['salary'].mean().sort_values(ascending=False)
        sorted_categories = mean_salaries.index

        with st.expander("mean_salaries"):
            st.write(mean_salaries)
            st.write(sorted_categories)

        fig_schedule_name2, ax_schedule_name2 = plt.subplots(figsize=(fig_width, fig_height))
        sns.barplot(x='schedule_name', y='salary', ax=ax_schedule_name2, data=df, order=sorted_categories, ci=95, palette=color_palette, capsize=.2)

        ax_schedule_name2.set_title(f'Средняя зарплата по schedule_name с доверительными интервалами')
        ax_schedule_name2.set_xlabel('График работы')
        ax_schedule_name2.set_ylabel('Средняя зарплата')
        ax_schedule_name2.tick_params(axis='x', labelrotation=90)

        st.pyplot(fig_schedule_name2)

        fig_schedule_name3, ax_schedule_name3 = plt.subplots(figsize=(fig_width, fig_height))
        sns.violinplot(
            x=df['schedule_name'],
            y=df['salary'],
            split=True,
            palette=color_palette,
            ax=ax_schedule_name3
        )
        st.pyplot(fig_schedule_name3)

        st.subheader("Результаты")
        st.write("Всего у нас 5 уникальных объектов:")
        st.write("Полный день. Самый частый график, который предлагает работодатель - 61,7%. При этом занимает 3 место по ЗП: медианное значение состовляет 70.000 рублей, 50% данных лежит в пределах 50к - 100к, max значение по усам достигает ~180к.")
        st.write("Сменный график. 17.6% от всех графиков. Последнее место по ЗП: медианное значение состовляет 62.010 рублей, 50% данных лежит в пределах 45.000 - 87.500, max значение по усам достигает ~140к.")
        st.write("Удаленная работа. Самый редкий график рабочего дня, который предлагает работодатель - 5.1%. Занимает 4 место по ЗП: медианное значение состовляет 58.500 рублей, 50% данных лежит в пределах 42.250 - 95.000, max значение по усам достигает ~160к.")
        st.write("Гибкий график. 7% от общего объёма вакансий. Занимает 2 место по ЗП: медианное значение состовляет 72.600 рублей, 50% данных лежит в пределах 43.400 - 120000, max значение по усам достигает ~240к.")
        st.write("Вахтовый метод. 8.6%. Очевидно 1 место по ЗП: медианное значение состовляет 155.000 рублей, 50% данных лежит в пределах 120.000 - 197.500, max значение по усам достигает ~320к.")


    if 'professional_roles_0_name' in df:
        st.header("Столбец professional_roles_0_name")
        nan_professional_roles = df['professional_roles_0_name'].isna().sum()
        st.write(f'Колчиество NaN объектов: {nan_professional_roles}')

        st.subheader("Графики")
        role_counts = df['professional_roles_0_name'].value_counts()

        top_roles = role_counts.index
        mean_salaries = df[df['professional_roles_0_name'].isin(top_roles[:20])].groupby('professional_roles_0_name')[
            'salary'].mean()

        role_counts = role_counts[:20]

        fig_prof_name1, ax_prof_name1 = plt.subplots(figsize=(fig_width, fig_height))
        barplot = sns.barplot(x=role_counts.values, y=role_counts.index, palette=color_palette, ax=ax_prof_name1)

        for index, value in enumerate(role_counts.index):
            mean_salary = mean_salaries[value]
            barplot.text(role_counts.values[index], index, f'Средняя ЗП: {mean_salary:.2f}', va='center', ha='left',
                         fontsize=9)

        ax_prof_name1.set_title('ТОП20 популярных профессий', fontsize=14)
        ax_prof_name1.set_xlabel('Количество', fontsize=9)
        ax_prof_name1.set_ylabel('Профессия')
        st.pyplot(fig_prof_name1)

        mean_salaries = df[df['professional_roles_0_name'].isin(top_roles[:20])].groupby('professional_roles_0_name')[
            'salary'].median()
        sorted_roles_by_salary = mean_salaries.sort_values(ascending=False)

        fig_prof_name2, ax_prof_name2 = plt.subplots(figsize=(fig_width, fig_height))
        barplot = sns.barplot(x=sorted_roles_by_salary.values, y=sorted_roles_by_salary.index, palette=color_palette, ax=ax_prof_name2)

        for index, value in enumerate(sorted_roles_by_salary.index):
            mean_salary = sorted_roles_by_salary[value]
            barplot.text(sorted_roles_by_salary.values[index], index, f'{mean_salary:.2f}', va='center', ha='left',
                         fontsize=9)

        ax_prof_name2.set_title('Медианная зарплата по профессиям ТОП20', fontsize=14)
        ax_prof_name2.set_xlabel('Медианная зарплата', fontsize=9)
        ax_prof_name2.set_ylabel('Профессия')
        st.pyplot(fig_prof_name2)

        mean_salaries = df[df['professional_roles_0_name'].isin(top_roles[:20])].groupby('professional_roles_0_name')[
            'salary'].mean()
        sorted_roles_by_salary = mean_salaries.sort_values(ascending=False)

        fig_prof_name3, ax_prof_name3 = plt.subplots(figsize=(fig_width, fig_height))
        barplot = sns.barplot(x=sorted_roles_by_salary.values, y=sorted_roles_by_salary.index, palette=color_palette, ax=ax_prof_name3)

        for index, value in enumerate(sorted_roles_by_salary.index):
            mean_salary = sorted_roles_by_salary[value]
            barplot.text(sorted_roles_by_salary.values[index], index, f'{mean_salary:.2f}', va='center', ha='left',
                         fontsize=9)

        ax_prof_name3.set_title('Средняя зарплата по профессиям ТОП20', fontsize=14)
        ax_prof_name3.set_xlabel('Средняя зарплата', fontsize=9)
        ax_prof_name3.set_ylabel('Профессиональная роль')
        st.pyplot(fig_prof_name3)

        mean_salaries = df[df['professional_roles_0_name'].isin(top_roles)].groupby('professional_roles_0_name')[
            'salary'].mean()
        sorted_roles_by_salary = mean_salaries.sort_values(ascending=True)[:20]

        fig_prof_name4, ax_prof_name4 = plt.subplots(figsize=(fig_width, fig_height))
        barplot = sns.barplot(x=sorted_roles_by_salary.values, y=sorted_roles_by_salary.index, palette=color_palette, ax=ax_prof_name4)

        for index, value in enumerate(sorted_roles_by_salary.index):
            mean_salary = sorted_roles_by_salary[value]
            barplot.text(sorted_roles_by_salary.values[index], index, f'{mean_salary:.2f}', va='center', ha='left',
                         fontsize=9)

        ax_prof_name4.set_title('Средняя зарплата по профессиям TAIL20', fontsize=14)
        ax_prof_name4.set_xlabel('Средняя зарплата', fontsize=9)
        ax_prof_name4.set_ylabel('Профессиональная роль')
        st.pyplot(fig_prof_name4)

        st.subheader("Тесты и результаты")
        salary_groups = [group['salary'].values for _, group in df.groupby('professional_roles_0_name')]
        statistic, p_value = kruskal(*salary_groups)

        st.write(f'Статистика Краскела-Уоллиса: {statistic}')
        st.write(f'P-value: {p_value}')

        if p_value < 0.05:
            st.write('Существует статистически значимая зависимость между профессией и зарплатой')
        else:
            st.write('Нет статистически значимой зависимости между профессиейи зарплатой')

        with st.expander("Выводы"):
            st.write("Медианные и среднии значения дают похожие результаты: самая большая ЗП у водителей, будь то машинисты или курьеры. Самая маленькая ЗП принадлежит работникам в сфере обслуживания: уборщик, дворник и т.п.Также был проведён тест Краскела-Уоллиса - непараметрический статистический тест, используемый для проверки значимости различий между тремя или более независимыми группами выборок. Он является обобщением критерия Манна-Уитни для более чем двух групп и применяется, когда соблюдение нормального распределения данных не может быть гарантировано. Опираясь на его результаты, можно с уверенностью сказать, что существуют статистически значимая зависимость между профессией и зарплатой.")

    if 'experience_name' in df:
        st.header("Столбец experience_name")
        st.subheader("Общая информация")
        st.write(df['experience_name'].info())

        st.subheader("Графики")
        fig_experience_name1, ax_experience_name1 = plt.subplots(figsize=(fig_width, fig_height))
        sns.barplot(y=df['experience_name'].index,
                    x=df['experience_name'], palette=color_palette, ax=ax_experience_name1)
        ax_experience_name1.set_title('Количество открытых вакансий по стажам')
        ax_experience_name1.set_xlabel('Количество')
        ax_experience_name1.set_ylabel('Стаж')
        st.pyplot(fig_experience_name1)

        fig_experience_name2, ax_experience_name2 = plt.subplots(figsize=(fig_width, fig_height))
        sns.violinplot(
            x=filtered_df['experience_name'],
            y=filtered_df['salary'],
            split=False,
            palette=color_palette,
            ax=ax_experience_name2
        )
        st.pyplot(fig_experience_name2)
        st.subheader("Тысты и результаты")
        salary_groups = [group['salary'].values for _, group in df.groupby('experience_name')]
        statistic, p_value = kruskal(*salary_groups)

        st.write(f'Статистика Краскела-Уоллиса: {statistic}')
        st.write(f'P-value: {p_value}')

        if p_value < 0.05:
            st.write('Существует статистически значимая зависимость между стажем и зарплатой')
        else:
            st.write('Нет статистически значимой зависимости между стажем и зарплатой')
        with st.expander("Выводы"):
            st.write("Тест Краскела-Уоллиса показал, что существуют статистически значимая зависимость между стажем и зарплатой. Нетрудно предположить - чем больше опыт работы, чем больше у тебя накопленных умений, знаний и понимания деятельности, тем больше тебе будут платить:")
            st.write("Нет опыта. Медиана равна 60.000, в то время, как первый и последний квантили бокса (25 и 75): 43.000 и 90.000 соотвественно.")
            st.write("От 1 года до 3 лет. Медиана находится в 80.000 рублей. Q1 равен 56.500, Q3 = 113.914.")
            st.write("От 3 до 6 лет. Значение медианы начинается подниматься 'на глазах', при данном опоте работы оно равно 110.000. Не отступают и квантили: 75.000 и 160.000 рублей.")
            st.write("Более 6 лет. Максимальное значение медианы, за опыт более 6 лет вы получите 130.000 рублей, 25 и 75 квантили находятся в диапазоне от 85000, до 200000.")

    if 'employment_name' in df:
        st.header("Столбец employment_name")

        st.subheader("Графики")

        fig_employment_name1, ax_employment_name1 = plt.subplots(figsize=(fig_width, fig_height))
        sns.barplot(y=df['employment_name'].index,
                    x=df['employment_name'], palette=color_palette, ax=ax_employment_name1)
        ax_employment_name1.set_title('Количество открытых вакансий по типу занятости')
        ax_employment_name1.set_xlabel('Количество')
        ax_employment_name1.tick_params(axis='x', labelrotation=90)
        ax_employment_name1.set_ylabel('Тип занятости')
        st.pyplot(fig_employment_name1)

        fig_employment_name2, ax_employment_name2 = plt.subplots(figsize=(fig_width, fig_height))
        sns.violinplot(
            x=filtered_df['employment_name'],
            y=filtered_df['salary'],
            split=False,
            palette=color_palette,
            ax=ax_employment_name2
        )
        st.pyplot(fig_employment_name2)

        st.subheader("Тесты и результаты")

        with st.expander("Выводы"):
            st.write("Полная занятость: Медиана равна 70.700. 25 и 75 перцентали равны 50000 и 108961, соответственно.")
            st.write("Частичная занятость: Медиана равна 65000. 25 и 75 перцентали равны 38547 и 116868, соответственно.")
            st.write("Проектная работа: Медиана равна 82000. 25 и 75 перцентали равны 50000 и 126500, соответственно.")
            st.write("Стажировка: Медиана равна 60000. 25 и 75 перцентали равны 45000 и 91860, соответственно.")
            st.write("Волонтерство: Медиана равна 65000. 25 и 75 перцентали равны 50000 и 102500, соответственно.")
            st.write("Как мы можем видеть, самым выгодным финансово типом занятости является проектная работа, самым 'дешёвым' типом из всех представленных - стажировка.")
else:
    logger.info("Datasset is empty on server")
    st.header("Датасет не загружен")
    st.subheader("Если у вас нет вашего датасета")
    st.write(
        "Если у вас нет датасета, но вы хотите посмотреть работу нашего приложения, то можете воспользоваться нашим датасетом, который был сокращен до размера < 200 Mb.")
    if st.button("Использовать наш датасет"):
        file_path = os.path.abspath('.') + "/base_datassets/final_data_converted.csv"

        # Чтение файла
        df = pd.read_csv(file_path)
        send_csv_to_backend(df)

    st.subheader("Загрузка CSV файла на сервер")

    uploaded_file = st.file_uploader("Выберите CSV файл для отправки", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Предпросмотр данных из файла:")
        st.write("Первые 5 строк вашего файла:")
        st.write(df.head())
        if st.button("Отправить файл на сервер"):
            send_csv_to_backend(df)

