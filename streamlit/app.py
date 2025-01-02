from utils import *
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/streamlit.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

set_logo_md()

st.title('Приложение: Предсказание ЗП на сайте HH.')

st.subheader("Описание проекта")
st.write("Проект направлен на анализ вакансий с целью предсказания зарплат и выявления ключевых навыков. Он включает в себя автоматизированный сбор данных с сайтов вакансий, использование моделей машинного и глубокого обучения для предсказания зарплат по описаниям позиций и анализ востребованных навыков. Также проект предусматривает визуализацию данных и анализ динамики изменений зарплат и навыков во времени.")

st.subheader("Участники проекта")

st.markdown(f"Аладинский Георгий Александрович [(@gogaTheBest)](https://t.me/gogaTheBest)")
st.markdown(f"Дмитрий Тапанович Мандал [(@dimatrp)](https://t.me/dimatrp)")
st.markdown(f"Панов Артём Сергеевич [(@arsepan)](https://t.me/arsepan)")
st.markdown(f"Больбот Елизавета Владимировна [(@piv_liker)](https://t.me/piv_liker)")

st.subheader("Весь проект")
st.markdown(f"Сам проект лежит на [гитхабе](https://github.com/HeadHunter-Team-AI-24-3/hh_regression)")