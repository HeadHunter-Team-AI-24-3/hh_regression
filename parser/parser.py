import os
import time
from datetime import datetime
import requests
import pandas as pd

'''
Функция для парсинга вакансий с HH.
    - page (int): номер страницы, с которой начинается парсинг (необходимо для обхода блокировки по лимиту);
    - per_page (int): Количество вакансий на одной странице.
    - date_from (int): Количество вакансий на одной странице.
    - date_to (int): Количество вакансий на одной странице.
Выход - List Json-файлов с данными.
'''


def get_vacancies(date_from, date_to, page: int = 0, per_page: int = 100):
    params = {
        'page': page,  # Номер страницы, стандарт = 0
        'per_page': per_page,  # Количество ваков на одной стр
        'only_with_salary': True,  # Только вакансии с указанным доходом
        'date_from': date_from,  # Дата и время начала
        'date_to': date_to  # Дата и время конца
    }
    response = requests.get('https://api.hh.ru/vacancies', params=params)
    if response.status_code == 200:
        vacancies = response.json()
        return vacancies, response.status_code
    else:
        return None, response.status_code


def json_list_to_dataframe(json_list):
    def flatten(json_object, prefix=''):
        flat_dict = {}
        for key, value in json_object.items():
            if isinstance(value, dict):
                # Если это словарь, вызываем flatten рекурсивно
                flat_dict.update(flatten(value, prefix + key + '_'))
            elif isinstance(value, list):
                # Если это список, обрабатываем каждый элемент
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        # Обрабатываем вложенные словари
                        flat_dict.update(flatten(item, prefix + key + f'_{i}_'))
                    else:
                        # Обработка неисков в списке (меньше встречается в данном контексте)
                        flat_dict[prefix + key + f'_{i}'] = item
            else:
                # Базовый случай, добавляем значение в плоский словарь
                flat_dict[prefix + key] = value
        return flat_dict

    flat_list = [flatten(item) for item in json_list]
    df = pd.DataFrame(flat_list)
    return df


month_from = 9
month_to = 11
vacancies_list = []
current_vac_counter = 0
for month in range(month_from, month_to):
    day_from = 1
    day_to = 28 + 1
    if month == 9:
        day_to = 1 + 1
    elif month == 10:
        day_from = 1 + 1
    for day in range(day_from, day_to):
        current_page = 0

        while True:
            print(month, day)
            date_from = datetime(2024, month, day).isoformat()
            date_to = datetime(2024, month, day, 23, 59, 59).isoformat()

            data, status = get_vacancies(page=current_page,
                                         date_from=date_from,
                                         date_to=date_to)

            if status == 400:
                if current_page != 0:
                    print(f'Добыта информация из {current_page} страниц!')
                    print()
                    break
            elif status == 403:
                print(f'Парсинг прекращён по причине блокировки. Откат: 20 сек.')
                time.sleep(20)
                continue

            if data:
                for vacancy in data['items']:
                    vacancies_list.append(vacancy)
                current_page += 1
            else:
                break

            if len(vacancies_list) == current_vac_counter:
                break
            else:
                current_vac_counter = len(vacancies_list)

print("\nПарсинг завершен.")
print()
print('*' * 70)
print()

df = json_list_to_dataframe(vacancies_list)

print('*' * 70)
print()
print(f'Было найдено {df.shape[0]} вакансий с появлением {df.duplicated().sum()} строк-дубликатов.')
print(f'В каждой вакансии {df.shape[1]} переменных.')

if not os.path.exists('../dataset'):
    os.makedirs('../dataset')
df.to_csv('../dataset/vacancies.csv', index=False)
