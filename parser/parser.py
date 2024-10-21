import time
from datetime import datetime
import requests

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


month_from = 9
month_to = 11
vacancies_list = []
current_vac_counter = 0
for month in range(month_from, month_to):
    day_from = 1
    day_to = 28 + 1
    if month == 9:
        day_to = 30 + 1
    elif month == 10:
        day_from = 31 + 1
    for day in range(day_from, day_to):
        current_page = 0

        while True:
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
