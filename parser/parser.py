import os
import time
from datetime import datetime

import pandas as pd
import requests


class Parser:
    def __init__(self, url=None):
        if url is None:
            self.url = "https://api.hh.ru"
        else:
            self.url = url

    def get_vacancies_with_salary(self, date_from, date_to, area, metro, page: int = 0, per_page: int = 100):
        """
        Функция для парсинга вакансий с HH.
            - page (int): номер страницы, с которой начинается парсинг (необходимо для обхода блокировки по лимиту);
            - per_page (int): Количество вакансий на одной странице.
        Выход - List Json-файлов с данными.
        """

        params = {
            "text": "",  # Параметр запроса поиска. Пусто, чтобы смотреть все ваки
            "page": page,  # Номер страницы, стандарт = 0
            "per_page": per_page,  # Количество ваков на одной стр
            "only_with_salary": True,  # Только вакансии с указанным доходом
            "date_from": date_from,  # Дата и время начала
            "date_to": date_to,  # Дата и время конца
            "area": area,
            "metro": metro,
        }
        response = requests.get(self.url + "/vacancies", params=params)

        if response.status_code == 200:
            vacancies = response.json()
            return vacancies, response.status_code
        else:
            return None, response.status_code

    def json_list_to_dataframe(self, json_list):
        def flatten(json_object, prefix=""):
            flat_dict = {}
            for key, value in json_object.items():
                if isinstance(value, dict):
                    # Если это словарь, вызываем flatten рекурсивно
                    flat_dict.update(flatten(value, prefix + key + "_"))
                elif isinstance(value, list):
                    # Если это список, обрабатываем каждый элемент
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            # Обрабатываем вложенные словари
                            flat_dict.update(flatten(item, prefix + key + f"_{i}_"))
                        else:
                            # Обработка неисков в списке (меньше встречается в данном контексте)
                            flat_dict[prefix + key + f"_{i}"] = item
                else:
                    # Базовый случай, добавляем значение в плоский словарь
                    flat_dict[prefix + key] = value
            return flat_dict

        flat_list = [flatten(item) for item in json_list]
        df = pd.DataFrame(flat_list)
        return df

    def get_openapi_fields(self, handle):
        """Функция для получения всех возможных полей по ручке handle.
        - handle: str"""

        response = requests.get(f"{self.url}/{handle}")
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Ошибка при запросе справочника")
            return []

    def extract_area_names(self, areas):
        """Функция для парсинга регионов из JSON-файла."""
        names = []
        for area in areas:
            names.append(area["id"])
            if "areas" in area and area["areas"]:
                names.extend(self.extract_area_names(area["areas"]))
        if names:
            return names
        return [""]

    def extract_metro_data(self, metro_data):
        """Функция для создания Dict city -> List[metro] из JSON-файла."""
        metro_dict = {}

        for data in metro_data:
            city = data["id"]
            metro_dict[city] = [station["id"] for line in data["lines"] for station in line["stations"]]

        return metro_dict

    def __call__(self, month_from: int, month_to: int, day_from: int, day_to: int):
        # Спарсим наименования всех регионов поиска
        areas_data = self.get_openapi_fields("areas")
        area_names = self.extract_area_names(areas_data)

        # Спарсим наименования всех метро по конкретным регионам
        metro_data = self.get_openapi_fields("metro")
        metro_dict = self.extract_metro_data(metro_data)
        metro_areas = set(metro_dict.keys())

        # Парсинг данных
        vacancies_list = []
        current_vac_counter = 0
        sleep_time = 20

        for month in range(month_from, month_to):
            for day in range(day_from, day_to):
                print(f"Парсинг {day}/{month}/{day}...")
                for area in area_names:
                    metro_values = metro_areas if area in metro_areas else [""]
                    for metro in metro_values:
                        current_page = 0

                        while True:
                            date_from = datetime(2024, month, day).isoformat()
                            date_to = datetime(2024, month, day, 23, 59, 59).isoformat()

                            data, status = self.get_vacancies_with_salary(
                                page=current_page, date_from=date_from, date_to=date_to, area=area, metro=metro
                            )

                            if status == 400:
                                if current_page != 0:
                                    print(f"Парсинг данных на {date_from}, {area}, metro: {metro} завершён.")
                                    print(f"Добыта информация из {current_page} страниц!")
                                    print()
                                    break
                            elif status == 403:
                                print(f"Парсинг прекращён по причине блокировки. Откат: {sleep_time} сек.")
                                time.sleep(sleep_time)
                                continue

                            if data:
                                for vacancy in data["items"]:
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
        print("*" * 70)
        print()

        df = self.json_list_to_dataframe(vacancies_list)

        print("*" * 70)
        print()
        print(f"Было найдено {df.shape[0]} вакансий с появлением {df.duplicated().sum()} строк-дубликатов.")
        print(f"В каждой вакансии {df.shape[1]} переменных.")

        if not os.path.exists("../dataset"):
            os.makedirs("../dataset")
        df.to_csv("../dataset/vacancies.csv", index=False)

        return df
