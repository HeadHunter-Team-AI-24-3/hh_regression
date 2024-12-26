from tqdm import tqdm
tqdm.pandas()
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

cat_columns = ['premium', 'has_test', 'response_letter_required', 'area_name', 'salary_currency', 'salary_gross', 'type_name', 'address_city', 'address_metro_station_name', 'address_metro_line_name', 'address_metro_stations_0_line_name', 'archived', 'employer_name', 'employer_accredited_it_employer', 'employer_trusted', 'schedule_name', 'accept_temporary', 'professional_roles_0_name', 'accept_incomplete_resumes', 'experience_name', 'employment_name', 'address_metro_stations_3_station_name', 'address_metro_stations_3_line_name', 'working_time_intervals_0_name', 'working_time_modes_0_name', 'working_days_0_name', 'branding_type', 'branding_tariff', 'department_name', 'insider_interview_id', 'brand_snippet_logo', 'brand_snippet_picture', 'brand_snippet_background_color', 'brand_snippet_background_gradient_angle', 'brand_snippet_background_gradient_color_list_0_position', 'brand_snippet_background_gradient_color_list_1_position', 'category']
text_columns = ['name', 'snippet_requirement', 'snippet_responsibility']
num_columns = ['name_length', 'length']

salary_currency_dict_to_RUR = {
    'RUR': 1,
    'USD': 96.09,
    'EUR': 104.4,
    'KZT': 0.198112,
    'BYR': 29.2,
    'UZS': 0.007498,
    'AZN': 56.52,
    'GEL': 35.38,
    'KGS': 1.12
}

categories = {
    'менеджер': 'Менеджмент',
    'руководитель': 'Менеджмент',
    'директор': 'Менеджмент',
    'координатор': 'Менеджмент',
    'управляющий': 'Менеджмент',

    'инженер': 'Инженерия',
    'конструктор': 'Инженерия',
    'технолог': 'Инженерия',
    'проектировщик': 'Инженерия',
    'разработчик': 'Инженерия',

    'продавец': 'Продажи',
    'менеджер по продажам': 'Продажи',
    'маркетолог': 'Маркетинг',
    'специалист по продажам': 'Продажи',
    'промоутер': 'Продажи',

    'бухгалтер': 'Финансы',
    'финансовый аналитик': 'Финансы',
    'экономист': 'Финансы',
    'контрактный управляющий': 'Финансы',

    'программист': 'IT',
    'аналитик': 'IT',
    'системный администратор': 'IT',
    'тестировщик': 'IT',

    'учитель': 'Образование',
    'воспитатель': 'Образование',
    'преподаватель': 'Образование',

    'врач': 'Здравоохранение',
    'медсестра': 'Здравоохранение',
    'фармацевт': 'Здравоохранение',

    'логист': 'Логистика',
    'кладовщик': 'Логистика',
    'курьер': 'Логистика',

    'работник': 'Производство',
    'сборщик': 'Производство',
    'оператор': 'Производство',
    'слесарь': 'Производство',
    'сварщик': 'Производство',

    'специалист': 'Служба поддержки',
    'консультант': 'Служба поддержки',
    'администратор': 'Служба поддержки',

    'художник': 'Искусство',
    'дизайнер': 'Искусство',
    'скульптор': 'Искусство',

    'помощник': 'Другие',
    'ассистент': 'Другие',
    'фасовщик': 'Другие',
}

def calculate_salary(row):
    if pd.isna(row['salary_from']):
        return row['salary_to']
    if pd.isna(row['salary_to']):
        return row['salary_from']
    return (row['salary_from'] + row['salary_to']) // 2

def categorize(name, categories):
    for key in categories:
        if key in name.lower():
            return categories[key]
    return 'Прочее'

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.dropna(axis=1, how='all')
    df.drop_duplicates(inplace=True)

    df['salary'] = df.progress_apply(calculate_salary, axis=1)
    for column in ['salary_from', 'salary_to']:
        del df[column]

    df['salary'] = df['salary'] * df['salary_currency'].map(salary_currency_dict_to_RUR)

    df['category'] = df['name'].progress_apply(lambda x: categorize(x, categories))
    df['name_length'] = df['name'].apply(len)
    df['length'] = df['snippet_requirement'].str.len()

    df.salary_gross.fillna(False, inplace=True)

    df['address_city'] = df['address_city'].replace('', np.nan)
    df['address_city'] = df['address_city'].fillna(df['area_name'])

    df['address_metro_station_name'] = df['address_metro_station_name'].fillna('UKN')
    df['address_metro_line_name'] = df['address_metro_line_name'].fillna('UKN')
    df['address_metro_stations_0_line_name'] = df['address_metro_stations_0_line_name'].fillna('UKN')
    df['employer_accredited_it_employer'].fillna('Unknown', inplace=True)
    df['address_metro_stations_3_station_name'] = df['address_metro_stations_3_station_name'].fillna('UKN')
    df['address_metro_stations_3_line_name'] = df['address_metro_stations_3_line_name'].fillna('UKN')

    df['working_time_intervals_0_name'] = df['working_time_intervals_0_name'].replace('Можно сменами по\xa04-6\xa0часов в\xa0день', 'Можно сменами по 4-6 часов в день')
    df['working_time_intervals_0_name'].fillna('Unknown', inplace=True)

    df['working_time_modes_0_name'] = df['working_time_modes_0_name'].replace('С\xa0началом дня после 16:00', 'С началом дня после 16:00')
    df['working_time_modes_0_name'].fillna('Unknown', inplace=True)

    df['working_days_0_name'] = df['working_days_0_name'].replace('По\xa0субботам и\xa0воскресеньям', 'По субботам и воскресеньям')
    df['working_days_0_name'].fillna('Unknown', inplace=True)

    df['branding_type'].fillna('Unknown', inplace=True)
    df['branding_tariff'].fillna('Unknown', inplace=True)
    df['department_name'].fillna('Unknown', inplace=True)

    df['insider_interview_id'] = df['insider_interview_id'].notna()
    df['brand_snippet_background_color'] = df['brand_snippet_background_color'].notna()
    df['brand_snippet_background_gradient_angle'] = df['brand_snippet_background_gradient_angle'].notna()
    df['brand_snippet_background_gradient_color_list_0_position'] = df['brand_snippet_background_gradient_color_list_0_position'].notna()
    df['brand_snippet_background_gradient_color_list_1_position'] = df['brand_snippet_background_gradient_color_list_1_position'].notna()
    df['brand_snippet_logo'] = df['brand_snippet_logo'].apply(lambda x: x if x == 'Unknown' else 'Other')
    df['brand_snippet_picture'] = df['brand_snippet_picture'].apply(lambda x: x if x == 'Unknown' else 'Other')

    df['length'] = df['length'].fillna(0)

    return df[cat_columns + num_columns + ['salary']]

def preprocess_data_for_model(df: pd.DataFrame) -> pd.DataFrame:

    df.reset_index(inplace=True, drop=True)

    scaler = StandardScaler()
    num_df = pd.DataFrame(scaler.fit_transform(df[num_columns]), columns=num_columns)

    label_columns = []
    ohe_columns = []

    for column in cat_columns:
        if df[column].nunique() > 10:
            label_columns.append(column)
        else:
            ohe_columns.append(column)

    to_bool = list(df[cat_columns].select_dtypes(include=['bool']).columns)
    df[['salary_gross', 'employer_accredited_it_employer']] = df[['salary_gross', 'employer_accredited_it_employer']].astype(bool).astype(int)
    df[to_bool] = df[to_bool].astype(int)

    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ohe_encoded = ohe.fit_transform(df[ohe_columns])
    ohe_feature_names = ohe.get_feature_names_out(ohe_columns).tolist()
    encoded_ohe_data = pd.DataFrame(ohe_encoded, columns=ohe_feature_names)

    label_encoder = LabelEncoder()
    for col in label_columns:
        df[col] = label_encoder.fit_transform(df[col])
    df[label_columns]

    merged_df = pd.concat([df[label_columns], encoded_ohe_data, num_df, df[['salary']]], axis=1)
    X = merged_df.drop(['salary'], axis=1)
    y = merged_df['salary']

    return train_test_split(X, y, test_size=0.2, random_state=12345)