import os
import re
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="ИИ-поиск поставщиков", layout="wide")

APP_DIR = Path(__file__).resolve().parent

REQUIRED_COLUMNS = [
    "id",
    "name",
    "category",
    "products",
    "city",
    "region",
    "min_order",
    "price_level",
    "delivery",
    "certificates",
    "website",
    "email",
    "phone",
    "comment",
]

DISPLAY_COLUMNS = [
    "score",
    "name",
    "category",
    "products",
    "city",
    "region",
    "min_order",
    "price_level",
    "delivery",
    "certificates",
    "website",
    "email",
    "phone",
    "comment",
]

ENV_PATHS = [
    Path("/env/local"),
    Path("env/local"),
    Path(".env/local"),
    Path(".env.local"),     
]

PLACEHOLDER_VALUES = {
    "",
    "AgentUrl",
    "ProjectId",
    "AgentId",
    "YaApiKey",
    "YANDEX_API_KEY",
    "YANDEX_AGENT_URL",
    "YANDEX_PROJECT_ID",
    "YANDEX_AGENT_ID",
}


def clean_env_value(value):
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


@st.cache_data(show_spinner=False)
def load_local_env_values():
    """Читает переменные из /env/local, env/local, .env/local или .env без вывода секретов в интерфейс."""
    values = {}
    for path in ENV_PATHS:
        if not path.exists() or not path.is_file():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = clean_env_value(value)
                if key and value:
                    values[key] = value
        except Exception:
            pass
    return values


def secret_value(names, default=""):
    """Берет секрет сначала из переменных окружения, потом из env/local, потом из st.secrets."""
    if isinstance(names, str):
        names = [names]

    for name in names:
        value = os.getenv(name, "")
        if value:
            return clean_env_value(value)

    local_values = load_local_env_values()
    for name in names:
        value = local_values.get(name, "")
        if value:
            return clean_env_value(value)

    try:
        for name in names:
            value = st.secrets.get(name, "")
            if value:
                return clean_env_value(value)
    except Exception:
        pass

    return default


def is_real_value(value):
    return clean_env_value(value) not in PLACEHOLDER_VALUES


@st.cache_data(show_spinner=False)
def load_default_csv():
    """Автоматически загружает CSV с поставщиками из проекта без окна загрузки в интерфейсе."""
    paths = [
        APP_DIR / "suppliers.csv",
        APP_DIR / "suppliers(3).csv",
        APP_DIR / "data" / "suppliers.csv",
        Path("suppliers.csv"),
        Path("suppliers(3).csv"),
        Path("data") / "suppliers.csv",
        Path("/mnt/data/suppliers.csv"),
        Path("/mnt/data/suppliers(3).csv"),
    ]
    for path in paths:
        if path.exists() and path.is_file():
            return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()



def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def split_words(value):
    return [word for word in re.split(r"[^а-яА-Яa-zA-Z0-9]+", normalize_text(value)) if len(word) > 1]


def prepare_data(df):
    df = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[REQUIRED_COLUMNS]
    for column in df.columns:
        if column != "min_order":
            df[column] = df[column].fillna("").astype(str).str.strip()
    df["min_order"] = pd.to_numeric(df["min_order"], errors="coerce").fillna(0).astype(int)
    df["search_text"] = (
        df["name"]
        + " "
        + df["category"]
        + " "
        + df["products"]
        + " "
        + df["city"]
        + " "
        + df["region"]
        + " "
        + df["certificates"]
        + " "
        + df["comment"]
    ).map(normalize_text)
    return df


def unique_sorted(df, column):
    return sorted([value for value in df[column].dropna().astype(str).str.strip().unique() if value])


def apply_filters(df, query, category, city, region, price_levels, delivery, certificate, max_min_order):
    result = df.copy()
    if query.strip():
        words = split_words(query)
        if words:
            mask = result["search_text"].apply(lambda text: all(word in text for word in words))
            if not mask.any():
                mask = result["search_text"].apply(lambda text: any(word in text for word in words))
            result = result[mask]
    if category != "Все":
        result = result[result["category"].map(normalize_text) == normalize_text(category)]
    if city != "Все":
        result = result[result["city"].map(normalize_text) == normalize_text(city)]
    if region != "Все":
        region_norm = normalize_text(region)
        result = result[
            (result["region"].map(normalize_text) == region_norm)
            | (result["region"].map(normalize_text) == "вся россия")
        ]
    if price_levels:
        normalized_prices = {normalize_text(value) for value in price_levels}
        result = result[result["price_level"].map(normalize_text).isin(normalized_prices)]
    if delivery != "Любая":
        result = result[result["delivery"].map(normalize_text) == normalize_text(delivery)]
    if certificate.strip():
        cert_words = split_words(certificate)
        result = result[
            result["certificates"].map(normalize_text).apply(lambda text: all(word in text for word in cert_words))
        ]
    if max_min_order > 0:
        result = result[result["min_order"] <= max_min_order]
    return result


def add_scores(df, query, category, city, region, certificate):
    result = df.copy()
    if result.empty:
        result["score"] = []
        return result
    query_words = split_words(query)
    cert_words = split_words(certificate)
    max_order = max(result["min_order"].max(), 1)
    scores = []
    for _, row in result.iterrows():
        score = 50
        category_text = normalize_text(row["category"])
        products_text = normalize_text(row["products"])
        name_text = normalize_text(row["name"])
        city_text = normalize_text(row["city"])
        region_text = normalize_text(row["region"])
        certificates_text = normalize_text(row["certificates"])
        comment_text = normalize_text(row["comment"])
        for word in query_words:
            if word in category_text:
                score += 12
            if word in products_text:
                score += 12
            if word in name_text:
                score += 8
            if word in city_text or word in region_text:
                score += 6
            if word in certificates_text or word in comment_text:
                score += 4
        if category != "Все" and normalize_text(category) == category_text:
            score += 20
        if city != "Все" and normalize_text(city) == city_text:
            score += 18
        if region != "Все" and normalize_text(region) == region_text:
            score += 16
        if region != "Все" and region_text == "вся россия":
            score += 12
        if normalize_text(row["price_level"]) == "низкая":
            score += 16
        if normalize_text(row["price_level"]) == "средняя":
            score += 8
        if normalize_text(row["delivery"]) == "есть":
            score += 14
        if normalize_text(row["delivery"]) == "по согласованию":
            score += 6
        if "нет данных" not in certificates_text and certificates_text:
            score += 12
        if cert_words and all(word in certificates_text for word in cert_words):
            score += 25
        score += int((1 - min(row["min_order"], max_order) / max_order) * 20)
        scores.append(score)
    result["score"] = scores
    return result.sort_values(["score", "min_order"], ascending=[False, True])


def records_for_ai(df, limit=15):
    columns = [
        "id",
        "name",
        "category",
        "products",
        "city",
        "region",
        "min_order",
        "price_level",
        "delivery",
        "certificates",
        "website",
        "email",
        "phone",
        "comment",
        "score",
    ]
    data = df.head(limit)[columns].to_dict(orient="records")
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_ai_prompt(user_question, filters, candidates_json):
    return f"""
Ты ИИ-ассистент сервиса поиска и сравнения поставщиков продуктов питания.
Работай только с данными из базы ниже. Не придумывай поставщиков, контакты, цены, сертификаты или условия.
Если каких-то данных нет, прямо напиши: «нет данных в базе».

Запрос пользователя:
{user_question}

Фильтры пользователя:
{json.dumps(filters, ensure_ascii=False, indent=2)}

Кандидаты из базы:
{candidates_json}

Сформируй ответ на русском языке и помоги выбрать, с кем связаться в первую очередь.
Обязательно сравни поставщиков по этим пунктам:
- минимальный объем заказа;
- примерная цена или уровень цены, если он доступен;
- наличие документов или сертификатов;
- условия доставки;
- регион работы;
- комментарии или заметки по поставщику;
- контакты.

Формат ответа:
1. Лучший выбор: название поставщика и 2-3 причины.
2. Еще 2-4 подходящих варианта: кратко, чем они полезны.
3. Сравнение найденных вариантов по минимальному заказу, цене, доставке, сертификатам, региону и заметкам.
4. Риски или пробелы в данных: что нужно уточнить перед контактом.
5. Кому написать/позвонить первым и почему.

Если кандидатов нет, предложи, какие фильтры ослабить или как переформулировать запрос.
""".strip()


@st.cache_resource(show_spinner=False)
def make_client(api_key, base_url, project_id):
    kwargs = {"api_key": api_key, "base_url": base_url}
    if project_id:
        kwargs["project"] = project_id
    return OpenAI(**kwargs)


def ask_yandex_ai(api_key, base_url, project_id, agent_id, prompt_text):
    client = make_client(api_key, base_url, project_id)
    response = client.responses.create(prompt={"id": agent_id}, input=prompt_text)
    text = getattr(response, "output_text", "")
    if text:
        return text
    return str(response)


def supplier_card(row):
    st.markdown(f"### {row['name']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Оценка", int(row["score"]))
    c2.metric("Мин. заказ", f"{int(row['min_order']):,}".replace(",", " "))
    c3.metric("Цена", row["price_level"] or "нет данных")
    c4.metric("Доставка", row["delivery"] or "нет данных")
    st.write(f"Категория: {row['category']}")
    st.write(f"Товары: {row['products']}")
    st.write(f"Город / регион: {row['city']} / {row['region']}")
    st.write(f"Документы: {row['certificates'] or 'нет данных'}")
    st.write(f"Контакты: {row['email']} · {row['phone']} · {row['website']}")
    if row["comment"]:
        st.info(row["comment"])


api_key = secret_value(["YANDEX_API_KEY", "YA_API_KEY", "YaApiKey"])
base_url = secret_value(["YANDEX_AGENT_URL", "YANDEX_BASE_URL", "AGENT_URL", "AgentUrl"])
project_id = secret_value(["YANDEX_PROJECT_ID", "PROJECT_ID", "ProjectId"])
agent_id = secret_value(["YANDEX_AGENT_ID", "AGENT_ID", "AgentId"])

yandex_ai_ready = all(is_real_value(value) for value in [api_key, base_url, project_id, agent_id])

st.title("ИИ-ассистент для поиска и сравнения поставщиков")
st.write(
    "Сервис ищет поставщиков по базе, фильтрует по категории, региону и условиям, "
    "а затем передает лучшие варианты в Яндекс ИИ для рекомендации."
)

raw_df = load_default_csv()

if raw_df.empty:
    st.error(
        "CSV с поставщиками не найден. Положите файл suppliers.csv в корень проекта "
        "рядом с app.py или в папку data/suppliers.csv."
    )
    st.stop()

df = prepare_data(raw_df)

categories = ["Все"] + unique_sorted(df, "category")
cities = ["Все"] + unique_sorted(df, "city")
regions = ["Все"] + unique_sorted(df, "region")
prices = unique_sorted(df, "price_level")
deliveries = ["Любая"] + unique_sorted(df, "delivery")

st.subheader("Поиск")
query = st.text_input(
    "Что нужно найти",
    placeholder="Например: упаковка Москва HACCP, молочная продукция с доставкой, ингредиенты до 30000",
)

f1, f2, f3 = st.columns(3)
category = f1.selectbox("Категория", categories)
city = f2.selectbox("Город", cities)
region = f3.selectbox("Регион", regions)

f4, f5, f6 = st.columns(3)
price_levels = f4.multiselect("Уровень цены", prices)
delivery = f5.selectbox("Доставка", deliveries)
certificate = f6.text_input("Сертификат или документ", placeholder="HACCP, ISO 22000, декларация")

max_order_value = int(df["min_order"].max())
if max_order_value > 0:
    slider_step = max(1, min(10000, max_order_value))
    max_min_order = st.slider(
        "Максимально допустимый минимальный заказ",
        min_value=0,
        max_value=max_order_value,
        value=0,
        step=slider_step,
    )
else:
    max_min_order = 0
    st.info(
        "В CSV нет числовых значений в колонке min_order, "
        "поэтому фильтр по минимальному заказу отключен."
    )

filters = {
    "category": category,
    "city": city,
    "region": region,
    "price_levels": price_levels,
    "delivery": delivery,
    "certificate": certificate,
    "max_min_order": max_min_order,
}

filtered = apply_filters(df, query, category, city, region, price_levels, delivery, certificate, max_min_order)
ranked = add_scores(filtered, query, category, city, region, certificate)

m1, m2, m3 = st.columns(3)
m1.metric("Всего в базе", len(df))
m2.metric("Найдено", len(ranked))
m3.metric("Передается в ИИ", min(len(ranked), 15))

if ranked.empty:
    st.warning("Ничего не найдено. Уберите часть фильтров, расширьте регион или оставьте поле сертификата пустым.")
    st.stop()

st.subheader("Лучшие совпадения")
st.dataframe(ranked[DISPLAY_COLUMNS], use_container_width=True)

csv_export = ranked[DISPLAY_COLUMNS].to_csv(index=False).encode("utf-8-sig")
st.download_button("Скачать найденных поставщиков CSV", csv_export, "selected_suppliers.csv", "text/csv")

st.subheader("Карточки поставщиков")
for _, row in ranked.head(5).iterrows():
    with st.container():
        supplier_card(row)
        st.markdown("---")

st.subheader("ИИ-рекомендация")
question = st.text_area(
    "Вопрос для ассистента",
    value=query if query.strip() else "Помоги выбрать лучших поставщиков из найденных вариантов и объясни, с кем связаться в первую очередь.",
    height=100,
)

if not yandex_ai_ready:
    st.warning(
        "Яндекс ИИ не настроен. Добавьте в /env/local или env/local переменные "
        "YANDEX_API_KEY, YANDEX_AGENT_URL, YANDEX_PROJECT_ID и YANDEX_AGENT_ID. "
        "Поля ввода в интерфейсе специально убраны, чтобы не показывать секреты пользователю."
    )

if st.button("Спросить Яндекс ИИ", type="primary", disabled=not yandex_ai_ready):
    with st.spinner("Яндекс ИИ анализирует поставщиков..."):
        try:
            prompt_text = build_ai_prompt(question, filters, records_for_ai(ranked))
            answer = ask_yandex_ai(api_key, base_url, project_id, agent_id, prompt_text)
            st.markdown(answer)
        except Exception as error:
            st.error(f"Ошибка запроса к Яндекс ИИ: {error}")

with st.expander("Почему этот формат подходит под задание"):
    st.write(
        "Прототип позволяет искать поставщиков по категории товара, учитывать город или регион, "
        "сравнивать минимальный заказ, цену, доставку, документы, заметки и контакты. "
        "Streamlit выбран как быстрый формат демонстрации: проверяющий сразу видит фильтры, таблицу, "
        "карточки поставщиков и ИИ-рекомендацию без отдельной фронтенд-разработки."
    )
