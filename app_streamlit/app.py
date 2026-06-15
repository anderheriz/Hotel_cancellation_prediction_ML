from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.pkl"
DATA_PATH = ROOT / "data" / "processed" / "hotel_bookings_clean.csv"
CONFIG_PATH = ROOT / "models" / "model_config.yaml"
TARGET = "is_canceled"


st.set_page_config(
    page_title="Hotel Risk Control",
    layout="wide",
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_config_text():
    if CONFIG_PATH.exists():
        return CONFIG_PATH.read_text(encoding="utf-8")
    return ""


def default_row(df):
    values = {}
    for column in df.drop(columns=[TARGET]).columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            values[column] = df[column].median()
        else:
            values[column] = df[column].mode(dropna=True).iloc[0]
    return values


def options_for(df, column, max_options=None):
    values = df[column].dropna().value_counts().index.tolist()
    if max_options:
        values = values[:max_options]
    return values


def apply_scenario(row, scenario):
    scenarios = {
        "Bajo riesgo": {
            "hotel": "Resort Hotel",
            "lead_time": 12,
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "adr": 95.0,
            "total_of_special_requests": 2,
            "required_car_parking_spaces": 1,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 2,
            "is_repeated_guest": 1,
        },
        "Riesgo medio": {
            "hotel": "City Hotel",
            "lead_time": 90,
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "adr": 110.0,
            "total_of_special_requests": 1,
            "required_car_parking_spaces": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "is_repeated_guest": 0,
        },
        "Alto riesgo": {
            "hotel": "City Hotel",
            "lead_time": 240,
            "market_segment": "Groups",
            "distribution_channel": "TA/TO",
            "deposit_type": "Non Refund",
            "customer_type": "Transient",
            "adr": 80.0,
            "total_of_special_requests": 0,
            "required_car_parking_spaces": 0,
            "previous_cancellations": 1,
            "previous_bookings_not_canceled": 0,
            "is_repeated_guest": 0,
        },
    }
    row.update(scenarios.get(scenario, {}))
    return row


def build_input(row, df):
    feature_columns = df.drop(columns=[TARGET]).columns.tolist()
    return pd.DataFrame([row], columns=feature_columns)


def risk_label(probability):
    if probability >= 0.65:
        return "Riesgo alto", "#b42318", "#fef3f2"
    if probability >= 0.35:
        return "Riesgo medio", "#b54708", "#fffaeb"
    return "Riesgo bajo", "#027a48", "#ecfdf3"


def impact_summary(probability):
    if probability >= 0.65:
        return "Reserva prioritaria para revisar condiciones, contacto previo y ocupacion prevista."
    if probability >= 0.35:
        return "Reserva a monitorizar: conviene confirmar intencion de viaje antes de la llegada."
    return "Reserva estable: no requiere una accion especial en este momento."


def scenario_note(scenario):
    notes = {
        "Bajo riesgo": "Ejemplo de reserva estable: poca antelacion, cliente con historial y senales positivas.",
        "Riesgo medio": "Ejemplo de reserva habitual: perfil frecuente en el canal online y riesgo moderado.",
        "Alto riesgo": "Ejemplo de reserva sensible: mucha antelacion, grupo/deposito y pocas senales de compromiso.",
        "Personalizado": "Ajusta los datos para simular una reserva real recibida por el hotel.",
    }
    return notes.get(scenario, "")


def complete_booking_features(row):
    row["total_guests"] = row["adults"] + row["children"] + row["babies"]
    row["booking_changes"] = 0
    row["days_in_waiting_list"] = 0
    row["has_company"] = 0
    row["has_agent"] = 1 if row["market_segment"] in ["Online TA", "Offline TA/TO", "Groups"] else 0
    row["is_repeated_guest"] = int(row["previous_bookings_not_canceled"] > 0)
    row["has_special_requests"] = int(row["total_of_special_requests"] > 0)
    row["needs_parking"] = int(row["required_car_parking_spaces"] > 0)
    row["has_previous_cancellations"] = int(row["previous_cancellations"] > 0)
    row["has_previous_bookings"] = int(row["previous_bookings_not_canceled"] > 0)
    return row


def fixed_scenario_prediction(model, df, scenario):
    scenario_row = default_row(df)
    scenario_row = apply_scenario(scenario_row, scenario)
    scenario_row = complete_booking_features(scenario_row)
    scenario_input = build_input(scenario_row, df)
    probability = float(model.predict_proba(scenario_input)[:, 1][0])
    label, color, background = risk_label(probability)
    return probability, label, color, background


def business_action(probability):
    if probability >= 0.65:
        return [
            "Revisar condiciones de deposito o prepago.",
            "Marcar la reserva para seguimiento comercial.",
            "Considerarla en decisiones de overbooking controlado.",
        ]
    if probability >= 0.35:
        return [
            "Monitorizar la reserva antes de la fecha de llegada.",
            "Enviar recordatorio o confirmacion previa.",
            "Revisar si hay senales adicionales de riesgo.",
        ]
    return [
        "Mantener flujo normal de la reserva.",
        "No aplicar acciones restrictivas.",
        "Priorizar seguimiento en reservas con mayor probabilidad.",
    ]


def reservation_inbox(model, df, current_row):
    def score_reservation(reservation_id, row_values, source):
        reservation = complete_booking_features(dict(row_values))
        probability = float(model.predict_proba(build_input(reservation, df))[:, 1][0])
        label, color, background = risk_label(probability)
        return {
            "reservation_id": reservation_id,
            "source": source,
            "risk": label,
            "probability": probability,
            "color": color,
            "background": background,
            "action": business_action(probability)[0],
        }

    high_risk = apply_scenario(default_row(df), "Alto riesgo")
    medium_risk = apply_scenario(default_row(df), "Riesgo medio")
    low_risk = apply_scenario(default_row(df), "Bajo riesgo")

    online_watch = default_row(df)
    online_watch.update(
        {
            "hotel": "City Hotel",
            "lead_time": 150,
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "deposit_type": "No Deposit",
            "total_of_special_requests": 0,
            "required_car_parking_spaces": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
        }
    )

    reservations = [
        ("BRM-2026-001", current_row, "Reserva en pantalla"),
        ("BRM-2026-014", high_risk, "Grupo / deposito"),
        ("BRM-2026-027", online_watch, "Online TA sin senales"),
        ("BRM-2026-042", medium_risk, "Seguimiento normal"),
        ("BRM-2026-058", low_risk, "Cliente estable"),
    ]

    records = [
        score_reservation(reservation_id, row_values, source)
        for reservation_id, row_values, source in reservations
    ]
    return sorted(records, key=lambda record: record["probability"], reverse=True)


def grouped_feature_importance(model, df):
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_model.feature_importances_

    X = df.drop(columns=[TARGET])
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    def original_name(feature_name):
        clean_name = feature_name.replace("num__", "").replace("cat__", "")
        if clean_name in numeric_features:
            return clean_name
        for column in sorted(categorical_features, key=len, reverse=True):
            if clean_name == column or clean_name.startswith(f"{column}_"):
                return column
        return clean_name

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    importance_df["variable"] = importance_df["feature"].apply(original_name)
    return (
        importance_df.groupby("variable", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )


def read_config_metrics(config_text):
    metrics = {}
    in_metrics = False
    for line in config_text.splitlines():
        if line.strip() == "metrics:":
            in_metrics = True
            continue
        if in_metrics and line and not line.startswith("  "):
            break
        if in_metrics and ":" in line:
            key, value = line.strip().split(":", 1)
            try:
                metrics[key] = float(value.strip())
            except ValueError:
                pass
    return metrics


model = load_model()
df = load_data()
config_text = load_config_text()
metrics = read_config_metrics(config_text)

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }
    html, body {
        background: #eef3f8;
        color: #101828;
    }
    .stApp {
        background: #eef3f8;
        color: #101828;
    }
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp div {
        color: #101828;
    }
    [data-testid="stAppViewContainer"] {
        background: #eef3f8;
    }
    [data-testid="stHeader"] {
        background: #eef3f8;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1760px;
    }
    .top-panel {
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 34px 24px 28px 24px;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9f5 55%, #eef6ff 100%);
        margin-top: 1rem;
        margin-bottom: 18px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
    }
    .eyebrow {
        color: #175cd3;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .top-panel h1 {
        color: #101828;
        font-size: 2.2rem;
        line-height: 1.15;
        margin: 0 0 8px 0;
        text-align: left;
    }
    .top-panel p {
        color: #475467;
        font-size: 1.02rem;
        margin: 0;
        max-width: none;
        white-space: nowrap;
        text-align: left;
    }
    .top-panel .eyebrow {
        text-align: left;
    }
    .section-title {
        color: #101828;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        text-align: center;
    }
    .form-section-title {
        color: #101828;
        font-size: 1.25rem;
        font-weight: 800;
        margin: 18px 0 12px 0;
        text-align: center;
    }
    [data-testid="stWidgetLabel"] {
        min-height: 1rem;
        margin-bottom: 0;
        display: flex;
        justify-content: center;
        text-align: center;
    }
    [data-testid="stWidgetLabel"] p {
        font-size: 0.76rem;
        line-height: 1.05;
        font-weight: 700;
        width: 100%;
        text-align: center;
    }
    div[data-testid="stSelectbox"],
    div[data-testid="stNumberInput"],
    div[data-testid="stSlider"] {
        margin-bottom: -0.22rem;
    }
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        min-height: 38px;
        height: 38px;
        font-size: 0.88rem;
    }
    div[data-baseweb="select"] > div {
        text-align: center;
    }
    div[data-baseweb="select"] [data-baseweb="tag"],
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        text-align: center !important;
    }
    div[data-baseweb="select"] > div > div {
        justify-content: center;
    }
    div[data-testid="stNumberInput"] input {
        text-align: center;
    }
    .scenario-note {
        border-left: 4px solid #175cd3;
        border-radius: 6px;
        padding: 10px 12px;
        background: #eff8ff;
        color: #344054;
        margin: 6px 0 16px 0;
        font-size: 0.92rem;
    }
    .scenario-row {
        margin: 0 auto 18px auto;
        max-width: 760px;
    }
    .personalized-card {
        border: 1px solid #bfd7ff;
        border-left: 4px solid #175cd3;
        border-radius: 8px;
        padding: 14px 18px;
        background: #eff8ff;
        min-height: 92px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }
    .personalized-card .scenario-card-title {
        font-size: 1.08rem;
        margin-bottom: 6px;
    }
    .personalized-card .scenario-card-text {
        font-size: 0.92rem;
        line-height: 1.35;
        max-width: 94%;
        margin: 0 auto;
    }
    .fixed-scenario-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        height: 100%;
    }
    .fixed-scenario-card {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 13px 14px;
        background: #ffffff;
        min-height: 132px;
        text-align: center;
    }
    .scenario-card-title {
        color: #101828 !important;
        font-size: 0.92rem;
        font-weight: 800;
        margin-bottom: 8px;
        text-align: center;
    }
    .scenario-card-prob {
        font-size: 1.5rem;
        line-height: 1;
        font-weight: 850;
        margin-bottom: 8px;
        text-align: center;
    }
    .scenario-card-text {
        color: #475467 !important;
        font-size: 0.8rem;
        line-height: 1.28;
        text-align: center;
    }
    .decision-card {
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 20px 22px;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
        text-align: center;
        height: 210px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .decision-label {
        color: #667085;
        font-size: 0.86rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .decision-number {
        font-size: 3.1rem;
        line-height: 1;
        font-weight: 800;
        margin: 0 0 8px 0;
    }
    .risk-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 7px 12px;
        font-weight: 700;
        font-size: 0.92rem;
    }
    .impact-box {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 13px 14px;
        background: #f9fafb;
        color: #344054;
        margin-top: 14px;
        font-size: 0.95rem;
    }
    .action-item {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 9px 12px;
        background: #ffffff;
        margin-bottom: 7px;
        color: #344054;
        text-align: center;
    }
    .action-panel {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px 18px;
        background: #ffffff;
        height: 210px;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .action-panel-title {
        color: #101828 !important;
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .inbox-section {
        margin-top: 22px;
    }
    .inbox-title {
        color: #101828 !important;
        font-size: 1.45rem;
        font-weight: 850;
        text-align: center;
        margin-bottom: 4px;
    }
    .inbox-subtitle {
        color: #667085 !important;
        font-size: 0.92rem;
        text-align: center;
        margin-bottom: 14px;
    }
    .inbox-table {
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.05);
        width: 1120px;
        max-width: 100%;
        margin: 0 auto;
    }
    .inbox-row {
        display: grid;
        grid-template-columns: 260px 190px 145px 420px;
        gap: 18px;
        align-items: center;
        padding: 12px 20px;
        border-bottom: 1px solid #e4e7ec;
    }
    .inbox-row:last-child {
        border-bottom: none;
    }
    .inbox-row > div:nth-child(3) {
        text-align: center;
        justify-self: center;
        width: 100%;
    }
    .inbox-header {
        background: #f8fafc;
        color: #667085 !important;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .booking-id {
        display: block;
        color: #101828 !important;
        font-weight: 850;
        font-size: 0.95rem;
    }
    .booking-source {
        display: block;
        color: #667085 !important;
        font-size: 0.78rem;
        margin-top: 2px;
    }
    .inbox-risk-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 0.84rem;
        font-weight: 800;
        text-align: center;
        min-width: 112px;
    }
    .inbox-probability {
        font-size: 1.15rem;
        font-weight: 850;
        text-align: center;
        justify-self: center;
        width: 100%;
    }
    .inbox-action {
        color: #344054 !important;
        font-size: 0.9rem;
        text-align: left;
        white-space: nowrap;
    }
    .quick-metrics {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
        margin: 0;
    }
    .quick-card {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px 15px;
        background: #ffffff;
        height: 210px;
        overflow: visible;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .quick-label {
        color: #667085 !important;
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 8px;
    }
    .quick-value {
        color: #101828 !important;
        font-size: 1.55rem;
        line-height: 1.18;
        font-weight: 800;
        word-break: normal;
        white-space: normal;
    }
    .quick-value.prediction-canceled {
        color: #b42318 !important;
        font-size: 3.1rem;
        line-height: 1;
    }
    .quick-value.prediction-not-canceled {
        color: #027a48 !important;
        font-size: 3.1rem;
        line-height: 1;
    }
    .threshold-inline {
        border-top: 1px solid #e4e7ec;
        margin-top: 14px;
        padding-top: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .threshold-label {
        color: #667085 !important;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .threshold-value {
        color: #101828 !important;
        font-size: 1.25rem;
        font-weight: 850;
    }
    .probability-bar {
        height: 10px;
        border-radius: 999px;
        background: #eef2f7;
        overflow: hidden;
        margin-top: 14px;
    }
    .probability-fill {
        height: 100%;
        border-radius: 999px;
        background: #5b7fdd;
    }
    .small-muted {color: #667085; font-size: 0.9rem;}
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] p {
        color: #101828;
    }
    h2, h3,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        text-align: center;
    }
    .top-panel h1,
    .top-panel p {
        text-align: left;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p,
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] p,
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] p,
    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] p {
        color: #344054 !important;
        font-weight: 650;
    }
    [data-testid="stRadio"] span {
        color: #344054 !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input {
        background: #ffffff !important;
        color: #101828 !important;
        border-color: #cbd5e1 !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input,
    [data-testid="stNumberInput"] input {
        color: #101828 !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: #175cd3 !important;
        border-color: #175cd3 !important;
    }
    [data-testid="stTickBar"] {
        color: #475467 !important;
    }
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] p {
        color: #101828 !important;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #101828 !important;
    }
    hr {
        border-color: #d0d5dd;
    }
    .stDataFrame {
        color: #101828;
    }
    @media (max-width: 900px) {
        .block-container {
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
        .quick-metrics {
            grid-template-columns: 1fr;
        }
        .scenario-row,
        .fixed-scenario-grid {
            grid-template-columns: 1fr;
        }
        .top-panel h1 {
            font-size: 1.8rem;
        }
        .top-panel p {
            white-space: normal;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-panel">
        <div class="eyebrow">Risk Management - Demo operativa</div>
        <h1>Booking Risk Management (BRM) </h1>
        <p>Estima el riesgo de cancelacion de cada reserva y convierte la prediccion en una accion concreta para proteger ocupacion, ingresos y planificacion.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

decision_container = st.container()
left = st.container()

with left:
    row = default_row(df)

    st.markdown('<div class="form-section-title">Datos comerciales de la reserva</div>', unsafe_allow_html=True)

    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6, gap="small")
    with col_a:
        hotel_options = options_for(df, "hotel")
        row["hotel"] = st.selectbox("Hotel", hotel_options, index=hotel_options.index(row["hotel"]))

        segment_options = options_for(df, "market_segment")
        row["market_segment"] = st.selectbox(
            "Segmento",
            segment_options,
            index=segment_options.index(row["market_segment"]),
        )

        channel_options = options_for(df, "distribution_channel")
        row["distribution_channel"] = st.selectbox(
            "Canal",
            channel_options,
            index=channel_options.index(row["distribution_channel"]),
        )

        deposit_options = options_for(df, "deposit_type")
        row["deposit_type"] = st.selectbox(
            "Deposito",
            deposit_options,
            index=deposit_options.index(row["deposit_type"]),
        )

    with col_b:
        row["lead_time"] = st.slider("Antelacion de reserva", 0, 500, int(row["lead_time"]), step=5)

        month_options = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        row["arrival_date_month"] = st.selectbox(
            "Mes de reserva",
            month_options,
            index=month_options.index(str(row["arrival_date_month"])),
        )

        year_options = sorted(df["arrival_date_year"].unique().tolist())
        row["arrival_date_year"] = st.selectbox(
            "Año de reserva",
            year_options,
            index=year_options.index(int(row["arrival_date_year"])),
        )

        row["arrival_date_day_of_month"] = st.slider(
            "Dia de llegada",
            1,
            31,
            int(row["arrival_date_day_of_month"]),
        )

    with col_c:
        total_nights = st.slider("Noches totales", 1, 30, int(max(1, row["total_nights"])))
        weekend_nights = st.slider(
            "Noches de fin de semana",
            0,
            min(10, total_nights),
            int(min(row["stays_in_weekend_nights"], total_nights)),
        )
        row["stays_in_weekend_nights"] = weekend_nights
        row["stays_in_week_nights"] = total_nights - weekend_nights
        row["total_nights"] = total_nights
        row["adr"] = st.number_input("ADR", min_value=0.0, max_value=600.0, value=float(row["adr"]), step=5.0)

    with col_d:
        row["adults"] = st.number_input("Adultos", min_value=0, max_value=6, value=int(max(1, row["adults"])), step=1)
        row["children"] = st.number_input("Ninos", min_value=0, max_value=5, value=int(row["children"]), step=1)
        row["babies"] = st.number_input("Bebes", min_value=0, max_value=3, value=int(row["babies"]), step=1)
        row["total_guests"] = row["adults"] + row["children"] + row["babies"]

    with col_e:
        customer_options = options_for(df, "customer_type")
        row["customer_type"] = st.selectbox(
            "Tipo de cliente",
            customer_options,
            index=customer_options.index(row["customer_type"]),
        )

        room_options = options_for(df, "reserved_room_type")
        row["reserved_room_type"] = st.selectbox(
            "Habitacion reservada",
            room_options,
            index=room_options.index(row["reserved_room_type"]),
        )

        meal_options = options_for(df, "meal")
        row["meal"] = st.selectbox(
            "Regimen",
            meal_options,
            index=meal_options.index(row["meal"]),
        )

        countries = options_for(df, "country", max_options=25)
        if row["country"] not in countries:
            row["country"] = countries[0]
        row["country"] = st.selectbox("Pais", countries, index=countries.index(row["country"]))

    with col_f:
        row["total_of_special_requests"] = st.slider("Peticiones especiales", 0, 5, int(row["total_of_special_requests"]))
        row["required_car_parking_spaces"] = st.slider("Parking", 0, 3, int(row["required_car_parking_spaces"]))
        row["previous_cancellations"] = st.slider("Cancelaciones previas", 0, 5, int(min(row["previous_cancellations"], 5)))
        row["previous_bookings_not_canceled"] = st.slider(
            "Reservas previas no canceladas",
            0,
            10,
            int(min(row["previous_bookings_not_canceled"], 10)),
        )

    row["booking_changes"] = 0
    row["days_in_waiting_list"] = 0
    row["has_company"] = 0
    row["has_agent"] = 1 if row["market_segment"] in ["Online TA", "Offline TA/TO", "Groups"] else 0
    row["is_repeated_guest"] = int(row["previous_bookings_not_canceled"] > 0)
    row["has_special_requests"] = int(row["total_of_special_requests"] > 0)
    row["needs_parking"] = int(row["required_car_parking_spaces"] > 0)
    row["has_previous_cancellations"] = int(row["previous_cancellations"] > 0)
    row["has_previous_bookings"] = int(row["previous_bookings_not_canceled"] > 0)

with decision_container:
    input_df = build_input(row, df)
    probability = float(model.predict_proba(input_df)[:, 1][0])
    prediction = int(probability >= 0.5)
    label, color, background = risk_label(probability)

    prediction_text = "Cancelada" if prediction else "No cancelada"
    prediction_class = "prediction-canceled" if prediction else "prediction-not-canceled"
    dec_a, dec_b, dec_c = st.columns([1, 1, 1.15], gap="large")

    with dec_a:
        st.markdown(
            f"""
            <div class="quick-metrics">
                <div class="quick-card">
                    <div class="quick-label">Prediccion</div>
                    <div class="quick-value {prediction_class}">{prediction_text}</div>
                    <div class="threshold-inline">
                        <span class="threshold-label">Umbral</span>
                        <span class="threshold-value">50%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dec_b:
        st.markdown(
            f"""
            <div class="decision-card">
                <div class="decision-label">Probabilidad estimada de cancelacion</div>
                <div class="decision-number" style="color: {color};">{probability:.1%}</div>
                <span class="risk-pill" style="color: {color}; background: {background};">{label}</span>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: {probability * 100:.1f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dec_c:
        action_html = "".join(
            f'<div class="action-item">{action}</div>' for action in business_action(probability)
        )
        st.markdown(
            (
                '<div class="action-panel">'
                '<div class="action-panel-title">Plan de accion sugerido</div>'
                f'{action_html}'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

st.divider()

inbox_records = reservation_inbox(model, df, row)
inbox_rows_html = "".join(
    (
        '<div class="inbox-row">'
        '<div>'
        f'<span class="booking-id">{record["reservation_id"]}</span>'
        f'<span class="booking-source">{record["source"]}</span>'
        '</div>'
        '<div>'
        f'<span class="inbox-risk-pill" style="color: {record["color"]}; background: {record["background"]};">{record["risk"]}</span>'
        '</div>'
        f'<div class="inbox-probability" style="color: {record["color"]};">{record["probability"]:.1%}</div>'
        f'<div class="inbox-action">{record["action"]}</div>'
        '</div>'
    )
    for record in inbox_records
)

st.markdown(
    (
        '<div class="inbox-section">'
        '<div class="inbox-title">Inbox de reservas priorizadas</div>'
        '<div class="inbox-subtitle">Bandeja operativa ordenada por probabilidad de cancelacion. IDs simulados para la demo.</div>'
        '<div class="inbox-table">'
        '<div class="inbox-row inbox-header">'
        '<div>ID reserva</div>'
        '<div>Tipo de riesgo</div>'
        '<div>Probabilidad</div>'
        '<div>Accion recomendada</div>'
        '</div>'
        f'{inbox_rows_html}'
        '</div>'
        '</div>'
    ),
    unsafe_allow_html=True,
)
