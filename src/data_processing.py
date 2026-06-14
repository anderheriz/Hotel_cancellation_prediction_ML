from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "hotel_bookings_clean.csv"
TRAIN_PATH = ROOT / "data" / "train" / "train.csv"
TEST_PATH = ROOT / "data" / "test" / "test.csv"
TARGET = "is_canceled"
RANDOM_STATE = 42


def load_raw_data(path=RAW_PATH):
    return pd.read_csv(path)


def clean_data(df):
    df_clean = df.copy()

    # Mantener duplicados exactos en la primera iteracion.
    # No hay booking_id, por lo que no podemos confirmar que sean errores.

    # Imputaciones sencillas.
    df_clean["children"] = df_clean["children"].fillna(0)
    df_clean["country"] = df_clean["country"].fillna("Unknown")

    # Variables binarias a partir de columnas con nulos o conteos.
    df_clean["has_company"] = df_clean["company"].notna().astype(int)
    df_clean["has_agent"] = df_clean["agent"].notna().astype(int)
    df_clean["has_special_requests"] = (df_clean["total_of_special_requests"] > 0).astype(int)
    df_clean["needs_parking"] = (df_clean["required_car_parking_spaces"] > 0).astype(int)
    df_clean["has_previous_cancellations"] = (df_clean["previous_cancellations"] > 0).astype(int)
    df_clean["has_previous_bookings"] = (df_clean["previous_bookings_not_canceled"] > 0).astype(int)

    # Variables agregadas.
    df_clean["total_nights"] = df_clean["stays_in_weekend_nights"] + df_clean["stays_in_week_nights"]
    df_clean["total_guests"] = df_clean["adults"] + df_clean["children"] + df_clean["babies"]

    # Eliminar registros anomalos claros para una primera iteracion.
    df_clean = df_clean[df_clean["total_guests"] > 0]
    df_clean = df_clean[df_clean["total_nights"] > 0]
    df_clean = df_clean[df_clean["adr"] >= 0]

    # Eliminar columnas no utilizables en el primer modelo.
    columns_to_drop = [
        "reservation_status",       # fuga de informacion
        "reservation_status_date",  # fuga de informacion
        "assigned_room_type",       # posible informacion posterior
        "arrival_date_week_number", # redundante con year/month/day
        "company",                  # muchos nulos; sustituida por has_company
        "agent",                    # identificador; sustituida por has_agent
    ]
    df_clean = df_clean.drop(columns=columns_to_drop)

    return df_clean.reset_index(drop=True)


def save_processed_data():
    df = load_raw_data()
    df_clean = clean_data(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)

    return df_clean


def save_train_test_data(df_clean=None, test_size=0.2):
    if df_clean is None:
        if PROCESSED_PATH.exists():
            df_clean = pd.read_csv(PROCESSED_PATH)
        else:
            df_clean = save_processed_data()

    train_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df_clean[TARGET],
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    return train_df, test_df


if __name__ == "__main__":
    clean = save_processed_data()
    train, test = save_train_test_data(clean)

    print("clean", clean.shape)
    print("train", train.shape)
    print("test", test.shape)
