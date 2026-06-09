import pickle
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import load_train_test, train_model

RANDOM_STATE = 42
BASELINE_MODEL_PATH = ROOT / "models" / "baseline_random_forest.pkl"


def main():
    X_train, X_test, y_train, y_test = load_train_test()

    baseline_estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model = train_model(X_train, y_train, baseline_estimator)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification report")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    BASELINE_MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(BASELINE_MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
    print(f"Saved baseline model: {BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    main()
