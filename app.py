import os
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=BASE_DIR
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "hospital.csv"
)

try:
    df = pd.read_csv(CSV_PATH)

except FileNotFoundError:
    raise FileNotFoundError(
        "hospital.csv not found. "
        "Make sure hospital.csv is in the same folder as app.py."
    )

df.columns = df.columns.str.strip()

features = [
    "Age",
    "Blood_Pressure",
    "Blood_Sugar",
    "BMI"
]

target = "Diabetes"

required_columns = features + [target]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in hospital.csv: {missing_columns}"
    )

for column in features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[target] = (
    df[target]
    .astype(str)
    .str.strip()
    .str.lower()
)

df[target] = df[target].map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0
})

df = df.dropna(
    subset=required_columns
).copy()


df[target] = df[target].astype(int)


X = df[features]

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

])


model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()
classifier = model.named_steps[
    "classifier"
]

coefficients = classifier.coef_[0]

feature_importance = []

for feature, coefficient in zip(
    features,
    coefficients
):

    feature_importance.append({

        "feature": feature,

        "coefficient": round(
            float(coefficient),
            4
        ),

        "impact": (
            "Positive"
            if coefficient > 0
            else "Negative"
        )
    })

feature_importance.sort(
    key=lambda x: abs(
        x["coefficient"]
    ),
    reverse=True
)

total_patients = len(df)

diabetes_patients = int(
    df[target].sum()
)

non_diabetes_patients = (
    total_patients -
    diabetes_patients
)

training_records = len(
    X_train
)

testing_records = len(
    X_test
)

metrics_data = {

    "accuracy": round(
        accuracy * 100,
        2
    ),

    "precision": round(
        precision * 100,
        2
    ),

    "recall": round(
        recall * 100,
        2
    ),

    "f1": round(
        f1 * 100,
        2
    ),

    "roc_auc": round(
        roc_auc * 100,
        2
    )
}
def dashboard_data():

    return {

        "accuracy":
            metrics_data["accuracy"],

        "precision":
            metrics_data["precision"],

        "recall":
            metrics_data["recall"],

        "f1":
            metrics_data["f1"],

        "roc_auc":
            metrics_data["roc_auc"],

        "total_patients":
            total_patients,

        "diabetes_patients":
            diabetes_patients,

        "non_diabetes_patients":
            non_diabetes_patients,

        "training_records":
            training_records,

        "testing_records":
            testing_records,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "feature_importance":
            feature_importance
    }

@app.route("/")
def home():

    return render_template(
        "index.html",
        **dashboard_data()
    )

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        patient_name = request.form.get(
            "patient_name",
            "Patient"
        ).strip()

        age = float(
            request.form.get(
                "age",
                ""
            )
        )

        blood_pressure = float(
            request.form.get(
                "blood_pressure",
                ""
            )
        )

        blood_sugar = float(
            request.form.get(
                "blood_sugar",
                ""
            )
        )

        bmi = float(
            request.form.get(
                "bmi",
                ""
            )
        )

        if not patient_name:

            patient_name = "Patient"


        if not 1 <= age <= 120:

            raise ValueError(
                "Age must be between 1 and 120."
            )


        if not 40 <= blood_pressure <= 250:

            raise ValueError(
                "Blood Pressure must be between 40 and 250."
            )


        if not 20 <= blood_sugar <= 600:

            raise ValueError(
                "Blood Sugar must be between 20 and 600."
            )


        if not 10 <= bmi <= 80:

            raise ValueError(
                "BMI must be between 10 and 80."
            )

        new_patient = pd.DataFrame(

            [[
                age,
                blood_pressure,
                blood_sugar,
                bmi
            ]],

            columns=features
        )
        prediction = model.predict(
            new_patient
        )[0]


        probability = model.predict_proba(
            new_patient
        )[0][1]


        probability_percentage = (
            probability * 100
        )

        if prediction == 1:

            result = (
                "Diabetes Risk Detected"
            )

            result_class = "positive"

        else:

            result = (
                "No Diabetes Risk Detected"
            )

            result_class = "negative"

        if probability < 0.30:

            risk_level = "Low Risk"

            risk_class = "low"

        elif probability < 0.70:

            risk_level = "Moderate Risk"

            risk_class = "moderate"

        else:

            risk_level = "High Risk"

            risk_class = "high"

        prediction_time = datetime.now().strftime(
            "%d %B %Y, %I:%M %p"
        )

        return render_template(

            "index.html",

            **dashboard_data(),

            patient_name=patient_name,

            patient_age=age,

            patient_blood_pressure=
                blood_pressure,

            patient_blood_sugar=
                blood_sugar,

            patient_bmi=bmi,

            result=result,

            result_class=result_class,

            probability=round(
                probability_percentage,
                2
            ),

            risk_level=risk_level,

            risk_class=risk_class,

            prediction_time=
                prediction_time
        )

    except ValueError as error:

        return render_template(

            "index.html",

            **dashboard_data(),

            error=str(error)
        )
    except Exception as error:

        return render_template(

            "index.html",

            **dashboard_data(),

            error=(
                f"Unexpected error: {error}"
            )
        )

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       DIABETES PREDICTION SYSTEM")
    print("=" * 60)

    print(
        f"Dataset records : {total_patients}"
    )

    print(
        f"Training records: {training_records}"
    )

    print(
        f"Testing records : {testing_records}"
    )

    print()
    print("MODEL PERFORMANCE")
    print("-" * 60)

    print(
        f"Accuracy  : {metrics_data['accuracy']}%"
    )

    print(
        f"Precision : {metrics_data['precision']}%"
    )

    print(
        f"Recall    : {metrics_data['recall']}%"
    )

    print(
        f"F1 Score  : {metrics_data['f1']}%"
    )

    print(
        f"ROC-AUC   : {metrics_data['roc_auc']}%"
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(cm)

    print()
    print("SERVER")
    print("-" * 60)

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
