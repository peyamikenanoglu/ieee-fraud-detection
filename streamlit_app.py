################################################
# IEEE-CIS Fraud Detection
# Streamlit Frontend for FastAPI Prediction Service
################################################

import json
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st


################################################
# Page Config
################################################

st.set_page_config(
    page_title="IEEE-CIS Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)


################################################
# Constants
################################################

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_THRESHOLD = 0.71


################################################
# Helper Functions
################################################

def load_default_transaction() -> Dict[str, Any]:
    sample_path = PROJECT_ROOT / "api_test_request.json"

    if sample_path.exists():
        try:
            with open(sample_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            return payload.get("transaction", payload)
        except Exception:
            pass

    return {
        "TransactionID": 1,
        "TransactionDT": 86400,
        "TransactionAmt": 68.5,
        "ProductCD": "W",
        "card1": 13926,
        "card2": 327,
        "card3": 150,
        "card4": "discover",
        "card5": 142,
        "card6": "credit",
        "addr1": 315,
        "addr2": 87,
        "dist1": 19,
        "dist2": None,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "C1": 1,
        "C2": 1,
        "C3": 0,
        "C4": 0,
        "C5": 0,
        "C6": 1,
        "C7": 0,
        "C8": 0,
        "C9": 1,
        "C10": 0,
        "C11": 2,
        "C12": 0,
        "C13": 1,
        "C14": 1,
        "D1": 14,
        "D2": 14,
        "D3": 13,
        "D4": 26,
        "D5": 10,
        "D6": None,
        "D7": None,
        "D8": None,
        "D9": None,
        "D10": 13,
        "D11": 13,
        "D12": None,
        "D13": None,
        "D14": None,
        "D15": 0,
        "M1": "T",
        "M2": "T",
        "M3": "T",
        "M4": "M2",
        "M5": "F",
        "M6": "T",
        "M7": None,
        "M8": None,
        "M9": None,
        "DeviceType": "desktop",
        "DeviceInfo": "Windows",
        "id_30": "Windows 10",
        "id_31": "chrome 63.0",
        "id_33": "1920x1080",
        "id_34": "match_status:2",
        "id_35": "T",
        "id_36": "F",
        "id_37": "T",
        "id_38": "T",
    }


def check_api_health(api_url: str) -> Dict[str, Any]:
    response = requests.get(f"{api_url}/health", timeout=5)
    response.raise_for_status()
    return response.json()


def predict_transaction(api_url: str, transaction: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    payload = {
        "transaction": transaction,
        "threshold": threshold,
    }
    response = requests.post(f"{api_url}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def risk_message(probability: float) -> str:
    if probability >= 0.71:
        return "High fraud risk"
    if probability >= 0.40:
        return "Medium fraud risk"
    return "Low fraud risk"


################################################
# Sidebar
################################################

st.sidebar.title("API Settings")
api_url = st.sidebar.text_input("FastAPI URL", value=DEFAULT_API_URL)
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.00,
    max_value=1.00,
    value=DEFAULT_THRESHOLD,
    step=0.01,
)

st.sidebar.markdown("### Model")
st.sidebar.write("Final model: **v7 Final Ensemble**")
st.sidebar.write("Weights: LightGBM 0.70, XGBoost 0.20, CatBoost 0.10")
st.sidebar.write("Validation ROC-AUC: **0.9292**")

if st.sidebar.button("Check API Health"):
    try:
        health = check_api_health(api_url)
        st.sidebar.success("API is running")
        st.sidebar.json(health)
    except Exception as error:
        st.sidebar.error("API is not reachable")
        st.sidebar.write(str(error))

st.sidebar.markdown("### Links")
st.sidebar.markdown(f"[API Docs]({api_url}/docs)")
st.sidebar.markdown(f"[Health Check]({api_url}/health)")


################################################
# Main Page
################################################

st.title("IEEE-CIS Fraud Detection")
st.caption("Production-style Streamlit frontend connected to a FastAPI fraud prediction service.")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Final Champion", "v7 Ensemble")
col_b.metric("Validation ROC-AUC", "0.9292")
col_c.metric("Decision Threshold", f"{threshold:.2f}")

st.markdown("---")

sample_transaction = load_default_transaction()

with st.expander("How this app works", expanded=False):
    st.write(
        "Enter transaction fields, click Predict, and Streamlit sends the transaction to the FastAPI backend. "
        "The API performs v7 feature engineering, applies preprocessing artifacts, runs LightGBM, XGBoost, and CatBoost, "
        "then returns the weighted ensemble fraud probability."
    )


################################################
# Input Form
################################################

st.subheader("Transaction Input")

with st.form("transaction_form"):
    st.markdown("#### Core Transaction Fields")

    c1, c2, c3, c4 = st.columns(4)
    transaction_id = c1.number_input("TransactionID", value=int(sample_transaction.get("TransactionID", 1)), step=1)
    transaction_dt = c2.number_input("TransactionDT", value=int(sample_transaction.get("TransactionDT", 86400)), step=1)
    transaction_amt = c3.number_input("TransactionAmt", value=float(sample_transaction.get("TransactionAmt", 68.5)), step=0.01)
    product_cd = c4.selectbox("ProductCD", options=["W", "C", "R", "H", "S"], index=0)

    st.markdown("#### Card and Address Fields")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    card1 = c1.number_input("card1", value=int(sample_transaction.get("card1", 13926)), step=1)
    card2 = c2.number_input("card2", value=int(sample_transaction.get("card2", 327)), step=1)
    card3 = c3.number_input("card3", value=int(sample_transaction.get("card3", 150)), step=1)
    card4 = c4.selectbox("card4", options=["visa", "mastercard", "discover", "american express", "Missing"], index=2)
    card5 = c5.number_input("card5", value=int(sample_transaction.get("card5", 142)), step=1)
    card6 = c6.selectbox("card6", options=["credit", "debit", "Missing"], index=0)

    c1, c2, c3, c4 = st.columns(4)
    addr1 = c1.number_input("addr1", value=int(sample_transaction.get("addr1", 315)), step=1)
    addr2 = c2.number_input("addr2", value=int(sample_transaction.get("addr2", 87)), step=1)
    dist1 = c3.number_input("dist1", value=float(sample_transaction.get("dist1", 19) or 0), step=1.0)
    with c4:
        dist2_missing = st.checkbox("dist2 missing", value=True)
        dist2_value = st.number_input(
            "dist2 value",
            value=0.0,
            step=1.0,
            key="dist2_value",
        )
        dist2 = None if dist2_missing else float(dist2_value)

    st.markdown("#### Email Fields")

    c1, c2 = st.columns(2)
    p_emaildomain = c1.text_input("P_emaildomain", value=str(sample_transaction.get("P_emaildomain", "gmail.com") or "")) or None
    r_emaildomain_input = c2.text_input("R_emaildomain", value="")
    r_emaildomain = r_emaildomain_input or None

    st.markdown("#### Count Features C1-C14")

    count_defaults = {f"C{i}": int(sample_transaction.get(f"C{i}", 0)) for i in range(1, 15)}
    count_values = {}
    for row_start in [1, 8]:
        cols = st.columns(7)
        for idx, feature_num in enumerate(range(row_start, row_start + 7)):
            feature_name = f"C{feature_num}"
            count_values[feature_name] = cols[idx].number_input(
                feature_name,
                value=count_defaults.get(feature_name, 0),
                step=1,
                key=f"count_{feature_name}",
            )

    st.markdown("#### Time Delta Features D1-D15")

    d_features = [
        "D1", "D2", "D3", "D4", "D5",
        "D6", "D7", "D8", "D9", "D10",
        "D11", "D12", "D13", "D14", "D15",
    ]

    d_demo_defaults = {
        "D1": 0.0,
        "D2": 0.0,
        "D3": 1.0,
        "D4": 26.0,
        "D5": 10.0,
        "D6": 0.0,
        "D7": 0.0,
        "D8": 0.0,
        "D9": 0.0,
        "D10": 0.0,
        "D11": 0.0,
        "D12": 0.0,
        "D13": 0.0,
        "D14": 0.0,
        "D15": 0.0,
    }

    d_values = {}
    d_missing = {}

    for row_start in [0, 5, 10]:
        cols = st.columns(5)

        for idx, feature_name in enumerate(d_features[row_start:row_start + 5]):
            with cols[idx]:
                default_value = sample_transaction.get(feature_name, None)

                default_missing = default_value is None
                default_numeric = (
                    float(default_value)
                    if default_value is not None
                    else d_demo_defaults.get(feature_name, 0.0)
                )

                d_missing[feature_name] = st.checkbox(
                    f"{feature_name} missing",
                    value=default_missing,
                    key=f"missing_{feature_name}",
                )

                numeric_value = st.number_input(
                    f"{feature_name} value",
                    value=default_numeric,
                    step=1.0,
                    key=f"value_{feature_name}",
                )

                d_values[feature_name] = None if d_missing[feature_name] else float(numeric_value)

    st.markdown("#### Match and Identity Fields")

    c1, c2, c3, c4, c5 = st.columns(5)
    m1 = c1.selectbox("M1", ["T", "F", None], index=0)
    m2 = c2.selectbox("M2", ["T", "F", None], index=0)
    m3 = c3.selectbox("M3", ["T", "F", None], index=0)
    m4 = c4.selectbox("M4", ["M0", "M1", "M2", None], index=2)
    m5 = c5.selectbox("M5", ["T", "F", None], index=1)

    c1, c2, c3, c4 = st.columns(4)
    m6 = c1.selectbox("M6", ["T", "F", None], index=0)
    m7 = c2.selectbox("M7", ["T", "F", None], index=2)
    m8 = c3.selectbox("M8", ["T", "F", None], index=2)
    m9 = c4.selectbox("M9", ["T", "F", None], index=2)

    c1, c2, c3, c4 = st.columns(4)
    device_type = c1.selectbox("DeviceType", ["desktop", "mobile", None], index=0)
    device_info = c2.text_input("DeviceInfo", value="Windows") or None
    id_30 = c3.text_input("id_30", value="Windows 10") or None
    id_31 = c4.text_input("id_31", value="chrome 63.0") or None

    c1, c2, c3, c4, c5 = st.columns(5)
    id_33 = c1.text_input("id_33", value="1920x1080") or None
    id_34 = c2.text_input("id_34", value="match_status:2") or None
    id_35 = c3.selectbox("id_35", ["T", "F", None], index=0)
    id_36 = c4.selectbox("id_36", ["T", "F", None], index=1)
    id_37 = c5.selectbox("id_37", ["T", "F", None], index=0)

    c1, _, _, _, _ = st.columns(5)
    id_38 = c1.selectbox("id_38", ["T", "F", None], index=0)

    submitted = st.form_submit_button("Predict Fraud Risk")


################################################
# Prediction
################################################

if submitted:
    transaction = {
        "TransactionID": int(transaction_id),
        "TransactionDT": int(transaction_dt),
        "TransactionAmt": float(transaction_amt),
        "ProductCD": product_cd,
        "card1": int(card1),
        "card2": int(card2),
        "card3": int(card3),
        "card4": card4,
        "card5": int(card5),
        "card6": card6,
        "addr1": int(addr1),
        "addr2": int(addr2),
        "dist1": float(dist1),
        "dist2": dist2,
        "P_emaildomain": p_emaildomain,
        "R_emaildomain": r_emaildomain,
        **{key: int(value) for key, value in count_values.items()},
        **d_values,
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "M4": m4,
        "M5": m5,
        "M6": m6,
        "M7": m7,
        "M8": m8,
        "M9": m9,
        "DeviceType": device_type,
        "DeviceInfo": device_info,
        "id_30": id_30,
        "id_31": id_31,
        "id_33": id_33,
        "id_34": id_34,
        "id_35": id_35,
        "id_36": id_36,
        "id_37": id_37,
        "id_38": id_38,
    }

    try:
        result = predict_transaction(api_url, transaction, threshold)

        st.markdown("---")
        st.subheader("Prediction Result")

        probability = float(result["fraud_probability"])
        prediction = int(result["prediction"])
        risk_label = result["risk_label"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fraud Probability", f"{probability:.4f}")
        col2.metric("Prediction", prediction)
        col3.metric("Risk Label", risk_label)
        col4.metric("Threshold", f"{float(result['threshold']):.2f}")

        st.progress(min(max(probability, 0.0), 1.0))
        st.info(risk_message(probability))

        if prediction == 1:
            st.error("The transaction is classified as fraud according to the selected threshold.")
        else:
            st.success("The transaction is classified as non-fraud according to the selected threshold.")

        st.markdown("#### Model Probabilities")
        model_probabilities = result.get("model_probabilities", {})
        st.dataframe(
            {
                "Model": list(model_probabilities.keys()),
                "Fraud Probability": [round(float(value), 6) for value in model_probabilities.values()],
            },
            use_container_width=True,
        )

        with st.expander("Raw API Response"):
            st.json(result)

        with st.expander("Transaction Payload Sent to API"):
            st.json({"transaction": transaction, "threshold": threshold})

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI backend. Start it with: uvicorn api.app:app --reload")
    except requests.exceptions.HTTPError as error:
        st.error("API returned an error.")
        st.write(str(error))
        try:
            st.json(error.response.json())
        except Exception:
            st.write(error.response.text)
    except Exception as error:
        st.error("Unexpected error during prediction.")
        st.write(str(error))
