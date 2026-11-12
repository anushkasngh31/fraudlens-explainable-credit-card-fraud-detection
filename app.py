import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="FraudLens",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


# Saved model and data locations are intentionally kept relative to app.py.
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "models/xgb_fraud_model.pkl")
    preprocessing = joblib.load(BASE_DIR / "models/preprocessing.pkl")
    threshold = joblib.load(BASE_DIR / "models/optimal_threshold.pkl")
    return model, preprocessing, float(threshold)


@st.cache_data
def load_transactions():
    return pd.read_csv(BASE_DIR / "data/sample_transactions.csv")


@st.cache_data
def load_model_results():
    return pd.read_csv(BASE_DIR / "results/model_comparison.csv")


@st.cache_data
def load_threshold_results():
    return pd.read_csv(BASE_DIR / "results/threshold_optimization.csv")


@st.cache_data
def load_test_results():
    return pd.read_csv(
        BASE_DIR / "results/test_threshold_comparison.csv",
        index_col=0,
    )


@st.cache_resource
def load_shap_explainer(_model_object):
    """Create a cached explainer without hashing the unhashable XGBoost model."""
    return shap.TreeExplainer(_model_object)


def compact_layout(figure, *, height=330):
    """Apply a consistent, compact layout to Plotly figures."""
    figure.update_layout(
        height=height,
        margin=dict(l=35, r=25, t=55, b=35),
        legend_title_text=None,
    )
    return figure


model, preprocessing, threshold = load_artifacts()
transactions = load_transactions()
model_results = load_model_results()
threshold_results = load_threshold_results()
test_results = load_test_results()

st.title("🛡️ FraudLens")
st.caption("Explainable & cost-sensitive fraud detection with XGBoost")
st.write(
    "Explore predictions, validation performance, threshold trade-offs, and "
    "SHAP explanations using the project's saved model artifacts and results."
)

tab_prediction, tab_analytics, tab_cost, tab_explain = st.tabs(
    [
        "💳 Transaction Prediction",
        "📊 Model Analytics",
        "💰 Cost Optimization",
        "🔍 Explainable AI",
    ]
)


with tab_prediction:
    st.subheader("Transaction Risk Prediction")
    transaction_type = st.radio(
        "Transaction type",
        ["All", "Genuine", "Fraud"],
        horizontal=True,
    )

    if transaction_type == "Genuine":
        filtered_transactions = transactions[transactions["Actual_Class"] == 0]
    elif transaction_type == "Fraud":
        filtered_transactions = transactions[transactions["Actual_Class"] == 1]
    else:
        filtered_transactions = transactions

    if filtered_transactions.empty:
        st.warning("No transactions are available for this filter.")
        st.stop()

    transaction_id = st.selectbox(
        "Select a transaction",
        options=filtered_transactions.index.tolist(),
        format_func=lambda row_id: (
            f"Transaction {row_id + 1} — "
            f"{'Fraud' if transactions.loc[row_id, 'Actual_Class'] == 1 else 'Genuine'}"
        ),
    )

    # These values are prepared here before the explainability tab executes.
    transaction = transactions.loc[[transaction_id]].copy()
    actual_class = int(transaction.at[transaction_id, "Actual_Class"])
    features = transaction.drop(columns=["Actual_Class"])
    features = features[preprocessing["feature_names"]].copy()
    features["Amount"] = preprocessing["amount_scaler"].transform(
        features[["Amount"]]
    ).ravel()
    features["Time"] = preprocessing["time_scaler"].transform(
        features[["Time"]]
    ).ravel()

    fraud_probability = float(model.predict_proba(features)[0, 1])
    predicted_class = int(fraud_probability >= threshold)

    probability_col, threshold_col, actual_col = st.columns(3)
    probability_col.metric("Fraud probability", f"{fraud_probability:.2%}")
    threshold_col.metric("Decision threshold", f"{threshold:.2f}")
    actual_col.metric(
        "Actual class",
        "Fraud" if actual_class == 1 else "Genuine",
    )

    st.progress(min(max(fraud_probability, 0.0), 1.0))
    if predicted_class:
        st.error("⚠️ Potential fraud detected")
    else:
        st.success("✅ Transaction classified as genuine")

    with st.expander("View raw transaction features"):
        st.dataframe(transaction, use_container_width=True)

    st.caption(
        "The filter uses the sample dataset's known labels for demonstration. "
        "A real-time system would not know the actual class at prediction time."
    )
    st.info(
        "This research demonstration is not a production banking system. "
        "Because the model was trained with class weighting, its scores should not "
        "be treated as calibrated real-world fraud probabilities."
    )


with tab_analytics:
    st.subheader("Validation Model Comparison")
    st.caption("Metrics below were calculated on validation data at the default 0.50 threshold.")

    metric_columns = {
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}",
        "PR-AUC": "{:.4f}",
        "ROC-AUC": "{:.4f}",
    }
    st.dataframe(
        model_results.style.format(metric_columns),
        use_container_width=True,
        hide_index=True,
    )

    xgb_rows = model_results.loc[model_results["Model"] == "XGBoost"]
    if not xgb_rows.empty:
        xgb_results = xgb_rows.iloc[0]
        pr_col, precision_col, recall_col = st.columns(3)
        pr_col.metric("XGBoost validation PR-AUC", f"{xgb_results['PR-AUC']:.4f}")
        precision_col.metric("XGBoost precision", f"{xgb_results['Precision']:.2%}")
        recall_col.metric("XGBoost recall", f"{xgb_results['Recall']:.2%}")

    selected_metric = st.selectbox(
        "Compare validation metric",
        ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1-Score"],
    )
    comparison_chart = px.bar(
        model_results,
        x="Model",
        y=selected_metric,
        color="Model",
        text_auto=".3f",
        title=f"Validation comparison — {selected_metric}",
    )
    comparison_chart.update_yaxes(range=[0, 1])
    comparison_chart.update_layout(showlegend=False)
    st.plotly_chart(compact_layout(comparison_chart), use_container_width=True)

    st.divider()
    st.subheader("Final Test-Set Evaluation")
    st.write(
        "Performance on 56,962 previously unseen credit-card transactions, "
        "including 98 fraud cases."
    )
    test_pr_col, test_roc_col = st.columns(2)
    test_pr_col.metric("Test PR-AUC", "0.8709")
    test_roc_col.metric("Test ROC-AUC", "0.9758")

    formatted_test_results = test_results.copy()
    formatted_test_results.index.name = "Decision threshold"
    st.dataframe(
        formatted_test_results.style.format(
            {
                "TP": "{:.0f}",
                "FP": "{:.0f}",
                "FN": "{:.0f}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1": "{:.2%}",
                "Estimated Cost": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    test_cost_chart = px.bar(
        formatted_test_results.reset_index(),
        x="Decision threshold",
        y="Estimated Cost",
        text_auto=".0f",
        title="Estimated test cost by decision threshold",
    )
    test_cost_chart.update_yaxes(title="Estimated cost (USD)")
    st.plotly_chart(compact_layout(test_cost_chart), use_container_width=True)
    st.warning(
        "The validation-selected threshold did not improve estimated cost on the "
        "unseen test set. Threshold choices should therefore be validated again "
        "before making operational claims."
    )


with tab_cost:
    st.subheader("Validation Threshold Optimization")
    st.write(
        "Use the slider to inspect the nearest saved validation result. The chart "
        "shows the full saved validation threshold sweep."
    )

    selected_threshold = st.slider(
        "Fraud detection threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
    )
    nearest_index = (threshold_results["Threshold"] - selected_threshold).abs().idxmin()
    selected_row = threshold_results.loc[nearest_index]

    fp_col, fn_col, cost_col = st.columns(3)
    fp_col.metric("False positives", f"{int(selected_row['FP']):,}")
    fn_col.metric("Missed fraud cases", f"{int(selected_row['FN']):,}")
    cost_col.metric("Estimated cost", f"${selected_row['Cost']:,.0f}")

    precision_col, recall_col, saved_threshold_col = st.columns(3)
    precision_col.metric("Precision", f"{selected_row['Precision']:.2%}")
    recall_col.metric("Recall", f"{selected_row['Recall']:.2%}")
    saved_threshold_col.metric("Nearest saved threshold", f"{selected_row['Threshold']:.2f}")

    cost_chart = px.line(
        threshold_results,
        x="Threshold",
        y="Cost",
        markers=True,
        title="Estimated validation cost by classification threshold",
    )
    cost_chart.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#EF553B",
        annotation_text="Saved validation threshold",
        annotation_position="top right",
    )
    cost_chart.add_vline(
        x=selected_threshold,
        line_dash="dot",
        line_color="#636EFA",
        annotation_text="Selected threshold",
        annotation_position="top left",
    )
    cost_chart.update_yaxes(title="Estimated cost (USD)")
    st.plotly_chart(compact_layout(cost_chart, height=350), use_container_width=True)
    st.caption(
        "Hypothetical validation costs (USD): 250 per missed fraud, 15 per false "
        "alarm, and 3 per detected fraud. These figures are scenario assumptions, "
        "not actual financial losses."
    )


with tab_explain:
    st.subheader("Global SHAP Feature Importance")
    st.write(
        "The summary plot shows feature influence across the saved evaluation sample. "
        "Features V1–V28 are anonymized PCA-derived variables."
    )
    image_left, image_center, image_right = st.columns([1, 3, 1])
    with image_center:
        st.image(
            str(BASE_DIR / "results/shap_beeswarm.png"),
            caption="Global SHAP feature importance",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Explain the Selected Transaction")
    st.caption(
        "Uses the transaction selected in the Transaction Prediction tab.")
    if st.button("Generate SHAP waterfall", type="primary"):
        with st.spinner("Calculating SHAP values..."):
            explainer = load_shap_explainer(model)
            explanation = explainer(features)
            plt.close("all")
            shap.plots.waterfall(explanation[0], max_display=10, show=False)
            waterfall_figure = plt.gcf()
            waterfall_figure.set_size_inches(8, 5)
            waterfall_figure.tight_layout()

            waterfall_left, waterfall_center, waterfall_right = st.columns([1, 3, 1])
            with waterfall_center:
                st.pyplot(waterfall_figure, use_container_width=True)
            plt.close(waterfall_figure)

        st.caption(
            "Positive SHAP values push the prediction toward fraud, while negative "
            "values push it toward genuine. SHAP values are expressed in the model's "
            "raw log-odds space."
        )
