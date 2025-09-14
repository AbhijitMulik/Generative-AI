import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="Excel → ML Dashboard with LIME", layout="wide")
st.title("📊 Excel → ML Dashboard with LIME Explainability")

# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
if uploaded is None:
    st.info("Upload an Excel (.xlsx/.xls) to begin.")
    st.stop()

df = pd.read_excel(uploaded)
st.success("✅ File uploaded")
st.dataframe(df.head(20))

# -----------------------------
# Column selection
# -----------------------------
with st.expander("Select columns & task"):
    all_cols = df.columns.tolist()
    target_col = st.selectbox("Target (what to predict)", all_cols)
    feature_cols = st.multiselect(
        "Features (inputs)", [c for c in all_cols if c != target_col],
        default=[c for c in all_cols if c != target_col]
    )

    if not feature_cols:
        st.warning("Please select at least one feature.")
        st.stop()

    # Auto-detect task type; allow override
    tgt = df[target_col]
    auto_task = "classification" if (tgt.dtype == "object" or tgt.nunique() <= 10) else "regression"
    task = st.radio("Task type", options=["regression", "classification"], index=0 if auto_task=="regression" else 1)

# -----------------------------
# Quick visualizations
# -----------------------------
st.subheader("📈 Quick Charts")
num_cols = df[feature_cols + [target_col]].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = list(set(feature_cols) - set(num_cols))

# 1) Line (first numeric over index)
if num_cols:
    st.altair_chart(
        alt.Chart(df.reset_index()).mark_line().encode(
            x="index:Q", y=f"{num_cols[0]}:Q"
        ).properties(title=f"{num_cols[0]} over index"),
        use_container_width=True,
    )

# 2) Histogram (target if numeric; else first numeric)
hist_col = target_col if target_col in num_cols else (num_cols[0] if num_cols else None)
if hist_col:
    fig, ax = plt.subplots()
    ax.hist(df[hist_col].dropna(), bins=30)
    ax.set_title(f"Distribution of {hist_col}")
    ax.set_xlabel(hist_col); ax.set_ylabel("Count")
    st.pyplot(fig)

# 3) Categorical vs numeric bar
if cat_cols and num_cols:
    group_col = cat_cols[0]
    value_col = num_cols[0]
    agg = df.groupby(group_col, dropna=False)[value_col].mean().reset_index()
    st.altair_chart(
        alt.Chart(agg).mark_bar().encode(x=f"{group_col}:N", y=f"{value_col}:Q", color=f"{group_col}:N")
        .properties(title=f"Average {value_col} by {group_col}"),
        use_container_width=True,
    )

# -----------------------------
# Prepare data
# -----------------------------
work = df[feature_cols + [target_col]].dropna().copy()
X = work[feature_cols]
y = work[target_col]

# Split
if len(work) < 50:
    st.warning("Dataset is quite small (<50 rows). Results may be unstable.")
test_size = 0.2 if len(work) >= 10 else 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42,
    stratify=y if task=="classification" else None
)

# Preprocess
num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
cat_feats = [c for c in X.columns if c not in num_feats]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
    ],
    remainder="drop",
)

# -----------------------------
# Model selection
# -----------------------------
st.subheader("🤖 Model Training")

if task == "regression":
    model_name = st.selectbox("Choose model", ["Linear Regression", "Random Forest Regressor"])
    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    # Fit transform
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    # Metrics
    st.markdown("**Metrics (Test)**")
    colA, colB, colC = st.columns(3)
    colA.metric("R²",  f"{r2_score(y_test, y_pred):.3f}")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    colB.metric("MAE", f"{mae:.3f}")
    colC.metric("RMSE", f"{rmse:.3f}")

else:
    model_name = st.selectbox("Choose model", ["Logistic Regression", "Random Forest Classifier"])
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    # Encode target if not numeric
    if y_train.dtype == "object":
        y_train = y_train.astype("category")
        y_test  = y_test.astype("category")

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    # Metrics
    st.markdown("**Metrics (Test)**")
    colA, colB = st.columns(2)
    colA.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    colB.metric("F1 (macro)", f"{f1_score(y_test, y_pred, average='macro'):.3f}")

# -----------------------------
# LIME Explainability
# -----------------------------
st.subheader("🟩 LIME Explanation")

# Feature names
try:
    feat_names = preprocessor.get_feature_names_out()
except Exception:
    feat_names = num_feats + cat_feats

X_train_for_lime = np.array(X_train_t)

if task == "regression":
    explainer = LimeTabularExplainer(
        training_data=X_train_for_lime,
        feature_names=feat_names,
        mode="regression",
        discretize_continuous=True,
        random_state=42
    )
    idx = st.slider("Select test row to explain", 0, X_test_t.shape[0]-1, 0)
    exp = explainer.explain_instance(
        data_row=X_test_t[idx],
        predict_fn=model.predict,
        num_features=min(10, len(feat_names))
    )

else:
    explainer = LimeTabularExplainer(
        training_data=X_train_for_lime,
        feature_names=feat_names,
        class_names=[str(c) for c in np.unique(y_train)],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )
    idx = st.slider("Select test row to explain", 0, X_test_t.shape[0]-1, 0)
    exp = explainer.explain_instance(
        data_row=X_test_t[idx],
        predict_fn=model.predict_proba,
        num_features=min(10, len(feat_names))
    )

st.markdown("**Original row (before preprocessing):**")
st.write(X_test.iloc[[idx]])

st.markdown("**LIME Explanation:**")
st.write(exp.as_list())

fig = exp.as_pyplot_figure()
st.pyplot(fig)
