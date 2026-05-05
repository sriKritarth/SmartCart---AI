"""
SmartCart Customer Segmentation App
-----------------------------------

Expected CSV file:
    smartcart_customers.csv

The app can either:
1. Load smartcart_customers.csv if it is present in the same folder, or
2. Let you upload the CSV from the sidebar.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from kneed import KneeLocator
except Exception:  # pragma: no cover - app should still work without kneed
    KneeLocator = None


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #ffffff 0%, #f7f7fb 100%);
            border: 1px solid #eeeeee;
            box-shadow: 0 4px 18px rgba(0,0,0,0.04);
        }
        .section-note {
            color: #6b7280;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .small-caption {
            color: #6b7280;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_CSV_PATH = Path("smartcart_customers.csv")
REQUIRED_COLUMNS = {
    "ID",
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
}
SPENDING_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
DROP_COLUMNS = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"]
CATEGORICAL_COLUMNS = ["Education", "living_with"]
NUMERIC_PROFILE_COLUMNS = [
    "Income",
    "Recency",
    "Age",
    "Customer_Tenure",
    "total_spendings",
    "total_children",
]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_input_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing_cols = sorted(list(REQUIRED_COLUMNS - set(df.columns)))
    return len(missing_cols) == 0, missing_cols


@st.cache_data(show_spinner=False)
def prepare_customer_data(
    raw_df: pd.DataFrame,
    age_limit: int,
    income_limit: int,
) -> Dict[str, pd.DataFrame | np.ndarray | StandardScaler | OneHotEncoder | PCA]:
    """Replicates and cleans the feature-engineering pipeline from the notebook."""
    df = raw_df.copy()

    # Basic cleaning
    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Feature engineering
    df["Age"] = 2026 - pd.to_numeric(df["Year_Birth"], errors="coerce")
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure"] = (reference_date - df["Dt_Customer"]).dt.days

    df["total_spendings"] = df[SPENDING_COLUMNS].sum(axis=1)
    df["total_children"] = df["Kidhome"] + df["Teenhome"]

    df["Education"] = df["Education"].replace(
        {
            "Graduation": "Graduate",
            "Basic": "UnderGraduate",
            "2n Cycle": "UnderGraduate",
            "Master": "PostGraduate",
            "PhD": "PostGraduate",
        }
    )

    df["living_with"] = df["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Widow": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone",
        }
    )

    cols_to_remove = DROP_COLUMNS + SPENDING_COLUMNS
    df_cleaned = df.drop(columns=cols_to_remove)

    # Outlier filtering from notebook, now controllable from sidebar
    rows_before_outlier_filter = len(df_cleaned)
    df_cleaned = df_cleaned[(df_cleaned["Age"] < age_limit)]
    df_cleaned = df_cleaned[(df_cleaned["Income"] < income_limit)]
    rows_after_outlier_filter = len(df_cleaned)

    # One-hot encoding
    
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded_array = encoder.fit_transform(df_cleaned[CATEGORICAL_COLUMNS])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=df_cleaned.index,
    )

    df_encoded = pd.concat([df_cleaned.drop(columns=CATEGORICAL_COLUMNS), encoded_df], axis=1)
    df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan).dropna()

    # Scaling and PCA
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_encoded)

    pca = PCA(n_components=3, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    return {
        "raw_df": raw_df,
        "featured_df": df,
        "cleaned_df": df_cleaned,
        "encoded_df": df_encoded,
        "x_scaled": x_scaled,
        "x_pca": x_pca,
        "scaler": scaler,
        "encoder": encoder,
        "pca": pca,
        "rows_before_outlier_filter": rows_before_outlier_filter,
        "rows_after_outlier_filter": rows_after_outlier_filter,
    }


@st.cache_data(show_spinner=False)
def evaluate_k_values(x_pca: np.ndarray, max_k: int) -> pd.DataFrame:
    results = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_pca)
        score = silhouette_score(x_pca, labels)
        results.append({"k": k, "wcss": kmeans.inertia_, "silhouette_score": score})
    return pd.DataFrame(results)


def detect_elbow(k_eval_df: pd.DataFrame) -> int | None:
    if KneeLocator is None or k_eval_df.empty:
        return None
    try:
        knee = KneeLocator(
            k_eval_df["k"].tolist(),
            k_eval_df["wcss"].tolist(),
            curve="convex",
            direction="decreasing",
        )
        return knee.knee
    except Exception:
        return None


def run_clustering(x_pca: np.ndarray, method: str, n_clusters: int) -> np.ndarray:
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return model.fit_predict(x_pca)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(x_pca)


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"₹{value:,.0f}"


def build_cluster_profile(df_with_clusters: pd.DataFrame) -> pd.DataFrame:
    available_cols = [c for c in NUMERIC_PROFILE_COLUMNS if c in df_with_clusters.columns]
    summary = df_with_clusters.groupby("Cluster")[available_cols].mean().round(2)
    summary["Customers"] = df_with_clusters.groupby("Cluster").size()
    summary = summary[["Customers"] + available_cols]
    return summary.reset_index()


def add_cluster_labels(df_encoded: pd.DataFrame, x_pca: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    output_df = df_encoded.copy()
    output_df["PCA1"] = x_pca[:, 0]
    output_df["PCA2"] = x_pca[:, 1]
    output_df["PCA3"] = x_pca[:, 2]
    output_df["Cluster"] = labels.astype(str)
    return output_df


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 SmartCart")
    st.caption("Customer segmentation dashboard")

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    st.divider()
    st.subheader("Model settings")
    cluster_method = st.selectbox("Clustering method", ["Agglomerative", "KMeans"], index=0)
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=4, step=1)
    max_k = st.slider("Max K for evaluation", min_value=4, max_value=15, value=10, step=1)

    st.divider()
    st.subheader("Outlier filters")
    age_limit = st.slider("Maximum age cutoff", min_value=50, max_value=120, value=90, step=1)
    income_limit = st.number_input(
        "Maximum income cutoff",
        min_value=50_000,
        max_value=2_000_000,
        value=600_000,
        step=50_000,
    )

    st.divider()
    show_raw_data = st.toggle("Show raw dataset", value=False)
    show_processed_data = st.toggle("Show processed dataset", value=False)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("SmartCart Customer Segmentation")
st.markdown(
    """
    <div class="section-note">
    An interactive Streamlit dashboard for customer clustering using feature engineering, encoding,
    scaling, PCA, and clustering. The default pipeline follows your notebook, with extra controls
    for upload, filtering, cluster selection, and visual analysis.
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    df_raw = load_csv_from_bytes(uploaded_file.getvalue())
    data_source_label = uploaded_file.name
elif DEFAULT_CSV_PATH.exists():
    df_raw = load_csv_from_path(str(DEFAULT_CSV_PATH))
    data_source_label = str(DEFAULT_CSV_PATH)
else:
    st.info(
        "Required data file not found. Please upload the CSV file from the sidebar."
    )
    st.stop()

is_valid, missing_columns = validate_input_data(df_raw)
if not is_valid:
    st.error("The uploaded CSV does not match the expected SmartCart customer dataset schema.")
    st.write("Missing columns:", missing_columns)
    st.stop()


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
try:
    data = prepare_customer_data(df_raw, age_limit=age_limit, income_limit=income_limit)
except Exception as exc:
    st.error("Something went wrong while preparing the dataset.")
    st.exception(exc)
    st.stop()

x_pca = data["x_pca"]
df_encoded = data["encoded_df"]
labels = run_clustering(x_pca, method=cluster_method, n_clusters=n_clusters)
df_clustered = add_cluster_labels(df_encoded, x_pca, labels)
k_eval_df = evaluate_k_values(x_pca, max_k=max_k)
elbow_k = detect_elbow(k_eval_df)
best_silhouette_row = k_eval_df.loc[k_eval_df["silhouette_score"].idxmax()]
cluster_profile = build_cluster_profile(df_clustered)


# -----------------------------------------------------------------------------
# Top metrics
# -----------------------------------------------------------------------------
st.caption(f"Data source: `{data_source_label}`")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Original customers", f"{len(df_raw):,}")
with m2:
    st.metric("After cleaning", f"{len(df_encoded):,}")
with m3:
    st.metric("Clusters", n_clusters)
with m4:
    st.metric("Best silhouette K", int(best_silhouette_row["k"]), f"score {best_silhouette_row['silhouette_score']:.3f}")

st.divider()


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_clusters, tab_evaluation, tab_data = st.tabs(
    ["Overview", "Cluster Analysis", "K Selection", "Data Preview"]
)


# -----------------------------------------------------------------------------
# Overview tab
# -----------------------------------------------------------------------------
with tab_overview:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("3D PCA customer map")
        pca_fig = px.scatter_3d(
            df_clustered,
            x="PCA1",
            y="PCA2",
            z="PCA3",
            color="Cluster",
            hover_data={
                "Income": ":,.0f",
                "total_spendings": ":,.0f",
                "Age": ":.0f",
                "Recency": ":.0f",
                "PCA1": ":.2f",
                "PCA2": ":.2f",
                "PCA3": ":.2f",
            },
            title=f"{cluster_method} Clustering on PCA Features",
            height=620,
        )
        pca_fig.update_traces(marker=dict(size=4, opacity=0.82))
        pca_fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), legend_title_text="Cluster")
        st.plotly_chart(pca_fig, use_container_width=True)

    with right:
        st.subheader("PCA explained variance")
        explained_variance = pd.DataFrame(
            {
                "Component": ["PCA1", "PCA2", "PCA3"],
                "Explained Variance": data["pca"].explained_variance_ratio_,
            }
        )
        variance_fig = px.bar(
            explained_variance,
            x="Component",
            y="Explained Variance",
            text=explained_variance["Explained Variance"].map(lambda x: f"{x:.1%}"),
            title="Variance captured by PCA components",
            height=350,
        )
        variance_fig.update_layout(yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(variance_fig, use_container_width=True)

        st.subheader("Cluster size")
        cluster_counts = df_clustered["Cluster"].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["Cluster", "Customers"]
        count_fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Customers",
            text="Customers",
            title="Customers per cluster",
            height=300,
        )
        count_fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(count_fig, use_container_width=True)

    st.subheader("Data cleaning impact")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows before outlier filter", f"{data['rows_before_outlier_filter']:,}")
    with c2:
        st.metric("Rows after outlier filter", f"{data['rows_after_outlier_filter']:,}")
    with c3:
        removed = data["rows_before_outlier_filter"] - data["rows_after_outlier_filter"]
        st.metric("Outlier rows removed", f"{removed:,}")


# -----------------------------------------------------------------------------
# Cluster Analysis tab
# -----------------------------------------------------------------------------
with tab_clusters:
    st.subheader("Cluster profile summary")
    display_profile = cluster_profile.copy()
    if "Income" in display_profile.columns:
        display_profile["Income"] = display_profile["Income"].map(format_currency)
    if "total_spendings" in display_profile.columns:
        display_profile["total_spendings"] = display_profile["total_spendings"].map(format_currency)
    st.dataframe(display_profile, use_container_width=True, hide_index=True)

    st.subheader("Income vs total spending")
    scatter_fig = px.scatter(
        df_clustered,
        x="Income",
        y="total_spendings",
        color="Cluster",
        size="total_spendings",
        hover_data={"Age": True, "Recency": True, "total_children": True},
        title="Customer spending behavior by cluster",
        height=520,
    )
    scatter_fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Compare cluster averages")
    metric_options = [col for col in NUMERIC_PROFILE_COLUMNS if col in df_clustered.columns]
    selected_metric = st.selectbox("Select metric", metric_options, index=metric_options.index("total_spendings") if "total_spendings" in metric_options else 0)

    metric_summary = (
        df_clustered.groupby("Cluster")[selected_metric]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("Cluster")
    )
    metric_fig = px.bar(
        metric_summary,
        x="Cluster",
        y=selected_metric,
        text=selected_metric,
        title=f"Average {selected_metric.replace('_', ' ').title()} by cluster",
        height=420,
    )
    metric_fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(metric_fig, use_container_width=True)


# -----------------------------------------------------------------------------
# K Selection tab
# -----------------------------------------------------------------------------
with tab_evaluation:
    st.subheader("Choosing the number of clusters")

    k1, k2 = st.columns(2)
    with k1:
        elbow_text = elbow_k if elbow_k is not None else "Not detected"
        st.metric("Elbow method K", elbow_text)
    with k2:
        st.metric(
            "Highest silhouette score",
            int(best_silhouette_row["k"]),
            f"{best_silhouette_row['silhouette_score']:.3f}",
        )

    combined_fig = go.Figure()
    combined_fig.add_trace(
        go.Scatter(
            x=k_eval_df["k"],
            y=k_eval_df["wcss"],
            mode="lines+markers",
            name="WCSS / Inertia",
            yaxis="y1",
        )
    )
    combined_fig.add_trace(
        go.Scatter(
            x=k_eval_df["k"],
            y=k_eval_df["silhouette_score"],
            mode="lines+markers",
            name="Silhouette Score",
            yaxis="y2",
        )
    )
    combined_fig.update_layout(
        title="Elbow and silhouette score comparison",
        xaxis=dict(title="Number of clusters K"),
        yaxis=dict(title="WCSS / Inertia"),
        yaxis2=dict(title="Silhouette Score", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
        margin=dict(l=0, r=0, t=80, b=0),
    )
    st.plotly_chart(combined_fig, use_container_width=True)

    st.dataframe(k_eval_df.round(4), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Data Preview tab
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset preview")

    if show_raw_data:
        st.markdown("**Raw uploaded data**")
        st.dataframe(df_raw, use_container_width=True)
    else:
        st.caption("Turn on 'Show raw dataset' in the sidebar to view the original CSV.")

    if show_processed_data:
        st.markdown("**Processed and clustered data**")
        st.dataframe(df_clustered, use_container_width=True)
    else:
        st.caption("Turn on 'Show processed dataset' in the sidebar to view the encoded clustering table.")

    st.subheader("Download results")
    csv_bytes = df_clustered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download clustered customers as CSV",
        data=csv_bytes,
        file_name="smartcart_clustered_customers.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "Built from the SmartCart clustering notebook: feature engineering → outlier filtering → one-hot encoding → scaling → PCA → clustering."
)
