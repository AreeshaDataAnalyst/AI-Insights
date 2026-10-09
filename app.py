
import os
import json

import pandas as pd
import numpy as np
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from the project root.
load_dotenv()

API_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom design ----------
st.markdown(
    """
    <style>
    .stApp {
        background: #F5F7FB;
        color: #172033;
    }
    [data-testid="stSidebar"] {
        background: #EDF1F8;
    }
    h1, h2, h3, p, label {
        color: #172033;
    }
    .hero {
        padding: 25px 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #E3ECFF, #E8F7F3);
        border: 1px solid #D9E4F5;
        margin-bottom: 22px;
    }
    .hero-title {
        font-size: 30px;
        font-weight: 750;
        color: #18263F;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #45546D;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E1E7F0;
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(35, 55, 90, 0.04);
    }
    div[data-testid="stMetricLabel"] p {
        color: #56647A !important;
    }
    div[data-testid="stMetricValue"] {
        color: #18263F !important;
    }
    .section-note {
        color: #56647A;
        font-size: 14px;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def load_uploaded_file(uploaded_file):
    """Read a supported CSV or Excel file."""
    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if extension == "csv":
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")

    if extension in ("xlsx", "xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV, XLSX or XLS file.")


def make_ai_payload(df, question):
    """Prepare safe, compact dataset context for the AI endpoint."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = [
        column for column in df.columns
        if column not in numeric_columns
    ]

    # JSON conversion makes NaN and NumPy values API-safe.
    if numeric_columns:
        numeric_summary = json.loads(
            df[numeric_columns].describe().round(2).to_json()
        )
    else:
        numeric_summary = {}

    categorical_summary = {}
    for column in categorical_columns[:30]:
        counts = df[column].fillna("(missing)").astype(str)
        categorical_summary[column] = (
            counts.value_counts().head(8).to_dict()
        )

    missing_by_column = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
    }

    # Only send a small sample, not the complete dataset.
    sample_rows = json.loads(
        df.head(8).to_json(orient="records", date_format="iso")
    )

    return {
        "question": question,
        "columns": [str(column) for column in df.columns],
        "rows": int(len(df)),
        "missing_by_column": missing_by_column,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "sample_rows": sample_rows,
    }


def show_empty_state():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">Your data. Clearer decisions.</div>
          <div class="hero-subtitle">
            Upload a CSV or Excel file to explore your data,
            review its quality, build charts and ask AI questions.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Upload your dataset using the sidebar to get started.")


# ---------- Sidebar ----------
st.sidebar.markdown("## 📊 InsightFlow AI")
st.sidebar.caption("Interactive Data Analytics Workspace")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, XLSX and XLS.",
)

st.sidebar.caption(
    "Your dataset is processed by this app. "
    "AI questions send summaries and up to 8 sample rows to Groq."
)

if uploaded_file is None:
    show_empty_state()
    st.stop()

# ---------- Read data ----------
try:
    df = load_uploaded_file(uploaded_file)
except Exception as error:
    st.error(f"Could not read this file: {error}")
    st.stop()

if df.empty:
    st.warning("This file contains no data rows.")
    st.stop()

# Clean column labels and make duplicate labels unique.
df.columns = [str(column).strip() for column in df.columns]

if any(column == "" for column in df.columns):
    df.columns = [
        column if column else f"Unnamed_{index + 1}"
        for index, column in enumerate(df.columns)
    ]

if df.columns.duplicated().any():
    df.columns = [
        f"{column}_{index + 1}" if duplicated else column
        for index, (column, duplicated)
        in enumerate(zip(df.columns, df.columns.duplicated()))
    ]

# ---------- Header ----------
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">InsightFlow AI</div>
      <div class="hero-subtitle">
        An intelligent workspace for exploring
        <b>{uploaded_file.name}</b>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Key statistics ----------
numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = [
    column for column in df.columns
    if column not in numeric_columns
]

missing_count = int(df.isna().sum().sum())
duplicate_count = int(df.duplicated().sum())

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Total Rows", f"{len(df):,}")
metric2.metric("Total Columns", f"{len(df.columns):,}")
metric3.metric("Missing Values", f"{missing_count:,}")
metric4.metric("Duplicate Rows", f"{duplicate_count:,}")

st.caption(
    f"Numeric columns: {len(numeric_columns)} · "
    f"Categorical/other columns: {len(categorical_columns)}"
)

# ---------- Main navigation ----------
dashboard_tab, quality_tab, explore_tab, ai_tab = st.tabs(
    [
        "📈 Dashboard",
        "🧹 Data Quality",
        "🔎 Explore Data",
        "✨ Ask AI",
    ]
)

# ---------- Dashboard ----------
with dashboard_tab:
    st.subheader("Dataset Overview")
    st.markdown(
        '<p class="section-note">Explore distributions and patterns in your data.</p>',
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("#### Numeric Distribution")

        if numeric_columns:
            selected_numeric = st.selectbox(
                "Choose a numeric column",
                numeric_columns,
                key="distribution_column",
            )

            chart_data = df[[selected_numeric]].dropna()
            if not chart_data.empty:
                fig = px.histogram(
                    chart_data,
                    x=selected_numeric,
                    nbins=25,
                    title=f"Distribution of {selected_numeric}",
                    template="plotly_white",
                )
                fig.update_layout(
                    title_x=0.03,
                    margin=dict(l=15, r=15, t=55, b=15),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("This column has no usable numeric values.")
        else:
            st.info("No numeric columns were detected in this dataset.")

    with chart_right:
        st.markdown("#### Category Breakdown")

        if categorical_columns:
            selected_category = st.selectbox(
                "Choose a category column",
                categorical_columns,
                key="category_column",
            )

            category_counts = (
                df[selected_category]
                .fillna("(missing)")
                .astype(str)
                .value_counts()
                .head(12)
                .rename_axis(selected_category)
                .reset_index(name="Count")
            )

            fig = px.bar(
                category_counts,
                x="Count",
                y=selected_category,
                orientation="h",
                title=f"Top values in {selected_category}",
                template="plotly_white",
            )
            fig.update_layout(
                title_x=0.03,
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=15, r=15, t=55, b=15),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns were detected.")

    if numeric_columns:
        st.markdown("#### Numeric Summary")
        st.dataframe(
            df[numeric_columns].describe().T.round(2),
            use_container_width=True,
        )

# ---------- Data Quality ----------
with quality_tab:
    st.subheader("Data Quality Checks")

    quality_left, quality_right = st.columns(2)

    with quality_left:
        st.markdown("#### Missing Values by Column")
        missing_by_column = df.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]

        if not missing_by_column.empty:
            missing_frame = (
                missing_by_column
                .rename_axis("Column")
                .reset_index(name="Missing Values")
            )
            fig = px.bar(
                missing_frame,
                x="Column",
                y="Missing Values",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found.")

    with quality_right:
        st.markdown("#### Duplicate Rows")
        if duplicate_count:
            st.warning(
                f"{duplicate_count:,} duplicate rows detected. "
                "Review them before removing anything."
            )
        else:
            st.success("No duplicate rows found.")

        st.markdown("#### Column Data Types")
        dtype_frame = pd.DataFrame(
            {
                "Column": df.columns.astype(str),
                "Data Type": [str(dtype) for dtype in df.dtypes],
                "Missing Values": [
                    int(value) for value in df.isna().sum().values
                ],
            }
        )
        st.dataframe(dtype_frame, use_container_width=True, hide_index=True)

    st.markdown("#### Preview Rows with Missing Values")
    rows_with_missing = df[df.isna().any(axis=1)]

    if not rows_with_missing.empty:
        st.dataframe(
            rows_with_missing.head(100),
            use_container_width=True,
        )
        st.caption("Showing at most 100 rows with missing values.")
    else:
        st.info("There are no rows with missing values.")

# ---------- Explore Data ----------
with explore_tab:
    st.subheader("Explore Your Dataset")

    selected_columns = st.multiselect(
        "Select columns to display",
        options=df.columns.tolist(),
        default=df.columns.tolist()[: min(8, len(df.columns))],
    )

    if selected_columns:
        filtered_df = df[selected_columns]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=380,
        )

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download displayed data as CSV",
            data=csv_data,
            file_name="insightflow_export.csv",
            mime="text/csv",
        )
    else:
        st.info("Select at least one column to display.")

    st.markdown("#### Filter Rows")

    filter_column = st.selectbox(
        "Choose a column to filter",
        options=df.columns.tolist(),
        key="filter_column",
    )

    filter_values = (
        df[filter_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if 0 < len(filter_values) <= 100:
        chosen_values = st.multiselect(
            "Choose one or more values",
            options=filter_values,
            key="filter_values",
        )

        if chosen_values:
            mask = df[filter_column].astype(str).isin(chosen_values)
            st.write(f"Matching rows: {int(mask.sum()):,}")
            st.dataframe(df[mask], use_container_width=True)
        else:
            st.caption("Select values to apply the filter.")
    else:
        st.caption(
            "This column has many unique values or no non-missing values. "
            "Use the column selector above to inspect the data."
        )

# ---------- Ask AI ----------
with ai_tab:
    st.subheader("Ask InsightFlow AI")
    st.write(
        "Ask questions about the uploaded dataset in everyday language."
    )

    with st.expander("Example questions"):
        st.markdown(
            """
            - What are the main patterns in this dataset?
            - Which numeric columns have the highest average?
            - What data quality issues should I investigate?
            - Summarize the categorical values.
            - What additional analysis would be useful?
            """
        )

    question = st.text_area(
        "Your question",
        placeholder="Example: What patterns do you notice in this data?",
        height=110,
        max_chars=1500,
        key="ai_question",
    )

    if st.button("Generate Insights", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Analyzing dataset summaries..."):
                    payload = make_ai_payload(df, question.strip())

                    response = requests.post(
                        f"{API_URL}/insights",
                        json=payload,
                        timeout=100,
                    )

                    if response.status_code != 200:
                        try:
                            detail = response.json().get(
                                "detail",
                                response.text[:400],
                            )
                        except ValueError:
                            detail = response.text[:400]

                        st.error(f"AI request failed: {detail}")
                    else:
                        result = response.json()
                        st.markdown("### AI Insights")
                        st.markdown(result.get("answer", "No answer returned."))
                        st.caption(
                            f"Model: {result.get('model', 'Not specified')}"
                        )

            except requests.ConnectionError:
                st.error(
                    "FastAPI backend is not running. "
                    "Start the backend in a separate terminal first."
                )
            except requests.Timeout:
                st.error("The request took too long. Please try again.")
            except Exception as error:
                st.error(f"Something went wrong: {error}")

# ---------- Footer ----------
st.divider()
st.caption(
    "InsightFlow AI · CSV & Excel Analytics · Interactive Charts · "
    "Data Quality · Groq-powered AI"
)