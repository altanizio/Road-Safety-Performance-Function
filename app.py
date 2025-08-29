import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from rspf import RSPF

st.set_page_config(page_title="RSPF - Road Safety Performance Function", layout="wide")
st.title("RSPF - Road Safety Performance Function")

uploaded_file = st.file_uploader(
    "Upload file (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"File `{uploaded_file.name}` uploaded successfully!")

        st.subheader("Data Preview")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("Model Settings")
        cols = list(df.columns)
        y = st.selectbox("Dependent variable (Y)", cols)
        xs_options = cols.copy()
        xs_options.remove(y)

        xs = st.multiselect("Independent variables (X)", xs_options)
        log_xs = st.multiselect("Independent variables with log", xs)
        constant = st.checkbox("Include constant", value=True)

        if "rspf_model" not in st.session_state:
            st.session_state.rspf_model = None

        if st.button("Run Model"):
            if not y or not xs:
                st.warning(
                    "Please select at least one dependent (Y) and independent (X) variable!"
                )
            else:
                try:
                    st.session_state.rspf_model = RSPF(
                        data=df, y=y, X=xs, X_log=log_xs, constant=constant
                    )
                    st.success("Model fitted successfully!")
                except Exception as e:
                    st.error(f"Error fitting model: {e}")

        if st.session_state.rspf_model is not None:
            rspf = st.session_state.rspf_model

            st.subheader("Model Metrics")
            st.dataframe(rspf.metrics(), use_container_width=True)

            st.subheader("Model Coefficients")
            st.dataframe(rspf.coef(), use_container_width=True)

            st.subheader("Estimated Formula")
            st.code(rspf._build_rsp_formula(), language="text")

            st.subheader("CURE Plot (Cumulative Residuals)")
            covariate = st.selectbox(
                "Select variable for the plot", xs, key="cov_select"
            )

            if st.button("Plot CURE"):
                if covariate:
                    cure_data = rspf.cureplot(df[covariate])

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=cure_data[covariate],
                            y=cure_data["CumulativeResiduals"],
                            mode="lines+markers",
                            name="Cumulative Residuals",
                            line=dict(color="steelblue"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=cure_data[covariate],
                            y=cure_data["UpperBound"],
                            mode="lines",
                            name="Upper Limit",
                            line=dict(color="lightcoral", dash="dash"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=cure_data[covariate],
                            y=cure_data["LowerBound"],
                            mode="lines",
                            name="Lower Limit",
                            line=dict(color="lightcoral", dash="dash"),
                        )
                    )

                    fig.update_layout(
                        title="CURE Plot",
                        xaxis_title=covariate,
                        yaxis_title="Cumulative Residuals",
                        template="plotly_white",
                        legend=dict(x=1.02, y=1),
                        autosize=True,
                    )

                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading file: {e}")
