"""Web app: streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Smart House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle():
    return pl.load_or_train()


@st.cache_data(show_spinner=False)
def get_clean():
    return pd.read_csv(pl.CLEAN_PATH)


st.title("🏠 Smart House Price Prediction & Market Segmentation")
st.caption("Regression · Random Forest · K-Means · Data Visualization")

if not os.path.exists(pl.BUNDLE_PATH):
    with st.spinner("First run: downloading the dataset from Kaggle and training models "
                    "(this can take a few minutes)..."):
        bundle = get_bundle()
else:
    bundle = get_bundle()
raw = get_clean()
feature_cols = [c for c in raw.columns if c != pl.TARGET]

with st.sidebar:
    st.header("Model summary")
    m = bundle["metrics"]
    st.metric("R² (test)", f"{m['R²']:.3f}")
    st.metric("MAE", f"{m['MAE']:,.0f}")
    st.metric("RMSE", f"{m['RMSE']:,.0f}")
    st.metric("Segments (k)", bundle["best_k"])
    if st.button("Retrain from Kaggle"):
        for p in (pl.BUNDLE_PATH, pl.CLEAN_PATH):
            if os.path.exists(p):
                os.remove(p)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

tab_pred, tab_csv, tab_eda, tab_model, tab_seg = st.tabs(
    ["🔮 Predict", "📄 Batch (CSV)", "📊 Data Explorer", "🎯 Model Results", "🧩 Market Segments"])

# ---------------------------------------------------------------- predict --
with tab_pred:
    st.subheader("Enter house details")
    with st.form("single"):
        cols = st.columns(3)
        values = {}
        for i, c in enumerate(feature_cols):
            with cols[i % 3]:
                s = raw[c]
                if pd.api.types.is_numeric_dtype(s):
                    is_int = pd.api.types.is_integer_dtype(s)
                    cast = int if is_int else float
                    values[c] = st.number_input(
                        c, min_value=cast(s.min()), max_value=cast(s.max()),
                        value=cast(s.median()), step=1 if is_int else 0.5)
                else:
                    values[c] = st.selectbox(c, sorted(s.dropna().unique().tolist()))
        go = st.form_submit_button("Predict price", type="primary")
    if go:
        res = pl.predict(bundle, pd.DataFrame([values])).iloc[0]
        a, b = st.columns(2)
        a.metric("Predicted price", f"${res['Predicted Price']:,.0f}")
        b.metric("Market segment", res["Market Segment"])

# -------------------------------------------------------------------- csv --
with tab_csv:
    st.subheader("Predict prices for many houses")
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in feature_cols))
    st.download_button("Download CSV template",
                       raw[feature_cols].head(5).to_csv(index=False),
                       "template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        try:
            out = pl.predict(bundle, pd.read_csv(up))
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False),
                               "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")

# -------------------------------------------------------------------- eda --
with tab_eda:
    st.subheader("Exploratory data analysis")
    st.write(f"{len(raw):,} houses after cleaning.")
    st.dataframe(raw.head(20), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.pyplot(pl.fig_price_dist(raw))
    c2.pyplot(pl.fig_corr(raw))
    cats = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
    for i in range(0, len(cats), 2):
        row = st.columns(2)
        for j, c in enumerate(cats[i:i + 2]):
            row[j].pyplot(pl.fig_by_cat(raw, c))

# ------------------------------------------------------------------ model --
with tab_model:
    st.subheader("Model comparison (5-fold cross-validation)")
    st.dataframe(bundle["cv_table"].style.format({"CV R²": "{:.3f}", "CV RMSE": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)
    st.write("Tuned Random Forest parameters:", bundle["best_params"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{m['MAE']:,.0f}")
    c2.metric("Test RMSE", f"{m['RMSE']:,.0f}")
    c3.metric("Test R²", f"{m['R²']:.3f}")
    a, b = st.columns(2)
    a.pyplot(pl.fig_actual_pred(bundle))
    b.pyplot(pl.fig_errors(bundle))
    st.pyplot(pl.fig_importance(bundle))

# -------------------------------------------------------------- segments --
with tab_seg:
    st.subheader("Market segmentation (K-Means)")
    st.write(f"Chosen k = **{bundle['best_k']}** (silhouette score "
             f"**{bundle['silhouette']:.3f}**), picked with the silhouette method.")
    st.pyplot(pl.fig_elbow(bundle))
    st.pyplot(pl.fig_pca(bundle))
    st.markdown("**Segment profiles** (named by average price)")
    st.dataframe(bundle["profile"], use_container_width=True)