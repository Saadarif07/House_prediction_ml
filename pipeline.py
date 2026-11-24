"""Smart House Price Prediction & Market Segmentation - ML pipeline.

Run `python pipeline.py` to train from the command line, or just start the web
app (`streamlit run app.py`), which trains automatically on first launch.
"""
import glob
import os
from datetime import datetime

import joblib
import kagglehub
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET = "zafarali27/house-price-prediction-dataset"
TARGET = "Price"
CURRENT_YEAR = datetime.now().year
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BUNDLE_PATH = os.path.join(MODEL_DIR, "bundle.joblib")
CLEAN_PATH = os.path.join(MODEL_DIR, "clean.csv")
sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------- data ----
def load_raw():
    """Download the dataset from Kaggle with kagglehub and read the CSV."""
    folder = kagglehub.dataset_download(DATASET)
    csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")
    return pd.read_csv(csvs[0])


def clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])
    df = df.drop_duplicates().dropna(subset=[TARGET])
    return df.reset_index(drop=True)


def add_age(df):
    """Feature engineering: YearBuilt -> Age of the house."""
    df = df.copy()
    if "YearBuilt" in df.columns:
        df["Age"] = CURRENT_YEAR - df["YearBuilt"]
        df = df.drop(columns="YearBuilt")
    return df


def make_preprocessor(num, cat):
    return ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), num),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)],
        sparse_threshold=0,
    )


# ------------------------------------------------------------- training ----
def train(log=print):
    os.makedirs(MODEL_DIR, exist_ok=True)
    log("Downloading dataset from Kaggle...")
    raw = clean(load_raw())
    raw.to_csv(CLEAN_PATH, index=False)
    df = add_age(raw)

    X, y = df.drop(columns=TARGET), df[TARGET]
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1) Compare models with 5-fold cross-validation
    log("Cross-validating models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    cv_rows = []
    for name, m in models.items():
        pipe = Pipeline([("prep", make_preprocessor(num, cat)), ("model", m)])
        r2 = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="r2")
        rmse = -cross_val_score(pipe, Xtr, ytr, cv=5, scoring="neg_root_mean_squared_error")
        cv_rows.append({"Model": name, "CV R²": r2.mean(), "CV RMSE": rmse.mean()})
    cv_table = pd.DataFrame(cv_rows).sort_values("CV R²", ascending=False)

    # 2) Tune the Random Forest with grid search
    log("Tuning Random Forest (grid search)...")
    grid = GridSearchCV(
        Pipeline([("prep", make_preprocessor(num, cat)),
                  ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [None, 10, 20],
         "model__min_samples_leaf": [1, 3, 5]},
        cv=5, scoring="r2", n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    pred = best.predict(Xte)
    metrics = {"MAE": mean_absolute_error(yte, pred),
               "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
               "R²": r2_score(yte, pred)}

    names = [n.split("__", 1)[1] for n in best.named_steps["prep"].get_feature_names_out()]
    importance = (pd.DataFrame({"Feature": names,
                                "Importance": best.named_steps["model"].feature_importances_})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))

    # 3) K-Means market segmentation (features only, no price)
    log("Clustering market segments...")
    prep_c = make_preprocessor(num, cat).fit(X)
    Z = prep_c.transform(X)
    ks, inertias, sils = list(range(2, 7)), [], []
    rng = np.random.RandomState(42)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Z, km.labels_, sample_size=min(5000, len(Z)), random_state=42))
    best_k = ks[int(np.argmax(sils))]
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Z)

    order = pd.Series(y.values).groupby(km.labels_).mean().sort_values().index.tolist()
    tier_names = {2: ["Budget", "Premium"], 3: ["Budget", "Standard", "Premium"],
                  4: ["Budget", "Economy", "Standard", "Premium"]}.get(
        best_k, [f"Tier {i + 1}" for i in range(best_k)])
    seg_map = {c: tier_names[i] for i, c in enumerate(order)}

    full = df.assign(Segment=[seg_map[c] for c in km.labels_])
    g = full.groupby("Segment")
    profile = g[num + [TARGET]].mean().round(1)
    profile.insert(0, "Houses", g.size())
    for c in cat:
        profile[f"Most common {c}"] = g[c].agg(lambda s: s.mode().iat[0])
    profile = profile.sort_values(TARGET)

    pca = PCA(n_components=2, random_state=42).fit(Z)
    idx = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)
    pts = pca.transform(Z[idx])
    pca_points = pd.DataFrame({"PC1": pts[:, 0], "PC2": pts[:, 1],
                               "Segment": full["Segment"].values[idx]})

    bundle = dict(model=best, prep_c=prep_c, kmeans=km, seg_map=seg_map,
                  features=X.columns.tolist(), metrics=metrics, cv_table=cv_table,
                  best_params=grid.best_params_, importance=importance,
                  y_test=yte.values, y_pred=pred, k_scores=(ks, inertias, sils),
                  best_k=best_k, silhouette=max(sils), profile=profile,
                  pca_points=pca_points)
    joblib.dump(bundle, BUNDLE_PATH)
    log("Done. Models saved.")
    return bundle


def load_or_train(log=print):
    if os.path.exists(BUNDLE_PATH) and os.path.exists(CLEAN_PATH):
        return joblib.load(BUNDLE_PATH)
    return train(log)


# ----------------------------------------------------------- prediction ----
def predict(bundle, new_df):
    """Return the input rows with predicted price and market segment."""
    data = add_age(new_df.copy())
    data.columns = [c.strip() for c in data.columns]
    missing = [c for c in bundle["features"] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = data[bundle["features"]]
    out = new_df.copy()
    out["Predicted Price"] = bundle["model"].predict(X).round(0)
    labels = bundle["kmeans"].predict(bundle["prep_c"].transform(X))
    out["Market Segment"] = [bundle["seg_map"][c] for c in labels]
    return out


# --------------------------------------------------------------- charts ----
def fig_price_dist(df):
    f, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[TARGET], kde=True, ax=ax, color="#4C78A8")
    ax.set_title("Price distribution")
    return f


def fig_corr(df):
    f, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    return f


def fig_by_cat(df, col):
    f, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=col, y=TARGET, ax=ax, palette="viridis", hue=col, legend=False)
    ax.set_title(f"Price by {col}")
    return f


def fig_actual_pred(b):
    f, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(b["y_test"], b["y_pred"], alpha=0.4, s=12)
    lim = [min(b["y_test"].min(), b["y_pred"].min()), max(b["y_test"].max(), b["y_pred"].max())]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set(xlabel="Actual price", ylabel="Predicted price", title="Actual vs predicted")
    ax.legend()
    return f


def fig_errors(b):
    f, ax = plt.subplots(figsize=(6, 4.5))
    sns.histplot(b["y_test"] - b["y_pred"], kde=True, ax=ax, color="#E45756")
    ax.set(xlabel="Prediction error (actual − predicted)", title="Error distribution")
    return f


def fig_importance(b):
    top = b["importance"].head(12).iloc[::-1]
    f, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top["Feature"], top["Importance"], color="#54A24B")
    ax.set_title("Top feature importances")
    return f


def fig_elbow(b):
    ks, inertias, sils = b["k_scores"]
    f, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    a1.plot(ks, inertias, "o-")
    a1.set(xlabel="k", ylabel="Inertia", title="Elbow method")
    a2.plot(ks, sils, "o-", color="#F58518")
    a2.set(xlabel="k", ylabel="Silhouette", title="Silhouette score")
    f.tight_layout()
    return f


def fig_pca(b):
    f, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.scatterplot(data=b["pca_points"], x="PC1", y="PC2", hue="Segment", s=14, ax=ax)
    ax.set_title("Market segments (PCA, 2D)")
    return f


if __name__ == "__main__":
    b = train()
    print("\nCross-validation:\n", b["cv_table"].to_string(index=False))
    print("\nBest params:", b["best_params"])
    print("Test metrics:", {k: round(v, 3) for k, v in b["metrics"].items()})
    print(f"Clusters: {b['best_k']} (silhouette {b['silhouette']:.3f})")
    print(b["profile"])