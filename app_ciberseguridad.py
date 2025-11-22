# app_ciberseguridad.py
# ============================================================
# Streamlit App - Ciberseguridad (EDA + Modelos + Anomalías)
# Basado en el notebook Taller_FINAL_CIBERSEGURIDAD.ipynb
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(
    page_title="Ciberseguridad: EDA + Modelos",
    layout="wide",
)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)


@st.cache_data(show_spinner=False)
def read_any_file(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """Lee CSV o XLSX en un DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Sube un .csv o .xlsx")


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Intenta convertir automáticamente columnas numéricas que vengan como texto."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[,\s]", "", regex=True)
                )
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
    return df


def plot_counts(df, col, title):
    fig, ax = plt.subplots()
    sns.countplot(x=df[col], ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Conteo")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def plot_confusion_matrix(cm, title="Matriz de confusión", cmap="Blues"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=["Pred Normal", "Pred Attack"],
                yticklabels=["Real Normal", "Real Attack"])
    ax.set_title(title)
    st.pyplot(fig)


st.sidebar.header("Carga de datos")
uploaded = st.sidebar.file_uploader("Sube tu dataset (.csv o .xlsx)", type=["csv", "xlsx"])

st.sidebar.header("Parámetros")
test_size = st.sidebar.slider("Test size (porcentaje para prueba)", 10, 40, 30, 5) / 100.0
max_depth_dt = st.sidebar.slider("Árbol - max_depth", 2, 20, 5, 1)
n_estimators_rf = st.sidebar.slider("RandomForest - n_estimators", 50, 500, 200, 50)
contamination_if = st.sidebar.slider("IsolationForest - contamination", 1, 49, 50, 1) / 100.0
n_neighbors_lof = st.sidebar.slider("LOF - n_neighbors", 5, 60, 20, 1)
sample_rows = st.sidebar.number_input("(Opcional) muestrear filas (0 = sin muestreo)", min_value=0, value=0, step=1000)

st.title("🔐 Ciberseguridad: EDA + Modelos + Detección de Anomalías")

st.markdown("""
Esta app replica el flujo del notebook: **EDA → preparación → modelos supervisados → anomalías**.  
1) Sube tu archivo (CSV/XLSX) con columnas típicas:  
- `Label` (Normal/Attack)  
- `Protocol` (TCP/UDP/ICMP/…)  
- Numéricas (p.ej. `Duration`, `PacketCount`, `ByteCount`, `SourcePort`, `DestinationPort`)  
""")


if not uploaded:
    st.info("Sube un archivo para comenzar.")
    st.stop()


df = read_any_file(uploaded)
df = safe_numeric(df)

if sample_rows and sample_rows > 0 and sample_rows < len(df):
    df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)

st.success(f"Dataset cargado: **{uploaded.name}** — {df.shape[0]:,} filas x {df.shape[1]} columnas")


st.subheader("🔧 Mapeo de columnas")
cols = df.columns.tolist()

col1, col2, col3 = st.columns(3)
with col1:
    label_col = st.selectbox("Columna de etiqueta binaria (Label)", options=cols, index=cols.index("Label") if "Label" in cols else 0)
with col2:
    protocol_col = st.selectbox("Columna de Protocolo", options=cols, index=cols.index("Protocol") if "Protocol" in cols else 0)

unique_lab = df[label_col].dropna().unique().tolist()
if len(unique_lab) < 2:
    st.error("La columna de etiqueta necesita al menos 2 categorías.")
    st.stop()

colA, colB = st.columns(2)
with colA:
    normal_value = st.selectbox("Valor que representa 'Normal'", options=unique_lab, index=0)
with colB:
    attack_value = st.selectbox("Valor que representa 'Attack'", options=unique_lab, index=1 if len(unique_lab) > 1 else 0)

df["attack_detected"] = df[label_col].map({normal_value: 0, attack_value: 1})
if df["attack_detected"].isna().any():
    st.warning("Hay valores en Label que no coinciden con Normal/Attack seleccionados. Se ignorarán en algunos cálculos.")

encoder_protocol = LabelEncoder()
try:
    df["Protocol_encoded"] = encoder_protocol.fit_transform(df[protocol_col].astype(str))
except Exception:
    st.warning("No se pudo codificar la columna de protocolo. Se creará Protocol_encoded=0.")
    df["Protocol_encoded"] = 0


st.subheader("1) Exploración inicial")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.metric("Filas", f"{len(df):,}")
with c2:
    st.metric("Columnas", f"{df.shape[1]}")
with c3:
    st.metric("Etiquetas únicas en Label", f"{df[label_col].nunique()}")

with st.expander("Ver primeras y últimas 5 filas"):
    st.write("Primeras 5 filas")
    st.dataframe(df.head())
    st.write("Últimas 5 filas")
    st.dataframe(df.tail())

with st.expander("Tipos de datos y valores faltantes"):
    st.write("Tipos de datos:")
    st.write(df.dtypes)
    st.write("Valores faltantes por columna:")
    st.write(df.isnull().sum())

col_info1, col_info2 = st.columns(2)
with col_info1:
    plot_counts(df, label_col, f"Distribución de {label_col}")
with col_info2:
    if protocol_col in df.columns:
        plot_counts(df, protocol_col, f"Distribución de {protocol_col}")


st.subheader("2) Estadísticos y correlación")

drop_cols = [label_col, protocol_col, "Protocol_encoded", "SourceIP", "DestinationIP"]
drop_cols = [c for c in drop_cols if c in df.columns]

df_num = df.drop(columns=drop_cols, errors="ignore")
df_num = df_num.select_dtypes(include=[np.number]).copy()

col_a, col_b = st.columns([1, 1])
with col_a:
    st.write("Estadísticos descriptivos:")
    st.dataframe(df_num.describe())

with col_b:
    st.write("Matriz de correlación (numéricas):")
    if df_num.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_num.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No hay suficientes columnas numéricas para correlación.")

st.write("Boxplots por variable vs `attack_detected`")
vars_for_box = df_num.columns.tolist()
sel_vars = st.multiselect("Elige variables para boxplot", options=vars_for_box, default=vars_for_box[:5])
for col in sel_vars:
    fig, ax = plt.subplots()
    sns.boxplot(x=df["attack_detected"], y=df[col], ax=ax)
    ax.set_title(f"{col} según ataque (0=Normal,1=Attack)")
    st.pyplot(fig)

st.write("Histogramas de variables numéricas")
if not df_num.empty:
    fig, axs = plt.subplots(
        nrows=int(np.ceil(len(df_num.columns)/3)),
        ncols=3,
        figsize=(15, 4 * int(np.ceil(len(df_num.columns)/3)))
    )
    axs = axs.flatten()
    for i, col in enumerate(df_num.columns):
        sns.histplot(df_num[col].dropna(), bins=30, ax=axs[i])
        axs[i].set_title(col)
    for j in range(i+1, len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    st.pyplot(fig)


st.subheader("3) Modelos supervisados (Árbol y Random Forest)")

features_base = df_num.columns.tolist()
if "Protocol_encoded" in df.columns:
    features = features_base + ["Protocol_encoded"]
else:
    features = features_base

data_sup = df.dropna(subset=features + ["attack_detected"]).copy()
if len(data_sup) < 10:
    st.warning("No hay suficientes filas tras limpiar NaN para entrenar modelos.")
else:
    X = data_sup[features]
    y = data_sup["attack_detected"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    st.write(f"Train: {X_train.shape}, Test: {X_test.shape}")

    st.markdown("**Árbol de Decisión**")
    dt_clf = DecisionTreeClassifier(criterion="gini", max_depth=max_depth_dt, random_state=42)
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)

    acc_dt = accuracy_score(y_test, y_pred_dt)
    st.write(f"Accuracy Árbol de Decisión: **{acc_dt:.4f}**")
    st.text("Classification Report - Árbol de Decisión")
    st.text(classification_report(y_test, y_pred_dt, target_names=["Normal", "Attack"]))

    cm_dt = confusion_matrix(y_test, y_pred_dt)
    plot_confusion_matrix(cm_dt, title="Matriz de confusión - Árbol de Decisión", cmap="Oranges")

    st.markdown("**Random Forest**")
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators_rf, max_depth=None, random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    st.write(f"Accuracy Random Forest: **{acc_rf:.4f}**")
    st.text("Classification Report - Random Forest")
    st.text(classification_report(y_test, y_pred_rf, target_names=["Normal", "Attack"]))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plot_confusion_matrix(cm_rf, title="Matriz de confusión - Random Forest", cmap="Greens")

    try:
        importances = pd.Series(rf_clf.feature_importances_, index=features).sort_values(ascending=False)
        st.write("**Importancias de variables (Random Forest)**")
        fig, ax = plt.subplots()
        sns.barplot(x=importances.values, y=importances.index, ax=ax, palette="viridis")
        ax.set_xlabel("Importancia")
        ax.set_ylabel("Variable")
        st.pyplot(fig)
    except Exception:
        st.info("No fue posible calcular importancias.")


st.subheader("4) Detección de anomalías (IsolationForest y LOF)")
cols_anom = ["Duration", "SourcePort", "DestinationPort", "PacketCount", "ByteCount", "Protocol_encoded"]
cols_anom = [c for c in cols_anom if c in df.columns]
data_anom = df.dropna(subset=cols_anom).copy()

if len(data_anom) < 10:
    st.info("No hay suficientes filas para anomalías.")
else:
    X_iso = data_anom[cols_anom]

    st.markdown("**IsolationForest**")
    iso = IsolationForest(n_estimators=200, contamination=contamination_if, random_state=42)
    iso.fit(X_iso)
    data_anom["anomaly_iso"] = iso.predict(X_iso)
    data_anom["anomaly_detected"] = data_anom["anomaly_iso"].map({1: 0, -1: 1})

    st.write("Distribución (0=normal, 1=anomalía):")
    st.write(data_anom["anomaly_detected"].value_counts())

    if "attack_detected" in data_anom.columns:
        st.write("Cruce attack_detected vs anomaly_detected")
        st.dataframe(pd.crosstab(
            data_anom["attack_detected"], data_anom["anomaly_detected"],
            rownames=["Real (Label)"], colnames=["Anomalía"]
        ))

    st.markdown("**LocalOutlierFactor (LOF)**")
    lof = LocalOutlierFactor(n_neighbors=n_neighbors_lof, contamination=contamination_if, novelty=False)
    y_pred_lof = lof.fit_predict(X_iso)
    data_anom["anomaly_lof"] = pd.Series(y_pred_lof, index=data_anom.index).map({1: 0, -1: 1})

    st.write("Distribución LOF (0=normal, 1=anomalía):")
    st.write(data_anom["anomaly_lof"].value_counts())

    if "attack_detected" in data_anom.columns:
        st.write("Cruce attack_detected vs anomaly_lof")
        st.dataframe(pd.crosstab(
            data_anom["attack_detected"], data_anom["anomaly_lof"],
            rownames=["Real (Label)"], colnames=["LOF Anomalía"]
        ))

st.success("Listo ✔  Ajusta hiperparámetros en el sidebar y vuelve a entrenar.")
