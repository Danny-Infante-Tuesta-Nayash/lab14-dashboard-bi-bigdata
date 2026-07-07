
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

st.set_page_config(
    page_title="Dashboard Analítico LAB 14",
    layout="wide"
)

st.title("Dashboard Analítico Personalizado - Laboratorio 14")
st.write("Aplicación de Data Engineering y Data Science para análisis predictivo.")

# ==============================
# CARGA DEL DATASET
# ==============================

df = pd.read_csv("dataset_personal.csv")

st.subheader("Vista general del dataset")
st.dataframe(df.head(10))

# ==============================
# DETECTAR VARIABLE OBJETIVO
# ==============================

posibles_objetivos = [
    "desercion",
    "morosidad",
    "enfermedad",
    "abandono",
    "compra_alta"
]

variable_objetivo = None

for columna in posibles_objetivos:
    if columna in df.columns:
        variable_objetivo = columna

if variable_objetivo is None:
    st.error("No se encontró la variable objetivo en el dataset.")
    st.stop()

st.info(f"Variable objetivo detectada: {variable_objetivo}")

# ==============================
# FILTROS DEL DASHBOARD
# ==============================

st.sidebar.header("Filtros")

if "segmento_edad" in df.columns:
    segmentos = st.sidebar.multiselect(
        "Selecciona segmento de edad",
        options=df["segmento_edad"].unique(),
        default=df["segmento_edad"].unique()
    )
    df_filtrado = df[df["segmento_edad"].isin(segmentos)]
else:
    df_filtrado = df.copy()

if "nivel_riesgo" in df.columns:
    riesgos = st.sidebar.multiselect(
        "Selecciona nivel de riesgo",
        options=df_filtrado["nivel_riesgo"].unique(),
        default=df_filtrado["nivel_riesgo"].unique()
    )
    df_filtrado = df_filtrado[df_filtrado["nivel_riesgo"].isin(riesgos)]

# ==============================
# KPI PRINCIPALES
# ==============================

st.subheader("Indicadores KPI")

total_registros = len(df_filtrado)
tasa_evento = df_filtrado[variable_objetivo].mean() * 100
promedio_ingresos = df_filtrado["ingresos"].mean()
promedio_satisfaccion = df_filtrado["satisfaccion"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de registros", total_registros)

with col2:
    st.metric("Tasa del evento objetivo", f"{tasa_evento:.2f}%")

with col3:
    st.metric("Ingreso promedio", f"{promedio_ingresos:.2f}")

with col4:
    st.metric("Satisfacción promedio", f"{promedio_satisfaccion:.2f}")

# ==============================
# VISUALIZACIÓN 1: GRÁFICO COMPARATIVO
# ==============================

st.subheader("Visualización 1: Nivel de riesgo según variable objetivo")

fig1, ax1 = plt.subplots(figsize=(7, 4))
sns.countplot(
    data=df_filtrado,
    x="nivel_riesgo",
    hue=variable_objetivo,
    ax=ax1
)
ax1.set_title("Nivel de riesgo según variable objetivo")
ax1.set_xlabel("Nivel de riesgo")
ax1.set_ylabel("Cantidad")
st.pyplot(fig1)

# ==============================
# VISUALIZACIÓN 2: DISTRIBUCIÓN ESTADÍSTICA
# ==============================

st.subheader("Visualización 2: Distribución de ingresos")

fig2, ax2 = plt.subplots(figsize=(7, 4))
sns.histplot(
    data=df_filtrado,
    x="ingresos",
    kde=True,
    ax=ax2
)
ax2.set_title("Distribución de ingresos")
ax2.set_xlabel("Ingresos")
ax2.set_ylabel("Frecuencia")
st.pyplot(fig2)

# ==============================
# VISUALIZACIÓN 3: BOXPLOT
# ==============================

st.subheader("Visualización 3: Valores extremos de ingresos")

fig3, ax3 = plt.subplots(figsize=(7, 4))
sns.boxplot(
    data=df_filtrado,
    x=variable_objetivo,
    y="ingresos",
    ax=ax3
)
ax3.set_title("Ingresos según variable objetivo")
ax3.set_xlabel("Variable objetivo")
ax3.set_ylabel("Ingresos")
st.pyplot(fig3)

# ==============================
# VISUALIZACIÓN 4: MAPA DE CALOR
# ==============================

st.subheader("Visualización 4: Mapa de correlaciones")

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(
    df_filtrado.select_dtypes(include=["int64", "float64"]).corr(),
    annot=True,
    cmap="Blues",
    fmt=".2f",
    ax=ax4
)
ax4.set_title("Correlación entre variables numéricas")
st.pyplot(fig4)

# ==============================
# STORYTELLING DE DATOS
# ==============================

st.subheader("Hallazgos principales")

st.write("""
**Hallazgo 1:**  
El dashboard permite observar la proporción de registros que presentan el evento objetivo y compararla con los registros que no lo presentan.

**Hallazgo 2:**  
Las variables económicas, como ingresos e índice de valor, muestran una relación importante con el comportamiento analizado.

**Hallazgo 3:**  
El nivel de riesgo permite segmentar los registros y facilita la identificación de grupos que requieren mayor atención.
""")

st.subheader("Recomendaciones")

st.write("""
**Recomendación 1:**  
Implementar acciones preventivas sobre los registros clasificados con mayor nivel de riesgo.

**Recomendación 2:**  
Monitorear periódicamente variables como ingresos, satisfacción, frecuencia de uso y antigüedad.

**Recomendación 3:**  
Utilizar el análisis predictivo como apoyo para la toma de decisiones organizacionales.
""")

# ==============================
# CONCLUSIÓN
# ==============================

st.subheader("Conclusión general")

st.write("""
El dashboard desarrollado permite visualizar de forma clara los principales indicadores del dataset personalizado. 
Además, facilita la interpretación de patrones relevantes para la toma de decisiones basada en datos.
""")
