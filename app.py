import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin


# -------------------------------------------------------------------
# Clases personalizadas utilizadas durante el entrenamiento
# -------------------------------------------------------------------


class DateSorter(BaseEstimator, TransformerMixin):
    """Convierte y ordena por fecha."""

    def _init_(self, date_column: str = "fecha"):
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.date_column in X.columns and not pd.api.types.is_datetime64_any_dtype(X[self.date_column]):
            X[self.date_column] = pd.to_datetime(X[self.date_column])
        if self.date_column in X.columns:
            X = X.sort_values(self.date_column).reset_index(drop=True)
        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Genera features temporales (mes, día, codificaciones cíclicas)."""

    def _init_(self, date_column: str = "fecha"):
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if self.date_column not in X.columns:
            return X

        X = X.copy()
        fechas = pd.to_datetime(X[self.date_column], errors="coerce")

        X["dia_semana"] = fechas.dt.dayofweek
        X["mes"] = fechas.dt.month
        X["is_weekend"] = X["dia_semana"].apply(lambda x: 1 if x >= 5 else 0)

        X["mes_sin"] = np.sin(2 * np.pi * X["mes"] / 12)
        X["mes_cos"] = np.cos(2 * np.pi * X["mes"] / 12)
        X["dia_sin"] = np.sin(2 * np.pi * X["dia_semana"] / 7)
        X["dia_cos"] = np.cos(2 * np.pi * X["dia_semana"] / 7)

        # Campos adicionales presentes en el pipeline original
        X["anio"] = fechas.dt.year
        X["dia"] = fechas.dt.day
        semana = fechas.dt.isocalendar().week
        X["semana_anio"] = semana.astype("Int64").fillna(0).astype(int)

        return X


class TemperatureFeatureCreator(BaseEstimator, TransformerMixin):
    """Crea columnas derivadas de temperatura."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if {"tmax", "tmin"} <= set(X.columns):
            X["t_med"] = (X["tmax"] + X["tmin"]) / 2
            X["t_amp"] = X["tmax"] - X["tmin"]
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """Elimina columnas específicas del DataFrame."""

    def _init_(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.columns_to_drop, errors="ignore")
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsoriza columnas numéricas para mitigar outliers."""

    def _init_(self, lower_percentile=0.01, upper_percentile=0.99, columns=None):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.columns = columns
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is None:
            self.columns = X.columns

        for col in self.columns:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors="coerce").dropna()
                if not series.empty:
                    self.lower_bounds[col] = np.percentile(series, self.lower_percentile * 100)
                    self.upper_bounds[col] = np.percentile(series, self.upper_percentile * 100)
                else:
                    self.lower_bounds[col] = 0
                    self.upper_bounds[col] = 0
        return self

    def transform(self, X: pd.DataFrame):
        X_transformed = X.copy()
        for col in self.columns:
            if col in X_transformed.columns and col in self.lower_bounds:
                X_transformed[col] = np.clip(
                    pd.to_numeric(X_transformed[col], errors="coerce"),
                    self.lower_bounds[col],
                    self.upper_bounds[col],
                )
        return X_transformed


# -------------------------------------------------------------------
# Configuración general
# -------------------------------------------------------------------

TARGET_COLUMN = "cantidad"
#REFERENCE_CSV = "final_entrega2_v2.csv"
REFERENCE_CSV = "final_2024-11-04.csv"

# -------------------------------------------------------------------
# Carga de artefactos entrenados
# -------------------------------------------------------------------


@st.cache_resource
def load_artifacts():
    try:
        fe_pipeline = joblib.load("artifacts/fe_pipeline.joblib")
        preprocessor = joblib.load("artifacts/preprocessor.joblib")
        model = joblib.load("artifacts/model.joblib")
    except ModuleNotFoundError as exc:
        st.error(
            "No se pudo cargar el modelo porque falta una dependencia: "
            f"{exc.name}. Instalá la librería correspondiente (ej. pip install {exc.name})."
        )
        return None, None, None
    except AttributeError:
        st.error(
            "El modelo necesita las clases personalizadas definidas en el notebook original. "
            "Verificá que las clases DateSorter, TemporalFeatureExtractor, TemperatureFeatureCreator, "
            "DropColumns y Winsorizer estén definidas antes de cargar el modelo."
        )
        return None, None, None
    except Exception as exc:
        st.error(f"Error inesperado al cargar los artefactos: {exc}")
        return None, None, None

    return fe_pipeline, preprocessor, model


# -------------------------------------------------------------------
# Datos de referencia para poblar selects y valores por defecto
# -------------------------------------------------------------------


@st.cache_data
def load_reference_data(path: str = REFERENCE_CSV):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.warning(
            f"No se encontró el archivo {path}. Los selectores tendrán opciones vacías."
        )
        return {
            "options": {},
            "defaults": {},
        }
    except Exception as exc:
        st.warning(
            f"No se pudo leer {path} para poblar los selectores: {exc}."
        )
        return {
            "options": {},
            "defaults": {},
        }

    def uniq(col):
        if col not in df.columns:
            return []
        return sorted({str(v).strip() for v in df[col].dropna().unique() if str(v).strip()})

    options = {
        "empresa": uniq("empresa"),
        "linea": uniq("linea"),
        "municipio": uniq("municipio"),
    }

    defaults = {}

    return {"options": options, "defaults": defaults}


# -------------------------------------------------------------------
# Función auxiliar para preparar el registro de entrada
# -------------------------------------------------------------------


def build_input_dataframe(form_data):
    """
    Convierte los valores provenientes del formulario en un DataFrame que el pipeline espera.
    """
    record = {}

    # Campos base
    record["fecha"] = form_data["fecha"].isoformat()
    record["empresa"] = form_data["empresa"]
    record["linea"] = form_data["linea"]
    record["municipio"] = form_data["municipio"]
    
    # Campos climáticos
    record["tmax"] = form_data["tmax"]
    record["tmin"] = form_data["tmin"]
    record["precip"] = form_data["precip"]
    record["viento"] = form_data["viento"]
    
    # Campo de feriado (solo is_feriado se usa, los otros se eliminan en el pipeline)
    record["is_feriado"] = int(form_data["is_feriado"])
    
    # Estos campos se eliminan en el preprocesador pero los incluimos para compatibilidad
    record["tipo_transporte"] = ""
    record["provincia"] = ""
    record["tipo_feriado"] = ""
    record["nombre_feriado"] = ""

    df = pd.DataFrame([record])
    
    # Asegurar tipos correctos
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].fillna("").astype(str)
    
    return df


# -------------------------------------------------------------------
# Aplicación Streamlit
# -------------------------------------------------------------------


st.title("🚌 Predicción de Pasajeros - Modelo Entrenado")
st.caption(
    "Este formulario utiliza el modelo y los pipelines exportados desde el notebook. "
    "*Nota*: El modelo NO usa features de lag, por lo que las predicciones se basan solo en "
    "características temporales y climáticas."
)

fe_pipeline, preprocessor, model = load_artifacts()
reference = load_reference_data()

if any(obj is None for obj in (fe_pipeline, preprocessor, model)):
    st.stop()

select_options = reference["options"]

st.subheader("📋 Ingresá los datos")
st.caption("Completá los campos con la información de la fecha a predecir.")

empresa_opts = select_options.get("empresa") or [""]
linea_opts = select_options.get("linea") or [""]
municipio_opts = select_options.get("municipio") or [""]

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📍 Información del servicio")
        fecha = st.date_input("Fecha", value=pd.Timestamp("2023-01-01").date())
        empresa = st.selectbox("Empresa", options=empresa_opts, index=0)
        linea = st.selectbox("Línea", options=linea_opts, index=0)
        municipio = st.selectbox("Municipio", options=municipio_opts, index=0)

    with col_right:
        st.markdown("### 🌤 Condiciones del día")
        is_feriado = st.selectbox(
            "¿Es feriado?", 
            options=[0, 1], 
            format_func=lambda x: "Sí" if x == 1 else "No", 
            index=0
        )
        tmax = st.number_input("Temperatura máxima (°C)", value=25.0, step=0.5)
        tmin = st.number_input("Temperatura mínima (°C)", value=18.0, step=0.5)
        precip = st.number_input("Precipitación (mm)", value=0.0, step=0.1, min_value=0.0)
        viento = st.number_input("Viento (km/h)", value=10.0, step=0.5, min_value=0.0)

    submitted = st.form_submit_button("🔮 Predecir", use_container_width=True)

if submitted:
    form_values = {
        "fecha": fecha,
        "empresa": empresa,
        "linea": linea,
        "municipio": municipio,
        "is_feriado": is_feriado,
        "tmax": tmax,
        "tmin": tmin,
        "precip": precip,
        "viento": viento,
    }

    input_df = build_input_dataframe(form_values)

    try:
        with st.spinner("Procesando datos..."):
            # Feature engineering
            fe_output = fe_pipeline.transform(input_df)
            
            # Preprocesamiento final
            processed = preprocessor.transform(fe_output)
            
            # Predicción
            prediction = float(model.predict(processed)[0])
        
        st.success("✅ Predicción completada")
        
        # Mostrar resultado
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Resultado de la predicción")
            
            # Si la predicción es negativa, tomar el valor absoluto y advertir
            if prediction < 0:
                st.warning(
                    f"⚠ El modelo devolvió un valor negativo ({prediction:,.0f}). "
                    "Esto indica que los datos ingresados se alejan de los patrones vistos "
                    "durante el entrenamiento."
                )
                prediction_display = abs(prediction)
                st.metric(
                    "Pasajeros estimados (valor absoluto)", 
                    f"{prediction_display:,.0f}",
                    delta="Predicción ajustada"
                )
            else:
                st.metric("Pasajeros estimados", f"{prediction:,.0f}")
        
        with col2:
            st.subheader("📊 Contexto")
            dia_nombres = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            dia_semana = fecha.weekday()
            st.info(f"*Día*: {dia_nombres[dia_semana]}")
            st.info(f"*Fin de semana*: {'Sí' if dia_semana >= 5 else 'No'}")
            st.info(f"*Feriado*: {'Sí' if is_feriado else 'No'}")
            
    except Exception as exc:
        st.error(f"❌ Ocurrió un error durante la predicción: {exc}")
        st.exception(exc)

st.divider()

with st.expander("ℹ Información del modelo"):
    st.markdown("""
    ### Características del modelo
    
    *Campos utilizados para la predicción:*
    - 📅 *Temporales*: Fecha (día, mes, día de semana, features cíclicas)
    - 🌡 *Climáticos*: Temperaturas (máx, mín, media, amplitud), precipitación, viento
    - 🚌 *Servicio*: Empresa, línea, municipio
    - 📆 *Contexto*: Es feriado, es fin de semana
    
    *Campos NO utilizados* (se eliminan en el preprocesamiento):
    - ❌ Tipo de transporte
    - ❌ Provincia
    - ❌ Tipo de feriado
    - ❌ Nombre del feriado
    
    *Nota importante*: Este modelo NO utiliza features de lag (datos históricos previos), 
    por lo que las predicciones se basan únicamente en patrones temporales y contextuales.
    """)

with st.expander("⚙ Detalles técnicos"):
    st.markdown("""
    ### Pipeline de procesamiento
    
    1. *Feature Engineering*:
       - Conversión y ordenamiento de fechas
       - Extracción de features temporales (día, mes, día de semana)
       - Creación de features cíclicas (sin/cos)
       - Cálculo de temperatura media y amplitud térmica
    
    2. *Preprocesamiento*:
       - Eliminación de columnas no utilizadas
       - Winsorización de outliers en variables climáticas
       - Imputación de valores faltantes
       - Normalización (MinMaxScaler)
       - Encoding de variables categóricas (OneHot)
    
    3. *Predicción*:
       - Modelo: Random Forest / XGBoost (según artefactos cargados)
    """)
