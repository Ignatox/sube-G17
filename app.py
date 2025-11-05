import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin


# -------------------------------------------------------------------
# Clases personalizadas utilizadas durante el entrenamiento
# -------------------------------------------------------------------


class DateSorter(BaseEstimator, TransformerMixin):
    """Convierte y ordena por fecha."""

    def __init__(self, date_column: str = "fecha"):
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

    def __init__(self, date_column: str = "fecha"):
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

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.columns_to_drop, errors="ignore")
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsoriza columnas numéricas para mitigar outliers."""

    def __init__(self, lower_percentile=0.01, upper_percentile=0.99, columns=None):
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


class HistoricalProfileEncoder(BaseEstimator, TransformerMixin):
    """Codifica el perfil histórico de cada línea-municipio-día_semana."""

    def __init__(self, group_cols=['linea', 'municipio', 'dia_semana']):
        self.group_cols = group_cols
        self.profiles = {}
        self.global_stats = {}

    def fit(self, X, y):
        """Calcula perfiles históricos desde los datos de entrenamiento"""
        df = X.copy()
        df['cantidad'] = y

        # 1. Estadísticas por grupo (línea + municipio + día de semana)
        group_stats = df.groupby(self.group_cols)['cantidad'].agg([
            'mean', 'std', 'median', 'min', 'max', 'count'
        ]).reset_index()
        self.profiles['main'] = group_stats

        # 2. Estadísticas por línea + municipio
        line_muni_stats = df.groupby(['linea', 'municipio'])['cantidad'].agg([
            'mean', 'std', 'median'
        ]).reset_index()
        line_muni_stats.columns = ['linea', 'municipio', 'lm_mean', 'lm_std', 'lm_median']
        self.profiles['line_muni'] = line_muni_stats

        # 3. Estadísticas por línea + día de semana
        line_day_stats = df.groupby(['linea', 'dia_semana'])['cantidad'].agg([
            'mean', 'std'
        ]).reset_index()
        line_day_stats.columns = ['linea', 'dia_semana', 'ld_mean', 'ld_std']
        self.profiles['line_day'] = line_day_stats

        # 4. Estadísticas por municipio + día de semana
        muni_day_stats = df.groupby(['municipio', 'dia_semana'])['cantidad'].agg([
            'mean', 'std'
        ]).reset_index()
        muni_day_stats.columns = ['municipio', 'dia_semana', 'md_mean', 'md_std']
        self.profiles['muni_day'] = muni_day_stats

        # 5. Estadísticas globales (fallback)
        self.global_stats = {
            'mean': df['cantidad'].mean(),
            'std': df['cantidad'].std(),
            'median': df['cantidad'].median()
        }

        return self

    def transform(self, X):
        """Agrega features basadas en perfiles históricos"""
        X_new = X.copy()

        # Merge con estadísticas principales
        X_new = X_new.merge(self.profiles['main'], on=self.group_cols, how='left')
        X_new = X_new.merge(self.profiles['line_muni'], on=['linea', 'municipio'], how='left')
        X_new = X_new.merge(self.profiles['line_day'], on=['linea', 'dia_semana'], how='left')
        X_new = X_new.merge(self.profiles['muni_day'], on=['municipio', 'dia_semana'], how='left')

        # Rellenar valores faltantes con estadísticas globales
        fill_cols = ['mean', 'std', 'median', 'lm_mean', 'lm_std', 'lm_median',
                     'ld_mean', 'ld_std', 'md_mean', 'md_std']
        for col in fill_cols:
            if col in X_new.columns:
                base_stat = 'mean' if 'mean' in col else 'std' if 'std' in col else 'median'
                X_new[col].fillna(self.global_stats[base_stat], inplace=True)

        # Features derivadas
        X_new['volatility'] = X_new['std'] / (X_new['mean'] + 1)
        X_new['normalized_demand'] = X_new['mean'] / (X_new['lm_mean'] + 1)

        return X_new


class WeatherImpactEncoder(BaseEstimator, TransformerMixin):
    """Codifica cómo el clima afecta históricamente a cada línea-municipio."""

    def __init__(self):
        self.weather_impacts = {}

    def fit(self, X, y):
        """Calcula sensibilidad al clima por grupo"""
        df = X.copy()
        df['cantidad'] = y

        for group in df.groupby(['linea', 'municipio']):
            key = group[0]
            data = group[1]

            if len(data) < 10:
                continue

            # Correlación entre lluvia y demanda
            rain_corr = data[['precip', 'cantidad']].corr().iloc[0, 1]

            # Diferencia de demanda en días lluviosos vs secos
            rainy_days = data[data['precip'] > 5]['cantidad'].mean()
            dry_days = data[data['precip'] <= 5]['cantidad'].mean()
            rain_impact = (rainy_days - dry_days) / dry_days if dry_days > 0 else 0

            # Sensibilidad a temperatura
            temp_corr = data[['t_med', 'cantidad']].corr().iloc[0, 1]

            self.weather_impacts[key] = {
                'rain_correlation': rain_corr if not np.isnan(rain_corr) else 0.0,
                'rain_impact_pct': rain_impact if not np.isnan(rain_impact) else 0.0,
                'temp_correlation': temp_corr if not np.isnan(temp_corr) else 0.0
            }

        return self

    def transform(self, X):
        """Agrega features de impacto climático"""
        X_new = X.copy()

        # Inicializar columnas
        X_new['rain_sensitivity'] = 0.0
        X_new['rain_impact'] = 0.0
        X_new['temp_sensitivity'] = 0.0

        # Aplicar perfiles
        for idx, row in X_new.iterrows():
            key = (row['linea'], row['municipio'])
            if key in self.weather_impacts:
                impacts = self.weather_impacts[key]
                X_new.loc[idx, 'rain_sensitivity'] = impacts['rain_correlation']
                X_new.loc[idx, 'rain_impact'] = impacts['rain_impact_pct']
                X_new.loc[idx, 'temp_sensitivity'] = impacts['temp_correlation']

        # Interacciones clima × sensibilidad
        X_new['adjusted_rain'] = X_new['precip'] * X_new['rain_sensitivity']
        X_new['adjusted_temp'] = X_new['t_med'] * X_new['temp_sensitivity']

        return X_new


class SeasonalityEncoder(BaseEstimator, TransformerMixin):
    """Codifica patrones estacionales (mensuales/semanales) por grupo."""

    def __init__(self):
        self.seasonal_patterns = {}

    def fit(self, X, y):
        """Calcula factores estacionales"""
        df = X.copy()
        df['cantidad'] = y

        # Por línea-municipio-mes
        monthly = df.groupby(['linea', 'municipio', 'mes'])['cantidad'].mean()

        # Normalizar por promedio anual de cada grupo
        for (linea, municipio) in df.groupby(['linea', 'municipio']).groups.keys():
            try:
                subset = monthly.loc[linea, municipio]
                if len(subset) > 0:
                    annual_avg = subset.mean()
                    if annual_avg > 0:
                        for mes in subset.index:
                            key = (linea, municipio, mes)
                            self.seasonal_patterns[key] = subset[mes] / annual_avg
            except (KeyError, IndexError):
                continue

        return self

    def transform(self, X):
        """Agrega factor estacional"""
        X_new = X.copy()
        X_new['seasonal_factor'] = 1.0

        for idx, row in X_new.iterrows():
            key = (row['linea'], row['municipio'], row['mes'])
            if key in self.seasonal_patterns:
                X_new.loc[idx, 'seasonal_factor'] = self.seasonal_patterns[key]

        return X_new


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
    except AttributeError as exc:
        st.error(
            "El modelo necesita las clases personalizadas definidas en el notebook original. "
            "Verificá que las clases DateSorter, TemporalFeatureExtractor, TemperatureFeatureCreator, "
            "DropColumns, Winsorizer, HistoricalProfileEncoder, WeatherImpactEncoder y SeasonalityEncoder "
            f"estén definidas antes de cargar el modelo. Error: {exc}"
        )
        return None, None, None
    except Exception as exc:
        st.error(f"Error inesperado al cargar los artefactos: {exc}")
        return None, None, None

    return fe_pipeline, preprocessor, model


# -------------------------------------------------------------------
# Funciones para visualizaciones
# -------------------------------------------------------------------


@st.cache_data
def load_full_data(path: str = REFERENCE_CSV):
    """Carga el dataset completo para visualizaciones"""
    try:
        df = pd.read_csv(path)
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        # Filtrar filas con NaN en columnas críticas
        df = df.dropna(subset=['tmax', 'tmin', 'precip', 'viento', 'cantidad'], how='any')
        # Agregar features temporales básicas
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['mes'] = df['fecha'].dt.month
        df['anio'] = df['fecha'].dt.year
        df['is_weekend'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
        df['t_med'] = (df['tmax'] + df['tmin']) / 2
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()


def create_temporal_distribution_chart(df):
    """Gráfico 1: Distribución temporal de pasajeros por mes"""
    if df.empty:
        return None
    
    monthly_data = df.groupby(['anio', 'mes'])['cantidad'].mean().reset_index()
    monthly_data['fecha_str'] = monthly_data.apply(lambda x: f"{int(x['anio'])}-{int(x['mes']):02d}", axis=1)
    
    chart = alt.Chart(monthly_data).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('fecha_str:O', 
                title='Período (Año-Mes)',
                axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('cantidad:Q', 
                title='Promedio de Pasajeros',
                scale=alt.Scale(zero=False)),
        color=alt.Color('anio:O', 
                       title='Año',
                       scale=alt.Scale(scheme='category10')),
        tooltip=['fecha_str', 'cantidad:Q', 'anio:O']
    ).properties(
        title='Evolución Temporal del Promedio de Pasajeros por Mes',
        width=700,
        height=400
    ).interactive()
    
    return chart


def create_weekday_pattern_chart(df):
    """Gráfico 2: Patrones por día de la semana"""
    if df.empty:
        return None
    
    dia_nombres = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    weekday_data = df.groupby('dia_semana')['cantidad'].agg(['mean', 'std']).reset_index()
    weekday_data['dia_nombre'] = weekday_data['dia_semana'].apply(lambda x: dia_nombres[int(x)])
    weekday_data['dia_orden'] = weekday_data['dia_semana']
    
    chart = alt.Chart(weekday_data).mark_bar(size=60).encode(
        x=alt.X('dia_nombre:N',
                title='Día de la Semana',
                sort=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']),
        y=alt.Y('mean:Q',
                title='Promedio de Pasajeros',
                scale=alt.Scale(zero=False)),
        color=alt.Color('mean:Q',
                       title='Promedio',
                       scale=alt.Scale(scheme='blues')),
        tooltip=['dia_nombre', alt.Tooltip('mean:Q', format='.0f'), alt.Tooltip('std:Q', format='.0f')]
    ).properties(
        title='Patrón de Pasajeros por Día de la Semana',
        width=700,
        height=400
    ).interactive()
    
    return chart


def create_weather_impact_chart(df):
    """Gráfico 3: Impacto del clima en la demanda"""
    if df.empty:
        return None
    
    # Crear bins de temperatura y precipitación
    df['temp_bin'] = pd.cut(df['t_med'], bins=5, labels=['Muy Frío', 'Frío', 'Templado', 'Cálido', 'Muy Cálido'])
    df['precip_cat'] = pd.cut(df['precip'], bins=[0, 0.1, 5, 20, 1000], 
                              labels=['Sin lluvia', 'Lluvia leve', 'Lluvia moderada', 'Lluvia intensa'])
    
    weather_data = df.groupby(['temp_bin', 'precip_cat'])['cantidad'].mean().reset_index()
    weather_data = weather_data.dropna()
    
    chart = alt.Chart(weather_data).mark_circle(size=200).encode(
        x=alt.X('temp_bin:O',
                title='Temperatura Media',
                sort=['Muy Frío', 'Frío', 'Templado', 'Cálido', 'Muy Cálido']),
        y=alt.Y('precip_cat:O',
                title='Precipitación',
                sort=['Sin lluvia', 'Lluvia leve', 'Lluvia moderada', 'Lluvia intensa']),
        size=alt.Size('cantidad:Q',
                     title='Pasajeros',
                     scale=alt.Scale(range=[50, 500])),
        color=alt.Color('cantidad:Q',
                      title='Pasajeros',
                      scale=alt.Scale(scheme='redyellowgreen')),
        tooltip=['temp_bin', 'precip_cat', alt.Tooltip('cantidad:Q', format='.0f')]
    ).properties(
        title='Impacto del Clima en la Demanda de Pasajeros',
        width=700,
        height=400
    ).interactive()
    
    return chart


def create_line_municipio_comparison(df):
    """Gráfico 4: Comparación entre líneas y municipios (top 10)"""
    if df.empty:
        return None
    
    # Top 10 líneas por promedio de pasajeros
    top_lineas = df.groupby('linea')['cantidad'].mean().nlargest(10).index.tolist()
    df_filtered = df[df['linea'].isin(top_lineas)]
    
    line_data = df_filtered.groupby('linea')['cantidad'].agg(['mean', 'std']).reset_index()
    line_data = line_data.sort_values('mean', ascending=False)
    
    chart = alt.Chart(line_data).mark_bar().encode(
        x=alt.X('linea:N',
                title='Línea',
                sort='-y'),
        y=alt.Y('mean:Q',
                title='Promedio de Pasajeros',
                scale=alt.Scale(zero=False)),
        color=alt.Color('mean:Q',
                       scale=alt.Scale(scheme='viridis')),
        tooltip=['linea', alt.Tooltip('mean:Q', format='.0f'), alt.Tooltip('std:Q', format='.0f')]
    ).properties(
        title='Top 10 Líneas por Promedio de Pasajeros',
        width=700,
        height=400
    ).interactive()
    
    return chart


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

# Configuración de la página
st.set_page_config(
    page_title="SUBE - Predicción de Pasajeros",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚌 Sistema de Predicción de Pasajeros - SUBE")
st.markdown("## **Grupo 17** - Ciencia de Datos 5k9 - 2025")
st.markdown("#### Aplicación interactiva para visualizar y predecir la demanda de pasajeros en líneas de colectivo")

# Navegación por tabs
tab1, tab2 = st.tabs(["📊 Exploración de Datos", "🔮 Predicción"])

# Cargar datos y artefactos
fe_pipeline, preprocessor, model = load_artifacts()
reference = load_reference_data()
df_full = load_full_data()

if any(obj is None for obj in (fe_pipeline, preprocessor, model)):
    st.error("⚠️ No se pudieron cargar los artefactos del modelo. Verificá que los archivos estén en la carpeta 'artifacts/'.")
    st.stop()

select_options = reference["options"]

# ============================================================================
# TAB 1: EXPLORACIÓN DE DATOS
# ============================================================================
with tab1:
    st.header("📊 Visualización e integración de los datos explorados")
    st.markdown("""
    Esta sección presenta visualizaciones interactivas que muestran los patrones y hallazgos 
    más relevantes del análisis de datos de transporte público.
    """)
    
    if df_full.empty:
        st.warning("⚠️ No se pudieron cargar los datos para visualización.")
    else:
        st.info(f"📈 Dataset cargado: {len(df_full):,} registros desde {df_full['fecha'].min().strftime('%Y-%m-%d')} hasta {df_full['fecha'].max().strftime('%Y-%m-%d')}")
        
        # Visualización 1: Evolución temporal
        st.subheader("1️⃣ Evolución Temporal de Pasajeros")
        st.markdown("""
        Este gráfico muestra la evolución del promedio mensual de pasajeros a lo largo del tiempo,
        permitiendo identificar tendencias y patrones estacionales.
        """)
        chart1 = create_temporal_distribution_chart(df_full)
        if chart1:
            st.altair_chart(chart1, use_container_width=True)
        
        st.divider()
        
        # Visualización 2: Patrones por día de la semana
        st.subheader("2️⃣ Patrones por Día de la Semana")
        st.markdown("""
        Análisis de la demanda promedio por día de la semana, mostrando las diferencias 
        entre días laborales y fines de semana.
        """)
        chart2 = create_weekday_pattern_chart(df_full)
        if chart2:
            st.altair_chart(chart2, use_container_width=True)
        
        st.divider()
        
        # Visualización 3: Impacto del clima
        st.subheader("3️⃣ Impacto del Clima en la Demanda")
        st.markdown("""
        Relación entre condiciones climáticas (temperatura y precipitación) y la demanda de pasajeros.
        El tamaño y color de los círculos representan el promedio de pasajeros.
        """)
        chart3 = create_weather_impact_chart(df_full)
        if chart3:
            st.altair_chart(chart3, use_container_width=True)
        
        st.divider()
        
        # Visualización 4: Top líneas
        st.subheader("4️⃣ Top 10 Líneas por Demanda")
        st.markdown("""
        Comparación de las líneas con mayor promedio de pasajeros transportados.
        """)
        chart4 = create_line_municipio_comparison(df_full)
        if chart4:
            st.altair_chart(chart4, use_container_width=True)
        
        st.divider()
        
        # Resumen estadístico
        with st.expander("📈 Resumen Estadístico del Dataset"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Registros", f"{len(df_full):,}")
            with col2:
                st.metric("Promedio Pasajeros", f"{df_full['cantidad'].mean():,.0f}")
            with col3:
                st.metric("Mediana Pasajeros", f"{df_full['cantidad'].median():,.0f}")
            with col4:
                st.metric("Líneas Únicas", f"{df_full['linea'].nunique():,}")
            
            st.dataframe(df_full[['cantidad', 't_med', 'precip', 'viento']].describe(), use_container_width=True)

# ============================================================================
# TAB 2: PREDICCIÓN
# ============================================================================
with tab2:
    st.header("🔮 Predicción de Pasajeros")
    st.markdown("""
    Utilizá esta herramienta para predecir la cantidad de pasajeros que utilizarán una línea 
    de colectivo en una fecha específica, basándote en las condiciones climáticas y características 
    del servicio.
    """)
    
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
           - Codificación de perfiles históricos
           - Análisis de impacto climático
           - Patrones estacionales
        
        2. *Preprocesamiento*:
           - Eliminación de columnas no utilizadas
           - Winsorización de outliers en variables climáticas
           - Imputación de valores faltantes
           - Normalización (MinMaxScaler)
           - Encoding de variables categóricas (OneHot)
        
        3. *Predicción*:
           - Modelo: Linear Regression (según artefactos cargados)
        """)
