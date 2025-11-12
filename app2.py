import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime
import json
import requests
from datetime import date as date_cls

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


class LagFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Crea features de lag para series temporales agrupadas.
    Replica la implementación utilizada en el notebook IGNA_Entrega3.
    """

    def __init__(
        self,
        target_col: str = "cantidad",
        date_col: str = "fecha",
        group_cols=None,
        lags=None,
        rolling_windows=None,
    ):
        self.target_col = target_col
        self.date_col = date_col
        self.group_cols = group_cols or ["linea", "municipio"]
        self.lags = lags or [28]
        self.rolling_windows = rolling_windows or [7, 28]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Asegurar tipo datetime y ordenar
        if self.date_col in X.columns and not pd.api.types.is_datetime64_any_dtype(X[self.date_col]):
            X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")

        if self.date_col in X.columns:
            X = X.sort_values(self.date_col).reset_index(drop=True)

        def add_lags(g: pd.DataFrame):
            g = g.sort_values(self.date_col)

            # Crear lags de la columna objetivo si existe
            if self.target_col in g.columns:
                for lag in self.lags:
                    g[f"lag_{lag}"] = g[self.target_col].shift(lag)

                # Indicadores de lag faltante
                for lag in self.lags:
                    g[f"has_lag_{lag}"] = g[f"lag_{lag}"].isnull().astype(int)

            return g

        if set(self.group_cols).issubset(set(X.columns)):
            X = X.groupby(self.group_cols, group_keys=False).apply(add_lags)
        else:
            X = add_lags(X)

        return X


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
# Funciones para visualizaciones INTERACTIVAS MEJORADAS
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


@st.cache_data(show_spinner=False)
def get_line_options(df: pd.DataFrame):
    if df.empty:
        return []
    return sorted(df['linea'].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def get_municipios_for_line(df: pd.DataFrame, linea: str):
    if df.empty:
        return []
    sub = df[df['linea'].astype(str) == str(linea)]
    return sorted(sub['municipio'].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def get_default_empresa_for_line_muni(df: pd.DataFrame, linea: str, municipio: str) -> str:
    if df.empty:
        return ""
    sub = df[(df['linea'].astype(str) == str(linea)) & (df['municipio'].astype(str) == str(municipio))]
    if sub.empty:
        return ""
    return sub['empresa'].astype(str).mode().iloc[0]


@st.cache_data(show_spinner=False)
def get_default_provincia_for_line_muni(df: pd.DataFrame, linea: str, municipio: str) -> str:
    if df.empty:
        return ""
    sub = df[(df['linea'].astype(str) == str(linea)) & (df['municipio'].astype(str) == str(municipio))]
    if sub.empty:
        return ""
    return sub['provincia'].astype(str).mode().iloc[0]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_municipio_centroid(municipio: str, provincia: str | None = None):
    """Obtiene lat/lon del municipio desde georef. Fallback: centroide de provincia."""
    base = "https://apis.datos.gob.ar/georef/api/municipios"
    params = {"aplanar": "true", "max": 10, "nombre": municipio}
    if provincia:
        params["provincia"] = provincia
    try:
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        munis = js.get("municipios", [])
        if munis:
            m = munis[0]
            lat = pd.to_numeric(m.get("centroide_lat"), errors="coerce")
            lon = pd.to_numeric(m.get("centroide_lon"), errors="coerce")
            if pd.notna(lat) and pd.notna(lon) and lat != 0 and lon != 0:
                return float(lat), float(lon)
    except Exception:
        pass

    # Fallback: centroide de provincia
    if not provincia:
        return None, None
    try:
        r = requests.get(
            "https://apis.datos.gob.ar/georef/api/provincias",
            params={"aplanar": "true", "nombre": provincia, "max": 1},
            timeout=20,
        )
        r.raise_for_status()
        js = r.json()
        provs = js.get("provincias", [])
        if provs:
            p = provs[0]
            lat = pd.to_numeric(p.get("centroide_lat"), errors="coerce")
            lon = pd.to_numeric(p.get("centroide_lon"), errors="coerce")
            if pd.notna(lat) and pd.notna(lon):
                return float(lat), float(lon)
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_weather(lat: float, lon: float, fecha: pd.Timestamp):
    """Obtiene clima diario (tmax, tmin, precip, viento) para una fecha concreta."""
    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
        return None
    day = fecha.strftime("%Y-%m-%d")
    tz = "America/Argentina/Buenos_Aires"
    today = pd.Timestamp.today().normalize()
    if fecha <= today:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"
    try:
        r = requests.get(
            url,
            params={
                "latitude": float(lat),
                "longitude": float(lon),
                "start_date": day,
                "end_date": day,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
                "timezone": tz,
            },
            timeout=30,
        )
        r.raise_for_status()
        js = r.json()
        daily = js.get("daily", {})
        if daily and daily.get("time"):
            idx = 0
            return {
                "tmax": daily.get("temperature_2m_max", [None])[idx],
                "tmin": daily.get("temperature_2m_min", [None])[idx],
                "precip": daily.get("precipitation_sum", [None])[idx],
                "viento": daily.get("windspeed_10m_max", [None])[idx],
            }
    except Exception:
        return None
    return None


# ============================================================================
# NUEVAS FUNCIONES DE VISUALIZACIÓN INTERACTIVA
# ============================================================================


def create_interactive_demand_explorer(df):
    """Gráfico interactivo simplificado: filtrar por municipio"""
    if df.empty:
        return None
    
    # OPTIMIZACIÓN: Agregar por mes si hay muchos datos
    num_days = df['fecha'].nunique()
    if num_days > 1000:
        df['fecha_mes'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
        muni_agg = df.groupby(['municipio', 'fecha_mes'])['cantidad'].sum().reset_index()
        muni_agg.rename(columns={'fecha_mes': 'fecha'}, inplace=True)
    else:
        muni_agg = df.groupby(['municipio', 'fecha'])['cantidad'].sum().reset_index()
    
    # OPTIMIZACIÓN: Top 15 municipios
    top_munis = df.groupby('municipio')['cantidad'].sum().nlargest(15).index.tolist()
    municipios = sorted(top_munis)
    
    # Selector simple
    municipio_selector = alt.binding_select(options=municipios, name='Municipio: ')
    municipio_selection = alt.param(
        bind=municipio_selector,
        value=municipios[0] if municipios else None
    )
    
    # Gráfico simplificado
    chart = alt.Chart(muni_agg).add_params(municipio_selection).transform_filter(
        alt.datum.municipio == municipio_selection
    ).mark_line(
        point=True,
        strokeWidth=3,
        interpolate='monotone'
    ).encode(
        x=alt.X('fecha:T', title='Fecha', axis=alt.Axis(format='%Y-%m')),
        y=alt.Y('cantidad:Q', title='Total Pasajeros', scale=alt.Scale(zero=False)),
        color=alt.value('#1f77b4'),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d'),
            alt.Tooltip('cantidad:Q', format=',.0f', title='Pasajeros')
        ]
    ).properties(
        title='📊 Demanda Total por Municipio',
        width=800,
        height=400
    )
    
    return chart


def create_heatmap_interactive(df):
    """Heatmap interactivo simplificado"""
    if df.empty:
        return None
    
    # Preparar datos agregados por mes y día (promedio general)
    heatmap_data = df.groupby(['mes', 'dia_semana'])['cantidad'].mean().reset_index()
    
    dia_nombres = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    mes_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    heatmap_data['dia_nombre'] = heatmap_data['dia_semana'].map(
        lambda x: dia_nombres[int(x)] if pd.notna(x) else ''
    )
    heatmap_data['mes_nombre'] = heatmap_data['mes'].map(
        lambda x: mes_nombres[int(x)-1] if pd.notna(x) else ''
    )
    
    # Heatmap simple sin filtros complejos
    heatmap = alt.Chart(heatmap_data).mark_rect(
        stroke='white',
        strokeWidth=2,
        cornerRadius=3
    ).encode(
        x=alt.X('dia_nombre:N', 
                title='Día de la Semana',
                sort=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y('mes_nombre:N',
                title='Mes',
                sort=mes_nombres),
        color=alt.Color('cantidad:Q',
                       title='Promedio Pasajeros',
                       scale=alt.Scale(scheme='yelloworangered'),
                       legend=alt.Legend(gradientLength=300)),
        tooltip=[
            alt.Tooltip('mes_nombre:N', title='Mes'),
            alt.Tooltip('dia_nombre:N', title='Día'),
            alt.Tooltip('cantidad:Q', format=',.0f', title='Promedio Pasajeros')
        ]
    ).properties(
        title='🔥 Heatmap: Patrones de Demanda por Día y Mes',
        width=700,
        height=450
    )
    
    return heatmap


def create_weather_scatter_matrix(df):
    """Scatter matrix simplificado con variables climáticas"""
    if df.empty:
        return None
    
    # OPTIMIZACIÓN: Agregar por semana si hay muchos datos
    num_records = df.groupby(['municipio', 'fecha']).size().shape[0]
    if num_records > 5000:
        df['fecha_semana'] = df['fecha'].dt.to_period('W').dt.to_timestamp()
        scatter_data = df.groupby(['fecha_semana']).agg({
            'cantidad': 'sum',
            't_med': 'mean',
            'precip': 'mean',
            'viento': 'mean'
        }).reset_index()
        scatter_data.rename(columns={'fecha_semana': 'fecha'}, inplace=True)
    else:
        scatter_data = df.groupby('fecha').agg({
            'cantidad': 'sum',
            't_med': 'mean',
            'precip': 'mean',
            'viento': 'mean'
        }).reset_index()
    
    # Gráfico 1: Temperatura vs Demanda
    temp_chart = alt.Chart(scatter_data).mark_circle(
        size=60,
        opacity=0.6
    ).encode(
        x=alt.X('t_med:Q', title='Temperatura Media (°C)', scale=alt.Scale(zero=False)),
        y=alt.Y('cantidad:Q', title='Pasajeros', scale=alt.Scale(zero=False)),
        color=alt.Color('precip:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title='Precipitación')),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d'),
            alt.Tooltip('t_med:Q', format='.1f', title='Temp (°C)'),
            alt.Tooltip('cantidad:Q', format=',.0f', title='Pasajeros'),
            alt.Tooltip('precip:Q', format='.1f', title='Precip (mm)')
        ]
    ).properties(
        title='🌡️ Temperatura vs Demanda',
        width=300,
        height=250
    )
    
    # Gráfico 2: Precipitación vs Demanda
    precip_chart = alt.Chart(scatter_data).mark_circle(
        size=60,
        opacity=0.6
    ).encode(
        x=alt.X('precip:Q', title='Precipitación (mm)', scale=alt.Scale(zero=False)),
        y=alt.Y('cantidad:Q', title='Pasajeros', scale=alt.Scale(zero=False)),
        color=alt.Color('t_med:Q', scale=alt.Scale(scheme='redyellowblue'), legend=alt.Legend(title='Temp (°C)')),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d'),
            alt.Tooltip('precip:Q', format='.1f', title='Precip (mm)'),
            alt.Tooltip('cantidad:Q', format=',.0f', title='Pasajeros'),
            alt.Tooltip('t_med:Q', format='.1f', title='Temp (°C)')
        ]
    ).properties(
        title='🌧️ Precipitación vs Demanda',
        width=300,
        height=250
    )
    
    # Gráfico 3: Viento vs Demanda
    viento_chart = alt.Chart(scatter_data).mark_circle(
        size=60,
        opacity=0.6
    ).encode(
        x=alt.X('viento:Q', title='Viento (km/h)', scale=alt.Scale(zero=False)),
        y=alt.Y('cantidad:Q', title='Pasajeros', scale=alt.Scale(zero=False)),
        color=alt.Color('t_med:Q', scale=alt.Scale(scheme='viridis'), legend=None),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d'),
            alt.Tooltip('viento:Q', format='.1f', title='Viento (km/h)'),
            alt.Tooltip('cantidad:Q', format=',.0f', title='Pasajeros'),
            alt.Tooltip('t_med:Q', format='.1f', title='Temp (°C)')
        ]
    ).properties(
        title='💨 Viento vs Demanda',
        width=300,
        height=250
    )
    
    return alt.hconcat(temp_chart, precip_chart, viento_chart).resolve_scale(color='independent')


def create_multi_line_selector(df):
    """Gráfico de líneas múltiples con selector interactivo de líneas"""
    if df.empty:
        return None
    
    # OPTIMIZACIÓN: Top 8 líneas por demanda (reducido de 15 para mejor visualización)
    top_lineas = df.groupby('linea')['cantidad'].mean().nlargest(8).index.tolist()
    df_filtered = df[df['linea'].isin(top_lineas)].copy()
    
    # OPTIMIZACIÓN: Agregar por mes si hay muchos días (>500)
    num_days = df_filtered['fecha'].nunique()
    if num_days > 500:
        df_filtered['fecha_mes'] = df_filtered['fecha'].dt.to_period('M').dt.to_timestamp()
        line_daily = df_filtered.groupby(['linea', 'fecha_mes'])['cantidad'].sum().reset_index()
        line_daily.rename(columns={'fecha_mes': 'fecha'}, inplace=True)
    else:
        # Agregar por línea y fecha
        line_daily = df_filtered.groupby(['linea', 'fecha'])['cantidad'].sum().reset_index()
    
    # Selector múltiple de líneas (Altair 5.0)
    line_selection = alt.selection_point(
        fields=['linea'],
        bind='legend',
        toggle=True
    )
    
    chart = alt.Chart(line_daily).mark_line(
        point=True,
        strokeWidth=2,
        interpolate='monotone'
    ).encode(
        x=alt.X('fecha:T', title='Fecha', axis=alt.Axis(format='%Y-%m')),
        y=alt.Y('cantidad:Q', title='Total Pasajeros', scale=alt.Scale(zero=False)),
        color=alt.Color('linea:N',
                       scale=alt.Scale(scheme='category20'),
                       legend=alt.Legend(
                           title='Líneas (click para seleccionar)',
                           columns=2,
                           symbolLimit=0
                       )),
        opacity=alt.condition(line_selection, alt.value(1.0), alt.value(0.2)),
        strokeWidth=alt.condition(line_selection, alt.value(3), alt.value(1)),
        tooltip=[
            alt.Tooltip('linea:N', title='Línea'),
            alt.Tooltip('fecha:T', format='%Y-%m-%d', title='Fecha'),
            alt.Tooltip('cantidad:Q', format=',.0f', title='Pasajeros')
        ]
    ).add_params(line_selection).properties(
        title='📈 Comparación de Líneas - Click en la leyenda para filtrar',
        width=800,
        height=450
    )
    
    return chart


def create_interactive_dashboard(df):
    """Dashboard completo con múltiples gráficos interconectados"""
    if df.empty:
        return None
    
    # OPTIMIZACIÓN: Agregar por mes si hay muchos días
    num_days = df['fecha'].nunique()
    if num_days > 1000:
        df['fecha_mes'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
        muni_daily = df.groupby(['municipio', 'fecha_mes'])['cantidad'].sum().reset_index()
        muni_daily.rename(columns={'fecha_mes': 'fecha'}, inplace=True)
    else:
        muni_daily = df.groupby(['municipio', 'fecha'])['cantidad'].sum().reset_index()
    
    # OPTIMIZACIÓN: Top 15 municipios
    muni_stats = df.groupby('municipio')['cantidad'].agg(['mean', 'sum']).reset_index()
    muni_stats = muni_stats.sort_values('sum', ascending=False).head(15)
    top_munis = muni_stats['municipio'].tolist()
    
    # Selección global por municipio (afecta a todos los gráficos) (Altair 5.0)
    municipios = sorted(top_munis)
    municipio_selector = alt.binding_select(options=['Todos'] + municipios, name='Municipio: ')
    municipio_selection = alt.param(
        bind=municipio_selector,
        value='Todos'
    )
    
    # Selección por brush en el gráfico temporal
    brush = alt.selection_interval(encodings=['x'])
    
    # Filtrar datos según selección (manejar 'Todos' en el filtro)
    base = alt.Chart(muni_daily).add_params(municipio_selection).transform_filter(
        (municipio_selection == 'Todos') | (alt.datum.municipio == municipio_selection)
    )
    
    # Gráfico 1: Línea temporal (con brush)
    timeline = base.mark_area(
        interpolate='monotone',
        opacity=0.7
    ).encode(
        x=alt.X('fecha:T', title='Fecha'),
        y=alt.Y('cantidad:Q', title='Pasajeros', aggregate='sum'),
        color=alt.value('#4a90e2'),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d'),
            alt.Tooltip('cantidad:Q', format=',.0f', aggregate='sum', title='Total Pasajeros')
        ]
    ).add_params(brush).properties(
        title='📅 Evolución Temporal',
        width=600,
        height=200
    )
    
    # Gráfico 2: Top municipios (se actualiza con filtro)
    top_munis = alt.Chart(muni_stats).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X('municipio:N', title='Municipio', sort='-y'),
        y=alt.Y('sum:Q', title='Total Pasajeros'),
        color=alt.Color('sum:Q', scale=alt.Scale(scheme='viridis'), legend=None),
        tooltip=[
            alt.Tooltip('municipio:N', title='Municipio'),
            alt.Tooltip('sum:Q', format=',.0f', title='Total Pasajeros'),
            alt.Tooltip('mean:Q', format=',.0f', title='Promedio Diario')
        ]
    ).transform_filter(municipio_selection).properties(
        title='🏆 Top 20 Municipios',
        width=600,
        height=200
    )
    
    # Gráfico 3: Distribución (se actualiza con brush)
    distribution = base.mark_bar(
        binSpacing=2,
        opacity=0.8
    ).encode(
        x=alt.X('cantidad:Q', title='Pasajeros', bin=alt.Bin(maxbins=30)),
        y=alt.Y('count()', title='Frecuencia'),
        color=alt.value('#e74c3c'),
        tooltip=[
            alt.Tooltip('cantidad:Q', bin=True, title='Rango Pasajeros'),
            alt.Tooltip('count()', title='Frecuencia')
        ]
    ).transform_filter(brush).properties(
        title='📊 Distribución (filtrada por rango temporal)',
        width=600,
        height=200
    )
    
    return alt.vconcat(timeline, top_munis, distribution)


def create_temporal_distribution_chart(df):
    """Gráfico temporal mejorado con más interactividad"""
    if df.empty:
        return None
    
    # OPTIMIZACIÓN: Ya está agregado por mes, pero podemos agregar por año si hay muchos meses
    num_months = df.groupby(['anio', 'mes']).size().shape[0]
    if num_months > 60:  # Más de 5 años de datos mensuales
        # Agregar por año
        yearly_data = df.groupby('anio')['cantidad'].mean().reset_index()
        yearly_data['mes'] = 6  # Mes medio para visualización
        monthly_data = yearly_data
        monthly_data['fecha_str'] = monthly_data['anio'].astype(str)
        monthly_data['fecha_dt'] = pd.to_datetime(monthly_data['anio'].astype(str) + '-06-01')
    else:
        monthly_data = df.groupby(['anio', 'mes'])['cantidad'].mean().reset_index()
        monthly_data['fecha_str'] = monthly_data.apply(lambda x: f"{int(x['anio'])}-{int(x['mes']):02d}", axis=1)
        monthly_data['fecha_dt'] = pd.to_datetime(monthly_data['anio'].astype(str) + '-' + monthly_data['mes'].astype(str) + '-01')
    
    # Selección por click (Altair 5.0)
    click = alt.selection_point(empty=True)
    
    # Selección por brush
    brush = alt.selection_interval(encodings=['x'])
    
    base = alt.Chart(monthly_data)
    
    # Línea principal
    line = base.mark_line(
        point=True,
        strokeWidth=3,
        interpolate='monotone'
    ).encode(
        x=alt.X('fecha_dt:T', 
                title='Período',
                axis=alt.Axis(format='%Y-%m')),
        y=alt.Y('cantidad:Q', 
                title='Promedio de Pasajeros',
                scale=alt.Scale(zero=False)),
        color=alt.Color('anio:O', 
                       title='Año',
                       scale=alt.Scale(scheme='category10')),
        tooltip=[
            alt.Tooltip('fecha_str:O', title='Período'),
            alt.Tooltip('cantidad:Q', format=',.0f', title='Promedio'),
            alt.Tooltip('anio:O', title='Año')
        ],
        opacity=alt.condition(brush, alt.value(1.0), alt.value(0.3))
    ).add_params(brush, click)
    
    # Puntos destacados
    points = base.mark_circle(
        size=150,
        opacity=0.8
    ).encode(
        x=alt.X('fecha_dt:T'),
        y=alt.Y('cantidad:Q'),
        color=alt.condition(
            click,
            alt.value('#ff7f0e'),
            alt.Color('anio:O', scale=alt.Scale(scheme='category10'))
        ),
        size=alt.condition(click, alt.value(300), alt.value(150)),
        tooltip=[
            alt.Tooltip('fecha_str:O', title='Período'),
            alt.Tooltip('cantidad:Q', format=',.0f', title='Promedio'),
            alt.Tooltip('anio:O', title='Año')
        ]
    ).add_params(click)
    
    chart = (line + points).properties(
        title='📈 Evolución Temporal del Promedio de Pasajeros por Mes',
        width=800,
        height=400
    )
    
    return chart


# ============================================================================
# GRÁFICOS MEJORADOS PARA PREDICCIÓN
# ============================================================================


def create_prediction_timeline_interactive(results_df):
    """Gráfico interactivo de predicciones con desglose por municipio"""
    if results_df.empty:
        return None
    
    results_df = results_df.copy()
    results_df['fecha'] = pd.to_datetime(results_df['fecha'])
    
    # OPTIMIZACIÓN: Si hay muchos municipios (>10), mostrar solo top 10 por predicción total
    num_munis = results_df['municipio'].nunique()
    if num_munis > 10:
        top_munis = results_df.groupby('municipio')['prediccion'].sum().nlargest(10).index.tolist()
        results_df = results_df[results_df['municipio'].isin(top_munis)]
    
    # Selección por click (Altair 5.0)
    click = alt.selection_point(fields=['fecha'], empty=True)
    
    # Selección múltiple de municipios (Altair 5.0)
    municipio_selection = alt.selection_point(
        fields=['municipio'],
        bind='legend',
        toggle=True
    )
    
    # Gráfico 1: Líneas por municipio
    lines_chart = alt.Chart(results_df).mark_line(
        point=True,
        strokeWidth=2,
        interpolate='monotone'
    ).encode(
        x=alt.X('fecha:T', title='Fecha', axis=alt.Axis(format='%Y-%m-%d')),
        y=alt.Y('prediccion:Q', title='Pasajeros Estimados', scale=alt.Scale(zero=False)),
        color=alt.Color('municipio:N',
                       scale=alt.Scale(scheme='category20'),
                       legend=alt.Legend(
                           title='Municipios (click para filtrar)',
                           columns=2
                       )),
        opacity=alt.condition(municipio_selection, alt.value(1.0), alt.value(0.2)),
        strokeWidth=alt.condition(municipio_selection, alt.value(3), alt.value(1)),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d', title='Fecha'),
            alt.Tooltip('municipio:N', title='Municipio'),
            alt.Tooltip('prediccion:Q', format=',.0f', title='Pasajeros')
        ]
    ).add_params(municipio_selection, click).properties(
        title='📊 Predicciones por Municipio',
        width=800,
        height=350
    )
    
    # Gráfico 2: Total agregado
    totals_df = results_df.groupby('fecha', as_index=False)['prediccion'].sum()
    total_chart = alt.Chart(totals_df).mark_area(
        interpolate='monotone',
        opacity=0.6
    ).encode(
        x=alt.X('fecha:T', title='Fecha'),
        y=alt.Y('prediccion:Q', title='Total Pasajeros', scale=alt.Scale(zero=False)),
        color=alt.value('#2b8a3e'),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d', title='Fecha'),
            alt.Tooltip('prediccion:Q', format=',.0f', title='Total Pasajeros')
        ]
    ).add_params(click).properties(
        title='📈 Total Agregado',
        width=800,
        height=200
    )
    
    # Gráfico 3: Desglose por municipio para fecha seleccionada
    detail_chart = alt.Chart(results_df).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X('municipio:N', title='Municipio', sort='-y'),
        y=alt.Y('prediccion:Q', title='Pasajeros', aggregate='sum'),
        color=alt.Color('municipio:N', scale=alt.Scale(scheme='category20'), legend=None),
        tooltip=[
            alt.Tooltip('municipio:N', title='Municipio'),
            alt.Tooltip('prediccion:Q', format=',.0f', aggregate='sum', title='Pasajeros')
        ]
    ).transform_filter(click).properties(
        title='🎯 Desglose por Municipio (click en una fecha)',
        width=800,
        height=250
    )
    
    return alt.vconcat(total_chart, lines_chart, detail_chart).resolve_scale(color='independent')


def create_prediction_weather_comparison(results_df):
    """Comparación de predicciones con variables climáticas"""
    if results_df.empty:
        return None
    
    results_df = results_df.copy()
    results_df['fecha'] = pd.to_datetime(results_df['fecha'])
    
    # Agregar temperatura media
    results_df['t_med'] = (results_df['tmax'] + results_df['tmin']) / 2
    
    # Selección por click
    click = alt.selection_point(fields=['fecha'], empty=True)
    
    # Gráfico 1: Predicción vs Temperatura
    temp_chart = alt.Chart(results_df).mark_circle(
        size=100,
        opacity=0.6
    ).encode(
        x=alt.X('t_med:Q', title='Temperatura Media (°C)', scale=alt.Scale(zero=False)),
        y=alt.Y('prediccion:Q', title='Pasajeros Estimados', scale=alt.Scale(zero=False)),
        color=alt.condition(
            click,
            alt.value('#ff7f0e'),
            alt.Color('precip:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title='Precipitación'))
        ),
        size=alt.condition(click, alt.value(300), alt.value(100)),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d', title='Fecha'),
            alt.Tooltip('municipio:N', title='Municipio'),
            alt.Tooltip('t_med:Q', format='.1f', title='Temp (°C)'),
            alt.Tooltip('prediccion:Q', format=',.0f', title='Pasajeros'),
            alt.Tooltip('precip:Q', format='.1f', title='Precip (mm)')
        ]
    ).add_params(click).properties(
        title='🌡️ Predicción vs Temperatura',
        width=350,
        height=300
    )
    
    # Gráfico 2: Predicción vs Precipitación
    precip_chart = alt.Chart(results_df).mark_circle(
        size=100,
        opacity=0.6
    ).encode(
        x=alt.X('precip:Q', title='Precipitación (mm)', scale=alt.Scale(zero=False)),
        y=alt.Y('prediccion:Q', title='Pasajeros Estimados', scale=alt.Scale(zero=False)),
        color=alt.condition(
            click,
            alt.value('#ff7f0e'),
            alt.Color('t_med:Q', scale=alt.Scale(scheme='redyellowblue'), legend=alt.Legend(title='Temp (°C)'))
        ),
        size=alt.condition(click, alt.value(300), alt.value(100)),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d', title='Fecha'),
            alt.Tooltip('municipio:N', title='Municipio'),
            alt.Tooltip('precip:Q', format='.1f', title='Precip (mm)'),
            alt.Tooltip('prediccion:Q', format=',.0f', title='Pasajeros'),
            alt.Tooltip('t_med:Q', format='.1f', title='Temp (°C)')
        ]
    ).add_params(click).properties(
        title='🌧️ Predicción vs Precipitación',
        width=350,
        height=300
    )
    
    # Gráfico 3: Predicción vs Viento
    viento_chart = alt.Chart(results_df).mark_circle(
        size=100,
        opacity=0.6
    ).encode(
        x=alt.X('viento:Q', title='Viento (km/h)', scale=alt.Scale(zero=False)),
        y=alt.Y('prediccion:Q', title='Pasajeros Estimados', scale=alt.Scale(zero=False)),
        color=alt.condition(
            click,
            alt.value('#ff7f0e'),
            alt.Color('t_med:Q', scale=alt.Scale(scheme='viridis'), legend=None)
        ),
        size=alt.condition(click, alt.value(300), alt.value(100)),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d', title='Fecha'),
            alt.Tooltip('municipio:N', title='Municipio'),
            alt.Tooltip('viento:Q', format='.1f', title='Viento (km/h)'),
            alt.Tooltip('prediccion:Q', format=',.0f', title='Pasajeros'),
            alt.Tooltip('t_med:Q', format='.1f', title='Temp (°C)')
        ]
    ).add_params(click).properties(
        title='💨 Predicción vs Viento',
        width=350,
        height=300
    )
    
    return alt.hconcat(temp_chart, precip_chart, viento_chart).resolve_scale(color='independent')


def create_prediction_heatmap(results_df):
    """Heatmap de predicciones por municipio y fecha"""
    if results_df.empty:
        return None
    
    results_df = results_df.copy()
    results_df['fecha'] = pd.to_datetime(results_df['fecha'])
    
    # OPTIMIZACIÓN: Limitar a top 15 municipios si hay muchos
    num_munis = results_df['municipio'].nunique()
    if num_munis > 15:
        top_munis = results_df.groupby('municipio')['prediccion'].sum().nlargest(15).index.tolist()
        results_df = results_df[results_df['municipio'].isin(top_munis)]
    
    # Selección por brush
    brush = alt.selection_interval()
    
    # Preparar datos para heatmap
    heatmap_data = results_df.pivot_table(
        index='municipio',
        columns='fecha',
        values='prediccion',
        aggfunc='sum'
    ).reset_index().melt(id_vars='municipio', var_name='fecha', value_name='prediccion')
    heatmap_data['fecha'] = pd.to_datetime(heatmap_data['fecha'])
    
    heatmap = alt.Chart(heatmap_data).mark_rect(
        stroke='white',
        strokeWidth=1,
        cornerRadius=2
    ).encode(
        x=alt.X('fecha:T', title='Fecha', axis=alt.Axis(format='%Y-%m-%d', labelAngle=-45)),
        y=alt.Y('municipio:N', title='Municipio', sort='-x'),
        color=alt.Color('prediccion:Q',
                       title='Pasajeros',
                       scale=alt.Scale(scheme='yelloworangered'),
                       legend=alt.Legend(gradientLength=300)),
        tooltip=[
            alt.Tooltip('fecha:T', format='%Y-%m-%d', title='Fecha'),
            alt.Tooltip('municipio:N', title='Municipio'),
            alt.Tooltip('prediccion:Q', format=',.0f', title='Pasajeros')
        ],
        opacity=alt.condition(brush, alt.value(1.0), alt.value(0.7))
    ).add_params(brush).properties(
        title='🔥 Heatmap de Predicciones por Municipio y Fecha',
        width=800,
        height=400
    )
    
    return heatmap


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

# Intentar cargar metadata del modelo para mostrar nombre/modelo
try:
    with open("artifacts/metadata.json", "r") as f:
        _meta = json.load(f)
    MODEL_NAME = _meta.get("model_name", "Modelo entrenado")
except Exception:
    MODEL_NAME = "Modelo entrenado"

if any(obj is None for obj in (fe_pipeline, preprocessor, model)):
    st.error("⚠️ No se pudieron cargar los artefactos del modelo. Verificá que los archivos estén en la carpeta 'artifacts/'.")
    st.stop()

select_options = reference["options"]

# ============================================================================
# TAB 1: EXPLORACIÓN DE DATOS (MEJORADO)
# ============================================================================
with tab1:
    st.header("📊 Visualización Interactiva de Datos")
    st.markdown("""
    Explora los datos históricos con visualizaciones interactivas avanzadas. 
    **Filtra, haz click, selecciona y descubre patrones ocultos.**
    """)
    
    if df_full.empty:
        st.warning("⚠️ No se pudieron cargar los datos para visualización.")
    else:
        st.info(f"📈 Dataset cargado: {len(df_full):,} registros desde {df_full['fecha'].min().strftime('%Y-%m-%d')} hasta {df_full['fecha'].max().strftime('%Y-%m-%d')}")
        
        # Visualización 1: Explorador interactivo de demanda
        st.subheader("🎯 Explorador Interactivo de Demanda")
        st.markdown("""
        **Filtra por municipio → Click en un punto → Ve el desglose por línea**
        
        Usa el selector de municipio para filtrar, luego haz click en cualquier punto del gráfico 
        para ver el desglose detallado por línea en ese día específico.
        """)
        chart_interactive = create_interactive_demand_explorer(df_full)
        if chart_interactive:
            st.altair_chart(chart_interactive, width='stretch')
        
        st.divider()
        
        # Visualización 2: Heatmap interactivo
        st.subheader("🔥 Heatmap: Patrones Temporales")
        st.markdown("""
        **Selecciona un municipio y arrastra sobre el heatmap para ver detalles**
        
        El heatmap muestra los patrones de demanda por día de la semana y mes. 
        Selecciona un área para ver el desglose detallado.
        """)
        chart_heatmap = create_heatmap_interactive(df_full)
        if chart_heatmap:
            st.altair_chart(chart_heatmap, width='stretch')
        
        st.divider()
        
        # Visualización 3: Análisis clima vs demanda
        st.subheader("🌦️ Análisis Clima vs Demanda")
        st.markdown("""
        **Explora cómo el clima afecta la demanda. Click en puntos para destacarlos.**
        
        Tres gráficos interconectados muestran la relación entre temperatura, precipitación, 
        viento y la demanda de pasajeros.
        """)
        chart_weather = create_weather_scatter_matrix(df_full)
        if chart_weather:
            st.altair_chart(chart_weather, width='stretch')
        
        st.divider()
        
        # Visualización 4: Comparación de líneas
        st.subheader("📈 Comparación de Líneas")
        st.markdown("""
        **Click en la leyenda para filtrar líneas específicas**
        
        Compara hasta 15 líneas principales. Haz click en los nombres de las líneas 
        en la leyenda para mostrar/ocultar cada una.
        """)
        chart_lines = create_multi_line_selector(df_full)
        if chart_lines:
            st.altair_chart(chart_lines, width='stretch')
        
        st.divider()
        
        # Visualización 5: Evolución temporal mejorada
        st.subheader("📅 Evolución Temporal Mejorada")
        st.markdown("""
        **Arrastra para seleccionar un rango temporal y click en puntos para destacarlos**
        """)
        chart_temporal = create_temporal_distribution_chart(df_full)
        if chart_temporal:
            st.altair_chart(chart_temporal, width='stretch')
        
        st.divider()
        
        # Visualización 6: Dashboard interactivo
        st.subheader("🎛️ Dashboard Interactivo")
        st.markdown("""
        **Vista general con múltiples gráficos interconectados**
        
        Selecciona un municipio y arrastra sobre el gráfico temporal para filtrar 
        la distribución de pasajeros.
        """)
        chart_dashboard = create_interactive_dashboard(df_full)
        if chart_dashboard:
            st.altair_chart(chart_dashboard, width='stretch')
        
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
            
            st.dataframe(df_full[['cantidad', 't_med', 'precip', 'viento']].describe(), width='stretch')

# ============================================================================
# TAB 2: PREDICCIÓN (MEJORADO)
# ============================================================================
with tab2:
    st.header("🔮 Predicción de Pasajeros")
    st.markdown("""
    Utilizá esta herramienta para predecir la cantidad de pasajeros que utilizarán una línea 
    de colectivo en una fecha específica, basándote en las condiciones climáticas y características 
    del servicio.
    """)
    
    st.subheader("📋 Ingresá los datos")
    st.caption("Elegí la fecha y la línea. La app predecirá para todos los municipios de esa línea.")

    line_opts = get_line_options(df_full)

    total_placeholder = None
    with st.form("prediction_form"):
        col_left, col_right = st.columns([3, 1])

        with col_left:
            st.markdown("### 📍 Selección")
            # Fechas con validación contra último dato del CSV (+16 días)
            last_date = pd.to_datetime(df_full['fecha'].max()) if not df_full.empty else pd.Timestamp.today()
            max_allowed = (last_date + pd.Timedelta(days=16)).date()
            default_start = min((last_date + pd.Timedelta(days=1)).date(), max_allowed) if not df_full.empty else pd.Timestamp.today().date()
            fecha_desde = st.date_input("Fecha desde", value=default_start, max_value=max_allowed)
            fecha_hasta = st.date_input(
                "Predecir hasta",
                value=default_start,
                min_value=fecha_desde,
                max_value=max_allowed,
            )
            linea = st.selectbox("Línea", options=line_opts, index=0)

        with col_right:
            st.markdown("### 🔢 Predicción total")
            total_placeholder = st.empty()
            total_placeholder.info("La suma total del rango aparecerá aquí luego de predecir.")

        submitted = st.form_submit_button("🔮 Predecir", use_container_width=False)

    if submitted:
        # Validación de fechas
        last_date = pd.to_datetime(df_full['fecha'].max()) if not df_full.empty else pd.Timestamp.today()
        max_limit = last_date + pd.Timedelta(days=16)
        fecha_desde_ts = pd.Timestamp(fecha_desde)
        fecha_hasta_ts = pd.Timestamp(fecha_hasta)

        if fecha_desde_ts > fecha_hasta_ts:
            st.error("La fecha 'desde' no puede ser mayor a 'hasta'.")
            st.stop()

        if fecha_hasta_ts > max_limit:
            st.error(
                f"La fecha seleccionada debe ser como máximo 16 días posterior al último dato del CSV ({last_date.date()}). "
                f"Máximo permitido: {max_limit.date()}"
            )
            st.stop()

        # Municipios para la línea seleccionada
        municipios = get_municipios_for_line(df_full, linea)
        if not municipios:
            st.warning("No se encontraron municipios para la línea seleccionada en el CSV.")
            st.stop()

        # Precalcular info por municipio
        muni_info = []
        for muni in municipios:
            prov = get_default_provincia_for_line_muni(df_full, linea, muni)
            emp = get_default_empresa_for_line_muni(df_full, linea, muni)
            lat, lon = fetch_municipio_centroid(muni, prov)
            muni_info.append({
                "municipio": muni,
                "provincia": prov or "",
                "empresa": emp,
                "lat": lat,
                "lon": lon,
            })

        # Construir registros para cada fecha del rango
        records = []
        for single_date in pd.date_range(fecha_desde_ts, fecha_hasta_ts, freq="D"):
            for info in muni_info:
                lat = info["lat"]
                lon = info["lon"]
                weather = fetch_daily_weather(lat, lon, single_date) if (lat is not None and lon is not None) else None

                if weather is None:
                    tmax = tmin = precip = viento = np.nan
                else:
                    tmax = weather.get("tmax")
                    tmin = weather.get("tmin")
                    precip = weather.get("precip")
                    viento = weather.get("viento")

                records.append({
                    "fecha": single_date.date().isoformat(),
                    "empresa": info["empresa"],
                    "linea": linea,
                    "municipio": info["municipio"],
                    "is_feriado": 0,
                    "tmax": tmax,
                    "tmin": tmin,
                    "precip": precip,
                    "viento": viento,
                    "tipo_transporte": "",
                    "provincia": info["provincia"],
                    "tipo_feriado": "",
                    "nombre_feriado": "",
                    "cantidad": np.nan,
                })

        input_df = pd.DataFrame.from_records(records)

        try:
            with st.spinner("Procesando datos..."):
                # Feature engineering
                fe_output = fe_pipeline.transform(input_df)

                # Preprocesamiento final
                processed = preprocessor.transform(fe_output)

                # Predicción
                y_pred = model.predict(processed)
                preds = pd.Series(np.array(y_pred).ravel(), name="prediccion")
                results = pd.concat([input_df[["municipio", "provincia", "linea", "fecha", "tmax", "tmin", "precip", "viento"]].reset_index(drop=True), preds], axis=1)

            st.success("✅ Predicción completada")

            # Mostrar total en el panel derecho junto al formulario
            if total_placeholder is not None:
                total_pred = results["prediccion"].sum()
                total_placeholder.markdown(
                    f"<div style='font-size:1.8rem; font-weight:700; color:#2b8a3e;'>"
                    f"Total estimado (rango):<br>{total_pred:,.0f} pasajeros" \
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Mostrar resultados por municipio (todas las fechas)
            st.subheader("🎯 Predicción por municipio")
            display_results = results.copy()
            display_results["fecha"] = pd.to_datetime(display_results["fecha"])
            display_results = display_results.sort_values(["fecha", "prediccion"], ascending=[True, False])
            st.dataframe(
                display_results.assign(
                    prediccion=lambda d: d["prediccion"].round(0).astype(int)
                ),
                width='stretch',
            )

            # NUEVOS GRÁFICOS INTERACTIVOS DE PREDICCIÓN
            
            # Gráfico 1: Timeline interactivo con desglose
            st.subheader("📊 Predicciones Interactivas por Municipio")
            st.markdown("""
            **Click en la leyenda para filtrar municipios, click en una fecha para ver detalles**
            """)
            chart_pred_timeline = create_prediction_timeline_interactive(display_results)
            if chart_pred_timeline:
                st.altair_chart(chart_pred_timeline, width='stretch')
            
            st.divider()
            
            # Gráfico 2: Comparación con clima
            st.subheader("🌦️ Predicciones vs Condiciones Climáticas")
            st.markdown("""
            **Explora cómo las predicciones se relacionan con el clima. Click en puntos para destacarlos.**
            """)
            chart_pred_weather = create_prediction_weather_comparison(display_results)
            if chart_pred_weather:
                st.altair_chart(chart_pred_weather, width='stretch')
            
            st.divider()
            
            # Gráfico 3: Heatmap de predicciones
            st.subheader("🔥 Heatmap de Predicciones")
            st.markdown("""
            **Vista general de todas las predicciones por municipio y fecha**
            """)
            chart_pred_heatmap = create_prediction_heatmap(display_results)
            if chart_pred_heatmap:
                st.altair_chart(chart_pred_heatmap, width='stretch')

        except Exception as exc:
            st.error(f"❌ Ocurrió un error durante la predicción: {exc}")
            st.exception(exc)

    st.divider()
    
    with st.expander("ℹ Información del modelo"):
        st.markdown(f"""
        ### Características del modelo
        
        *Campos utilizados automáticamente para cada municipio de la línea seleccionada:*
        - 📅 *Temporales*: Fecha, día de la semana, mes, indicadores cíclicos y bandera de fin de semana.
        - 🌡 *Clima por municipio*: Temperaturas (máx/mín/media), amplitud térmica, precipitación y viento obtenidos en tiempo real desde Open-Meteo.
        - 🚌 *Servicio*: Línea, municipio y empresa predominante según el histórico SUBE.
        - 📆 *Contexto*: Feriados/flags y lags (lag_1, lag_7, lag_28) si existen registros recientes en la serie.
        
        *Entradas que no necesitás cargar manualmente:*
        - ❌ Provincia, tipo/nombre de feriado o tipo de transporte (el pipeline las gestiona o descarta).
        - ❌ Clima: se consulta automáticamente por municipio utilizando su centroide geográfico.
        
        *Lags*: cuando hay histórico disponible se generan lags por línea+municipio; si están ausentes se imputan durante el preprocesamiento.
        """)

    with st.expander("⚙ Detalles técnicos"):
        st.markdown("""
        ### Pipeline de procesamiento
        
        1. *Ingesta dinámica*:
           - Obtención de clima diario (Open-Meteo) por municipio usando coordenadas del georef oficial.
           - Construcción automática de registros para todos los municipios asociados a la línea elegida.
           - Validación temporal: solo se aceptan fechas hasta 16 días después del último dato del CSV histórico.
        
        2. *Feature Engineering* (pipeline serializado en `fe_pipeline.joblib`):
           - Ordenamiento temporal, extracción de calendarios, funciones cíclicas y banderas de fin de semana.
           - Cálculo de temperatura media/amplitud, creación de lags (lag_1, lag_7, lag_28) por línea/municipio y marcadores de disponibilidad.
           - Codificadores personalizados: perfiles históricos, sensibilidad al clima y patrones estacionales.
        
        3. *Preprocesamiento final* (`preprocessor.joblib`):
           - Drop de columnas auxiliares, winsorización climática, imputaciones numéricas/categóricas, MinMaxScaler y OneHotEncoder.
        
        4. *Modelo*:
           - {MODEL_NAME} entrenado en el notebook IGNA_Entrega3, reutilizado en esta app para inferir un valor por municipio y sumar el total.
        """)

