import io
from pathlib import Path

import altair as alt
import datetime as dt
import joblib
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import date as date_cls
from datetime import datetime

WEEKDAY_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

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


class DropNaRows(BaseEstimator, TransformerMixin):
    """Elimina filas con NaN en columnas específicas, usado tras generar lags."""

    def __init__(
        self,
        columns=None,
        how="any",
        skip_if_target_nan: bool = True,
        target_col: str = "cantidad",
    ):
        self.columns = columns or []
        self.how = how
        self.skip_if_target_nan = skip_if_target_nan
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.skip_if_target_nan = True
        if "target_col" not in self.__dict__:
            self.target_col = "cantidad"
        if "columns" not in self.__dict__:
            self.columns = []
        if "how" not in self.__dict__:
            self.how = "any"

    def transform(self, X):
        X_new = X.copy()
        skip_if_target_nan = getattr(self, "skip_if_target_nan", True)
        target_col = getattr(self, "target_col", "cantidad")
        if not self.columns:
            return X_new.reset_index(drop=True)

        subset = [col for col in self.columns if col in X_new.columns]
        if not subset:
            return X_new.reset_index(drop=True)

        mask = X_new[subset].isna()
        if skip_if_target_nan and target_col in X_new.columns:
            future_rows = X_new[target_col].isna()
            mask.loc[future_rows, :] = False

        if self.how == "all":
            drop_mask = mask.all(axis=1)
        else:
            drop_mask = mask.any(axis=1)

        X_new = X_new.loc[~drop_mask]
        return X_new.reset_index(drop=True)


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
DATA_DIR = Path("artifacts")
PREDICTION_REFERENCE_PATH = DATA_DIR / "prediction_reference.csv"
SUBE_DATA_URL_TEMPLATE = "https://archivos-datos.transporte.gob.ar/upload/Dat_Ab_Usos/dat-ab-usos-{year}.csv"

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


def _download_remote_usage_data(year: int) -> pd.DataFrame:
    """Descarga el CSV anual de usos SUBE desde datos.transporte."""
    url = SUBE_DATA_URL_TEMPLATE.format(year=year)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content))


def _prepare_prediction_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra AMBA y deja solo las columnas necesarias para predicción."""
    if df.empty:
        return df

    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    column_map = {
        "DIA_TRANSPORTE": "fecha",
        "NOMBRE_EMPRESA": "empresa",
        "LINEA": "linea",
        "AMBA": "amba",
        "TIPO_TRANSPORTE": "tipo_transporte",
        "JURISDICCION": "jurisdiccion",
        "PROVINCIA": "provincia",
        "MUNICIPIO": "municipio",
        "CANTIDAD": "cantidad",
        "DATO_PRELIMINAR": "dato_preliminar",
    }
    df = df.rename(columns=column_map)

    if "amba" in df.columns:
        df = df[df["amba"].astype(str).str.upper() == "SI"]

    keep_cols = [
        "fecha",
        "empresa",
        "linea",
        "jurisdiccion",
        "provincia",
        "municipio",
        "tipo_transporte",
        "cantidad",
        "dato_preliminar",
    ]
    df = df[[col for col in keep_cols if col in df.columns]].copy()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df = df.dropna(subset=["fecha", "linea", "municipio", "cantidad"])
    df["linea"] = df["linea"].astype(str).str.strip()
    df["municipio"] = df["municipio"].astype(str).str.strip()
    df["empresa"] = df["empresa"].astype(str).str.strip()
    df["provincia"] = df["provincia"].astype(str).str.strip()
    df["cantidad"] = df["cantidad"].astype(int)

    df = df.sort_values("fecha")
    return df


@st.cache_data(show_spinner=True)
def load_prediction_reference(refresh_key: str):
    """Descarga (o reutiliza) el dataset de usos SUBE para predicción."""
    _ = refresh_key  # fuerza la invalidación diaria del caché
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current_year = pd.Timestamp.today().year
    errors = []

    for year in (current_year, current_year - 1):
        try:
            remote_df = _download_remote_usage_data(year)
            prepared = _prepare_prediction_reference(remote_df)
            if not prepared.empty:
                prepared.to_csv(PREDICTION_REFERENCE_PATH, index=False)
                return prepared
        except Exception as exc:
            errors.append(str(exc))

    # Fallback to local copy if network fails
    if PREDICTION_REFERENCE_PATH.exists():
        cached = pd.read_csv(PREDICTION_REFERENCE_PATH, parse_dates=["fecha"])
        return cached

    raise RuntimeError(
        "No se pudo descargar el dataset actualizado. Errores: " + " | ".join(errors)
    )


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
    """
    Explorador interactivo (Gráfico 1) con filtros previos de Streamlit.
    """
    if df.empty:
        return None

    required_cols = {"fecha", "municipio", "linea", "cantidad"}
    if not required_cols.issubset(df.columns):
        return None

    data = df[list(required_cols)].dropna(subset=["fecha", "municipio", "linea"]).copy()
    if not pd.api.types.is_datetime64_any_dtype(data["fecha"]):
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
    data = data.dropna(subset=["fecha"])

    data_2024 = data[data["fecha"].dt.year == 2024].copy()
    if data_2024.empty:
        st.info("No se encontraron datos del 2024; se usará el último año disponible.")
        last_year = data["fecha"].dt.year.max()
        if pd.isna(last_year):
            return None
        data_2024 = data[data["fecha"].dt.year == last_year].copy()
        if data_2024.empty:
            return None

    data_2024["fecha_mes"] = data_2024["fecha"].dt.to_period("M").dt.to_timestamp()

    muni_month = (
        data_2024.groupby(["municipio", "fecha_mes"], as_index=False)["cantidad"]
        .sum()
        .rename(columns={"fecha_mes": "fecha"})
    )
    line_month = (
        data_2024.groupby(["municipio", "linea", "fecha_mes"], as_index=False)["cantidad"]
        .sum()
        .rename(columns={"fecha_mes": "fecha"})
    )

    month_names = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    muni_month["mes_nombre"] = muni_month["fecha"].dt.month.map(
        lambda m: month_names[int(m) - 1] if pd.notna(m) else None
    )
    line_month["mes_nombre"] = line_month["fecha"].dt.month.map(
        lambda m: month_names[int(m) - 1] if pd.notna(m) else None
    )

    municipios = sorted(muni_month["municipio"].dropna().unique().tolist())
    if not municipios:
        return None

    col_left, col_right = st.columns(2)
    with col_left:
        selected_municipio = st.selectbox(
            "Municipio", municipios, index=0, key="explorer_municipio"
        )

    available_months = muni_month.loc[
        muni_month["municipio"] == selected_municipio, "mes_nombre"
    ].dropna().unique().tolist()
    ordered_months = [m for m in month_names if m in available_months]
    month_options = ["Todos"] + ordered_months
    with col_right:
        selected_month = st.selectbox(
            "Mes (opcional)", month_options, index=0, key="explorer_mes"
        )

    muni_filtered = muni_month[muni_month["municipio"] == selected_municipio].copy()
    line_filtered = line_month[line_month["municipio"] == selected_municipio].copy()
    if selected_month != "Todos":
        muni_filtered = muni_filtered[muni_filtered["mes_nombre"] == selected_month]
        line_filtered = line_filtered[line_filtered["mes_nombre"] == selected_month]

    if muni_filtered.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return None

    date_selection = alt.selection_point(
        fields=["fecha"],
        on="click",
        nearest=True,
        empty="none",
        clear="dblclick"
    )

    base = alt.Chart(muni_filtered)
    line_chart = base.mark_line(
        color="#1f77b4",
        strokeWidth=3,
        point=False
    ).encode(
        x=alt.X(
            "fecha:T",
            title="Mes (puntos = 1.º día)",
            axis=alt.Axis(format="%b", tickCount="month")
        ),
        y=alt.Y("cantidad:Q", title="Total pasajeros", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("fecha:T", title="Mes", format="%Y-%m-%d"),
            alt.Tooltip("cantidad:Q", title="Pasajeros", format=",.0f"),
        ],
    )

    point_chart = base.mark_point(
        size=140,
        filled=True,
        stroke="#0d3b66",
        strokeWidth=1.5
    ).encode(
        x="fecha:T",
        y="cantidad:Q",
        tooltip=[
            alt.Tooltip("fecha:T", title="Mes", format="%B %Y"),
            alt.Tooltip("cantidad:Q", title="Pasajeros", format=",.0f"),
        ],
        color=alt.condition(
            date_selection,
            alt.value("#d62728"),
            alt.value("#1f77b4"),
        ),
    ).add_params(date_selection)

    timeline = (
        (line_chart + point_chart)
        .properties(
            title=f"📊 Demanda mensual - {selected_municipio}",
            width=900,
            height=320,
        )
    )

    detail_base = alt.Chart(line_filtered)
    bars = detail_base.transform_filter(date_selection).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        y=alt.Y("linea:N", title="Línea", sort="-x"),
        x=alt.X("cantidad:Q", title="Pasajeros"),
        color=alt.Color("linea:N", title="Línea", legend=None),
        tooltip=[
            alt.Tooltip("linea:N", title="Línea"),
            alt.Tooltip("cantidad:Q", title="Pasajeros", format=",.0f"),
        ],
    )

    labels = detail_base.transform_filter(date_selection).mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="#222"
    ).encode(
        y="linea:N",
        x="cantidad:Q",
        text=alt.Text("cantidad:Q", format=",.0f"),
    )

    instruction_layer = (
        alt.Chart(pd.DataFrame({
            "mensaje": ["Seleccioná un punto del gráfico superior para ver el desglose."]
        }))
        .mark_text(
            align="center",
            baseline="middle",
            fontSize=15,
            color="#6c757d"
        )
        .encode(text="mensaje:N")
        .transform_filter(~date_selection)
    )

    detail_chart = (
        alt.layer(bars, labels, instruction_layer)
        .properties(
            title="🔍 Desglose por línea del punto seleccionado",
            width=900,
            height=320,
        )
    )

    return alt.vconcat(timeline, detail_chart).resolve_scale(color="independent")


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


def show_municipality_comparator(df):
    """Comparador entre dos municipios con métricas clave."""
    required_cols = {"fecha", "municipio", "cantidad"}
    if df.empty or not required_cols.issubset(df.columns):
        st.warning("No hay datos suficientes para el comparador de municipios.")
        return

    data = df[list(required_cols)].dropna(subset=["fecha", "municipio"]).copy()
    if data.empty:
        st.warning("No hay datos suficientes para el comparador de municipios.")
        return

    if not pd.api.types.is_datetime64_any_dtype(data["fecha"]):
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
    data = data.dropna(subset=["fecha"])

    top_munis = (
        data.groupby("municipio")["cantidad"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    if len(top_munis) < 2:
        st.warning("Se necesitan al menos dos municipios para comparar.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        muni_a = st.selectbox(
            "Municipio A",
            options=top_munis,
            index=0,
            key="comp_muni_a",
            help="Municipio base para la comparación",
        )
    with col_b:
        default_b = 1 if len(top_munis) > 1 else 0
        muni_b = st.selectbox(
            "Municipio B",
            options=top_munis,
            index=default_b,
            key="comp_muni_b",
            help="Municipio contra el cual comparar",
        )

    if muni_a == muni_b:
        st.info("Seleccioná dos municipios distintos para ver la comparación.")
        return

    compare_data = data[data["municipio"].isin([muni_a, muni_b])].copy()
    daily = (
        compare_data.groupby(["fecha", "municipio"])["cantidad"]
        .sum()
        .reset_index()
    )

    pivot = (
        daily.pivot(index="fecha", columns="municipio", values="cantidad")
        .fillna(0)
        .sort_index()
    )
    pivot["diferencia"] = pivot[muni_a] - pivot[muni_b]
    pivot = pivot.reset_index()

    timeline_chart = (
        alt.Chart(daily)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("cantidad:Q", title="Pasajeros"),
            color=alt.Color(
                "municipio:N",
                title="Municipio",
                scale=alt.Scale(
                    domain=[muni_a, muni_b],
                    range=["#1f77b4", "#ff7f0e"],
                ),
            ),
            tooltip=[
                alt.Tooltip("fecha:T", title="Fecha", format="%Y-%m-%d"),
                alt.Tooltip("municipio:N", title="Municipio"),
                alt.Tooltip("cantidad:Q", title="Pasajeros", format=",.0f"),
            ],
        )
        .properties(width=900, height=320, title="📈 Evolución diaria comparada")
    )

    diff_chart = (
        alt.Chart(pivot)
        .mark_bar()
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("diferencia:Q", title=f"Δ Pasajeros ({muni_a} - {muni_b})"),
            color=alt.condition(
                alt.datum.diferencia >= 0,
                alt.value("#2b8a3e"),
                alt.value("#c70039"),
            ),
            tooltip=[
                alt.Tooltip("fecha:T", title="Fecha", format="%Y-%m-%d"),
                alt.Tooltip("diferencia:Q", title="Diferencia", format=",.0f"),
            ],
        )
        .properties(width=900, height=200, title="↕️ Diferencia diaria de pasajeros")
    )

    stats = (
        compare_data.groupby("municipio")["cantidad"]
        .agg(total="sum", promedio="mean", maximo="max")
        .loc[[muni_a, muni_b]]
    )
    stats["promedio"] = stats["promedio"].round(0)
    stats["maximo"] = stats["maximo"].round(0)

    diff_total = stats.loc[muni_a, "total"] - stats.loc[muni_b, "total"]
    diff_avg = stats.loc[muni_a, "promedio"] - stats.loc[muni_b, "promedio"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            f"Total {muni_a}",
            f"{stats.loc[muni_a, 'total']:,.0f}",
            delta=f"{diff_total:,.0f}",
            delta_color="normal",
        )
    with col2:
        st.metric(
            f"Total {muni_b}",
            f"{stats.loc[muni_b, 'total']:,.0f}",
            delta=f"{-diff_total:,.0f}",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            f"Promedio diario {muni_a}",
            f"{stats.loc[muni_a, 'promedio']:,.0f}",
            delta=f"{diff_avg:,.0f}",
            delta_color="normal",
        )
    with col4:
        st.metric(
            f"Pico diario {muni_a}",
            f"{stats.loc[muni_a, 'maximo']:,.0f}",
            help="Mayor cantidad diaria registrada",
        )

    st.altair_chart(timeline_chart, use_container_width=True)
    st.altair_chart(diff_chart, use_container_width=True)


def show_contribution_insights(df):
    """Análisis de contribución por día/mes para un municipio dado."""
    required_cols = {"fecha", "municipio", "cantidad", "dia_semana", "mes"}
    if df.empty or not required_cols.issubset(df.columns):
        st.warning("No hay columnas suficientes para el análisis de contribución.")
        return

    data = df[list(required_cols)].dropna(subset=["fecha", "municipio"]).copy()
    if data.empty:
        st.warning("No hay datos suficientes para el análisis de contribución.")
        return

    if not pd.api.types.is_datetime64_any_dtype(data["fecha"]):
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
    data = data.dropna(subset=["fecha"])

    dia_nombres = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    mes_nombres = [
        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
    ]
    data["dia_nombre"] = data["dia_semana"].map(
        lambda x: dia_nombres[int(x)] if pd.notna(x) else None
    )
    data["mes_nombre"] = data["mes"].map(
        lambda x: mes_nombres[int(x) - 1] if pd.notna(x) else None
    )

    municipios = sorted(data["municipio"].dropna().unique().tolist())
    if not municipios:
        st.warning("No se encontraron municipios para el análisis de contribución.")
        return

    min_date = data["fecha"].min().date()
    max_date = data["fecha"].max().date()

    col_left, col_right = st.columns(2)
    with col_left:
        selected_muni = st.selectbox(
            "Municipio (contribución)", municipios, key="contrib_muni"
        )
    with col_right:
        start_date, end_date = st.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="contrib_rango",
        )

    if isinstance(start_date, tuple):
        start_date, end_date = start_date

    filtered = data[
        (data["municipio"] == selected_muni)
        & (data["fecha"].dt.date >= start_date)
        & (data["fecha"].dt.date <= end_date)
    ].copy()

    if filtered.empty:
        st.warning("No hay datos para el rango seleccionado.")
        return

    timeline_stack = (
        filtered.groupby(["fecha", "dia_nombre"])["cantidad"]
        .sum()
        .reset_index()
    )

    stack_chart = (
        alt.Chart(timeline_stack)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("cantidad:Q", stack="normalize", title="Participación"),
            color=alt.Color(
                "dia_nombre:N",
                title="Día de la semana",
                scale=alt.Scale(scheme="category10"),
            ),
            tooltip=[
                alt.Tooltip("fecha:T", title="Fecha", format="%Y-%m-%d"),
                alt.Tooltip("dia_nombre:N", title="Día"),
                alt.Tooltip("cantidad:Q", title="Pasajeros", format=",.0f"),
            ],
        )
        .properties(
            width=900,
            height=250,
            title="🧩 Participación por día de la semana en el tiempo",
        )
    )

    heatmap_data = (
        filtered.groupby(["mes_nombre", "dia_nombre"])["cantidad"]
        .mean()
        .reset_index()
    )

    heatmap_chart = (
        alt.Chart(heatmap_data)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X(
                "dia_nombre:N",
                title="Día",
                sort=dia_nombres,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "mes_nombre:N",
                title="Mes",
                sort=mes_nombres,
            ),
            color=alt.Color(
                "cantidad:Q",
                title="Promedio de pasajeros",
                scale=alt.Scale(scheme="yelloworangered"),
            ),
            tooltip=[
                alt.Tooltip("mes_nombre:N", title="Mes"),
                alt.Tooltip("dia_nombre:N", title="Día"),
                alt.Tooltip("cantidad:Q", title="Pasajeros", format=",.0f"),
            ],
        )
        .properties(
            width=400,
            height=300,
            title="🔥 Intensidad promedio por día y mes",
        )
    )

    summary = (
        filtered.groupby("dia_nombre")["cantidad"]
        .agg(
            total="sum",
            promedio="mean",
            maximo="max",
        )
        .reindex(dia_nombres)
        .dropna()
    )
    summary["participacion"] = (
        summary["total"] / summary["total"].sum() * 100
    ).round(1)
    summary = summary.round({"promedio": 0, "maximo": 0})

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.altair_chart(stack_chart, use_container_width=True)
    with col_b:
        st.altair_chart(heatmap_chart, use_container_width=True)

    st.markdown(f"### 📋 Resumen por día - {selected_muni}")
    st.dataframe(
        summary.rename(
            columns={
                "total": "Total",
                "promedio": "Promedio",
                "maximo": "Máximo",
                "participacion": "% Participación",
            }
        ),
        use_container_width=True,
    )


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
    """Dashboard completo con filtros previos de Streamlit."""
    if df.empty:
        return None

    if not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        df = df.copy()
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"])

    num_days = df["fecha"].nunique()
    if num_days > 1000:
        df["fecha_mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
        muni_daily = (
            df.groupby(["municipio", "fecha_mes"])["cantidad"]
            .sum()
            .reset_index()
            .rename(columns={"fecha_mes": "fecha"})
        )
    else:
        muni_daily = (
            df.groupby(["municipio", "fecha"])["cantidad"].sum().reset_index()
        )

    muni_stats = (
        df.groupby("municipio")["cantidad"]
        .agg(["mean", "sum"])
        .reset_index()
        .sort_values("sum", ascending=False)
        .head(15)
    )
    municipios = sorted(muni_stats["municipio"].tolist())
    if not municipios:
        return None

    selector_options = ["Todos"] + municipios
    selected_municipio = st.selectbox(
        "Municipio (dashboard)",
        selector_options,
        index=0,
        key="dashboard_municipio",
    )

    if selected_municipio == "Todos":
        muni_daily_filtered = muni_daily.copy()
        muni_stats_filtered = muni_stats.copy()
    else:
        muni_daily_filtered = muni_daily[
            muni_daily["municipio"] == selected_municipio
        ].copy()
        muni_stats_filtered = muni_stats[
            muni_stats["municipio"] == selected_municipio
        ].copy()

    if muni_daily_filtered.empty:
        st.warning("No hay datos para el municipio seleccionado en el dashboard.")
        return None

    brush = alt.selection_interval(encodings=["x"])

    base = alt.Chart(muni_daily_filtered)

    timeline = base.mark_area(
        interpolate="monotone",
        opacity=0.7,
    ).encode(
        x=alt.X("fecha:T", title="Fecha"),
        y=alt.Y("cantidad:Q", title="Pasajeros", aggregate="sum"),
        color=alt.value("#4a90e2"),
        tooltip=[
            alt.Tooltip("fecha:T", format="%Y-%m-%d"),
            alt.Tooltip("cantidad:Q", format=",.0f", aggregate="sum", title="Total Pasajeros"),
        ],
    ).add_params(brush).properties(
        title="📅 Evolución Temporal",
        width=600,
        height=200,
    )

    top_title = (
        "🏆 Top 15 Municipios"
        if selected_municipio == "Todos"
        else f"🏙️ Totales para {selected_municipio}"
    )
    top_chart = alt.Chart(muni_stats_filtered).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
    ).encode(
        x=alt.X("municipio:N", title="Municipio", sort="-y"),
        y=alt.Y("sum:Q", title="Total Pasajeros"),
        color=alt.Color("sum:Q", scale=alt.Scale(scheme="viridis"), legend=None),
        tooltip=[
            alt.Tooltip("municipio:N", title="Municipio"),
            alt.Tooltip("sum:Q", format=",.0f", title="Total Pasajeros"),
            alt.Tooltip("mean:Q", format=",.0f", title="Promedio Diario"),
        ],
    ).properties(
        title=top_title,
        width=600,
        height=200,
    )

    distribution = base.mark_bar(
        binSpacing=2,
        opacity=0.8,
    ).encode(
        x=alt.X("cantidad:Q", title="Pasajeros", bin=alt.Bin(maxbins=30)),
        y=alt.Y("count()", title="Frecuencia"),
        color=alt.value("#e74c3c"),
        tooltip=[
            alt.Tooltip("cantidad:Q", bin=True, title="Rango Pasajeros"),
            alt.Tooltip("count()", title="Frecuencia"),
        ],
    ).transform_filter(brush).properties(
        title="📊 Distribución (filtrada por rango temporal)",
        width=600,
        height=200,
    )

    return alt.vconcat(timeline, top_chart, distribution)


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
    results_df['dia_semana'] = results_df['fecha'].dt.dayofweek.map(
        lambda idx: WEEKDAY_NAMES[idx] if pd.notna(idx) and int(idx) < len(WEEKDAY_NAMES) else ""
    )
    
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
            alt.Tooltip('dia_semana:N', title='Día'),
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
            alt.Tooltip('dia_semana:N', title='Día'),
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
            alt.Tooltip('dia_semana:N', title='Día'),
            alt.Tooltip('prediccion:Q', format=',.0f', aggregate='sum', title='Pasajeros')
        ]
    ).transform_filter(click).properties(
        title='🎯 Desglose por Municipio (click en una fecha)',
        width=800,
        height=250
    )

    # Panel 4: Condiciones climáticas promedio para la fecha seleccionada
    climate_chart = (
        alt.Chart(results_df)
        .transform_fold(
            ["tmax", "tmin", "precip", "viento"],
            as_=["variable", "valor"],
        )
        .transform_filter(click)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "variable:N",
                title="Variable",
                sort=["tmax", "tmin", "precip", "viento"],
            ),
            y=alt.Y(
                "valor:Q",
                aggregate="mean",
                title="Valor medio (fecha seleccionada)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "variable:N",
                scale=alt.Scale(
                    domain=["tmax", "tmin", "precip", "viento"],
                    range=["#d62728", "#1f77b4", "#9467bd", "#2ca02c"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("fecha:T", format="%Y-%m-%d", title="Fecha"),
                alt.Tooltip("dia_semana:N", title="Día"),
                alt.Tooltip("variable:N", title="Variable"),
                alt.Tooltip("mean(valor):Q", format=",.2f", title="Valor"),
            ],
        )
        .properties(
            title="🌡️ Condiciones climáticas del día seleccionado",
            width=800,
            height=220,
        )
    )
    
    return alt.vconcat(total_chart, lines_chart, detail_chart, climate_chart).resolve_scale(color='independent')


def create_prediction_total_chart(results_df):
    """Línea de tiempo agregada para el total de pasajeros predichos."""
    if results_df.empty:
        return None

    totals_df = (
        results_df.assign(fecha=pd.to_datetime(results_df["fecha"]))
        .groupby("fecha", as_index=False)["prediccion"]
        .sum()
    )
    totals_df["dia_semana"] = totals_df["fecha"].dt.dayofweek.map(
        lambda idx: WEEKDAY_NAMES[idx] if pd.notna(idx) and int(idx) < len(WEEKDAY_NAMES) else ""
    )

    return (
        alt.Chart(totals_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("fecha:T", title="Fecha", axis=alt.Axis(format="%Y-%m-%d")),
            y=alt.Y("prediccion:Q", title="Pasajeros estimados", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("fecha:T", format="%Y-%m-%d", title="Fecha"),
                alt.Tooltip("dia_semana:N", title="Día"),
                alt.Tooltip("prediccion:Q", format=",.0f", title="Total pasajeros"),
            ],
        )
        .properties(title="🧮 Total combinado de pasajeros", width=800, height=320)
    )


def create_prediction_total_with_climate(results_df):
    """Vista simplificada con total diario y clima asociado."""
    if results_df.empty:
        return None

    df = results_df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    agg = (
        df.groupby("fecha", as_index=False)
        .agg(
            prediccion=("prediccion", "sum"),
            tmax=("tmax", "mean"),
            tmin=("tmin", "mean"),
            precip=("precip", "mean"),
            viento=("viento", "mean"),
        )
        .fillna(np.nan)
    )
    agg["dia_semana"] = agg["fecha"].dt.dayofweek.map(
        lambda idx: WEEKDAY_NAMES[idx] if pd.notna(idx) and int(idx) < len(WEEKDAY_NAMES) else ""
    )

    click = alt.selection_point(fields=["fecha"], empty=True)

    total_chart = (
        alt.Chart(agg)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("fecha:T", title="Fecha", axis=alt.Axis(format="%Y-%m-%d")),
            y=alt.Y("prediccion:Q", title="Pasajeros estimados", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("fecha:T", format="%Y-%m-%d", title="Fecha"),
                alt.Tooltip("dia_semana:N", title="Día"),
                alt.Tooltip("prediccion:Q", format=",.0f", title="Total pasajeros"),
                alt.Tooltip("tmax:Q", format=",.1f", title="T° Máx"),
                alt.Tooltip("tmin:Q", format=",.1f", title="T° Mín"),
                alt.Tooltip("precip:Q", format=",.1f", title="Precip (mm)"),
                alt.Tooltip("viento:Q", format=",.1f", title="Viento (km/h)"),
            ],
        )
        .add_params(click)
        .properties(width=800, height=320, title="🧮 Total diario de pasajeros")
    )

    climate_chart = (
        alt.Chart(agg)
        .transform_fold(
            ["tmax", "tmin", "precip", "viento"],
            as_=["variable", "valor"],
        )
        .transform_filter(click)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("variable:N", title="Variable", sort=["tmax", "tmin", "precip", "viento"]),
            y=alt.Y(
                "valor:Q",
                aggregate="mean",
                title="Valor (fecha seleccionada)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "variable:N",
                scale=alt.Scale(
                    domain=["tmax", "tmin", "precip", "viento"],
                    range=["#d62728", "#1f77b4", "#9467bd", "#2ca02c"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("fecha:T", format="%Y-%m-%d", title="Fecha"),
                alt.Tooltip("dia_semana:N", title="Día"),
                alt.Tooltip("variable:N", title="Variable"),
                alt.Tooltip("mean(valor):Q", format=",.2f", title="Valor"),
            ],
        )
        .properties(
            width=800,
            height=220,
            title="🌡️ Condiciones climáticas del día seleccionado",
        )
    )

    return alt.vconcat(total_chart, climate_chart)


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
prediction_refresh_key = pd.Timestamp.today().strftime("%Y-%m-%d")
try:
    df_prediction_ref = load_prediction_reference(prediction_refresh_key)
except Exception as prediction_error:
    st.warning(
        f"No se pudieron descargar los datos actualizados de SUBE ({prediction_error}). "
        "Se continuará utilizando únicamente el dataset local."
    )
    df_prediction_ref = pd.DataFrame()

df_prediction_source = df_prediction_ref if not df_prediction_ref.empty else df_full
if not df_full.empty and not df_prediction_source.empty:
    valid_lines = set(df_full["linea"].astype(str).unique())
    original_count = len(df_prediction_source)
    df_prediction_source = df_prediction_source[
        df_prediction_source["linea"].astype(str).isin(valid_lines)
    ].copy()
    filtered_count = len(df_prediction_source)
    if filtered_count < original_count:
        st.info(
            "Algunas líneas recientes no están presentes en el dataset base 2024, "
            "por lo que fueron excluidas de las predicciones."
        )

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
            st.altair_chart(chart_interactive, use_container_width=True)
        
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
            st.altair_chart(chart_heatmap, use_container_width=True)
        
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
            st.altair_chart(chart_weather, use_container_width=True)
        
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
            st.altair_chart(chart_lines, use_container_width=True)
        
        st.divider()
        
        # Visualización 5: Evolución temporal mejorada
        st.subheader("📅 Evolución Temporal Mejorada")
        st.markdown("""
        **Arrastra para seleccionar un rango temporal y click en puntos para destacarlos**
        """)
        chart_temporal = create_temporal_distribution_chart(df_full)
        if chart_temporal:
            st.altair_chart(chart_temporal, use_container_width=True)
        
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
            st.altair_chart(chart_dashboard, use_container_width=True)
        
        st.divider()

        st.subheader("⚖️ Comparador de Municipios")
        st.markdown("""
        **Elegí dos municipios para analizar sus trayectorias y detectar ventajas competitivas.**
        
        El panel muestra:
        - Evolución diaria superpuesta.
        - Diferencia absoluta día a día.
        - KPIs agregados con deltas instantáneos.
        """)
        show_municipality_comparator(df_full)

        st.divider()

        st.subheader("🧩 Contribución por Día y Mes")
        st.markdown("""
        **Explorá cómo se reparte la demanda dentro de un municipio según día de la semana y mes.**
        
        Filtrá un rango temporal para:
        - Ver la participación relativa de cada día.
        - Identificar meses/días con mayor intensidad.
        - Revisar métricas consolidadas por día.
        """)
        show_contribution_insights(df_full)

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
            
            st.dataframe(
                df_full[['cantidad', 't_med', 'precip', 'viento']].describe(),
                use_container_width=True,
            )

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

    line_opts = get_line_options(df_prediction_source)
    prediction_last_date = (
        pd.to_datetime(df_prediction_source["fecha"].max())
        if not df_prediction_source.empty
        else None
    )
    if prediction_last_date is not None:
        st.caption(
            f"Último dato SUBE (AMBA): {prediction_last_date.date()} • "
            "La predicción permite hasta 16 días posteriores."
        )

    total_placeholder = None
    with st.form("prediction_form"):
        col_left, col_right = st.columns([3, 1])

        with col_left:
            st.markdown("### 📍 Selección")
            # Fechas con validación contra último dato del CSV (+16 días)
            last_date = prediction_last_date if prediction_last_date is not None else pd.Timestamp.today()
            max_allowed = (last_date + pd.Timedelta(days=16)).date()
            default_start = min((last_date + pd.Timedelta(days=1)).date(), max_allowed) if not df_prediction_source.empty else pd.Timestamp.today().date()
            fecha_desde = st.date_input(
                "Fecha desde",
                value=default_start,
                max_value=max_allowed,
                key="prediction_fecha_desde",
            )

            fecha_hasta_key = "prediction_fecha_hasta"
            stored_hasta = st.session_state.get(fecha_hasta_key)
            if isinstance(stored_hasta, dt.date):
                if stored_hasta < fecha_desde:
                    st.session_state[fecha_hasta_key] = fecha_desde
                elif stored_hasta > max_allowed:
                    st.session_state[fecha_hasta_key] = max_allowed
            else:
                st.session_state.pop(fecha_hasta_key, None)

            fecha_hasta = st.date_input(
                "Predecir hasta",
                value=default_start,
                min_value=fecha_desde,
                max_value=max_allowed,
                key=fecha_hasta_key,
            )
            linea = st.selectbox("Línea", options=line_opts, index=0)

        with col_right:
            st.markdown("### 🔢 Predicción total")
            total_placeholder = st.empty()
            total_placeholder.info("La suma total del rango aparecerá aquí luego de predecir.")

        submitted = st.form_submit_button("🔮 Predecir", use_container_width=False)

    if submitted:
        # Validación de fechas
        last_date = prediction_last_date if prediction_last_date is not None else pd.Timestamp.today()
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
        municipios = get_municipios_for_line(df_prediction_source, linea)
        if not municipios:
            st.warning("No se encontraron municipios para la línea seleccionada en el CSV.")
            st.stop()

        # Precalcular info por municipio
        muni_info = []
        for muni in municipios:
            prov = get_default_provincia_for_line_muni(df_prediction_source, linea, muni)
            emp = get_default_empresa_for_line_muni(df_prediction_source, linea, muni)
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

            display_results = results.copy()
            display_results["fecha"] = pd.to_datetime(display_results["fecha"])
            display_results = display_results.sort_values(["fecha", "prediccion"], ascending=[True, False])
            multiple_municipios = display_results["municipio"].nunique() > 1

            if multiple_municipios:
                st.subheader("🎯 Predicción por municipio")
                st.dataframe(
                    display_results.assign(
                        prediccion=lambda d: d["prediccion"].round(0).astype(int)
                    ),
                    use_container_width=True,
                )
            else:
                st.subheader("📈 Predicción por fecha")
                st.caption("Esta línea solo presenta un municipio, por lo que no se muestra el desglose.")
                fecha_summary = (
                    display_results.groupby("fecha", as_index=False)["prediccion"]
                    .sum()
                    .assign(prediccion=lambda d: d["prediccion"].round(0).astype(int))
                )
                st.dataframe(fecha_summary, use_container_width=True)
            
            chart_pred_total = create_prediction_total_chart(display_results)
            chart_total_climate = create_prediction_total_with_climate(display_results)

            if multiple_municipios:
                # Gráfico 1: Timeline interactivo con desglose
                st.subheader("📊 Predicciones Interactivas por Municipio")
                st.markdown("""
                **Click en la leyenda para filtrar municipios, click en una fecha para ver detalles**
                """)
                chart_pred_timeline = create_prediction_timeline_interactive(display_results)
                if chart_pred_timeline:
                    st.altair_chart(chart_pred_timeline, use_container_width=True)

                if chart_pred_total:
                    st.divider()
                    st.subheader("🧮 Total combinado por fecha")
                    st.altair_chart(chart_pred_total, use_container_width=True)
                
                st.divider()
                
                # Gráfico 2: Comparación con clima
                st.subheader("🌦️ Predicciones vs Condiciones Climáticas")
                st.markdown("""
                **Explora cómo las predicciones se relacionan con el clima. Click en puntos para destacarlos.**
                """)
                chart_pred_weather = create_prediction_weather_comparison(display_results)
                if chart_pred_weather:
                    st.altair_chart(chart_pred_weather, use_container_width=True)
                
                st.divider()
                
                # Gráfico 3: Heatmap de predicciones
                st.subheader("🔥 Heatmap de Predicciones")
                st.markdown("""
                **Vista general de todas las predicciones por municipio y fecha**
                """)
                chart_pred_heatmap = create_prediction_heatmap(display_results)
                if chart_pred_heatmap:
                    st.altair_chart(chart_pred_heatmap, use_container_width=True)
            else:
                if chart_total_climate:
                    st.subheader("🧮 Total de pasajeros y clima")
                    st.altair_chart(chart_total_climate, use_container_width=True)
                elif chart_pred_total:
                    st.subheader("🧮 Total de pasajeros estimados")
                    st.altair_chart(chart_pred_total, use_container_width=True)
                st.info("No se muestran gráficos con desglose porque la línea solo incluye un municipio.")

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
