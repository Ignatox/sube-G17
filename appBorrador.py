import altair as alt
import pandas as pd
import streamlit as st


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


def show_weekday_participation_over_time(df):
    """Participación por día de la semana en el tiempo (componentes experimentales)."""
    required_cols = {"fecha", "municipio", "cantidad", "dia_semana"}
    if df.empty or not required_cols.issubset(df.columns):
        st.warning("No hay datos suficientes para la visualización de participación.")
        return

    data = df[list(required_cols)].dropna(subset=["fecha", "municipio"]).copy()
    if data.empty:
        st.warning("No hay datos suficientes para la visualización de participación.")
        return

    if not pd.api.types.is_datetime64_any_dtype(data["fecha"]):
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
    data = data.dropna(subset=["fecha"])

    dia_nombres = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    data["dia_nombre"] = data["dia_semana"].map(
        lambda x: dia_nombres[int(x)] if pd.notna(x) else None
    )

    municipios = sorted(data["municipio"].dropna().unique().tolist())
    if not municipios:
        st.warning("No se encontraron municipios para esta visualización.")
        return

    min_date = data["fecha"].min().date()
    max_date = data["fecha"].max().date()

    col_left, col_right = st.columns(2)
    with col_left:
        selected_muni = st.selectbox(
            "Municipio (participación)", municipios, key="part_muni"
        )
    with col_right:
        start_date, end_date = st.date_input(
            "Rango de fechas (participación)",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="part_rango",
        )

    if isinstance(start_date, tuple):
        start_date, end_date = start_date

    filtered = data[
        (data["municipio"] == selected_muni)
        & (data["fecha"].dt.date >= start_date)
        & (data["fecha"].dt.date <= end_date)
    ].copy()

    if filtered.empty:
        st.warning("No hay datos para el rango seleccionado (participación).")
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

    st.altair_chart(stack_chart, use_container_width=True)

