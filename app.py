import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Frontera Eficiente de Markowitz", layout="wide")
st.title("Frontera Eficiente de Markowitz")
st.caption("Retornos anuales · portafolios de mínima varianza, máximo retorno y máximo Sharpe · Línea del Mercado de Capitales (CML)")

# ---------------------------------------------------------------- funciones
def estadisticos(R: pd.DataFrame):
    mu = R.mean().values                 # retorno esperado anual (media aritmética)
    cov = R.cov().values                 # covarianza muestral
    return mu, cov

def ret_vol(w, mu, cov):
    return float(w @ mu), float(np.sqrt(w @ cov @ w))

def optimizar(mu, cov, lo, hi, objetivo, rf=0.0, ret_obj=None):
    n = len(mu)
    w0 = np.full(n, 1.0 / n)
    bounds = [(lo, hi)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if objetivo == "minvar":
        f = lambda w: w @ cov @ w
    elif objetivo == "maxret":
        f = lambda w: -(w @ mu)
    elif objetivo == "sharpe":
        f = lambda w: -((w @ mu - rf) / np.sqrt(w @ cov @ w))
    elif objetivo == "target":
        f = lambda w: w @ cov @ w
        cons.append({"type": "eq", "fun": lambda w: w @ mu - ret_obj})
    res = minimize(f, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    w = np.where(np.abs(res.x) < 1e-8, 0, res.x)
    return w / w.sum()

def frontera(mu, cov, lo, hi, n_puntos):
    w_min = optimizar(mu, cov, lo, hi, "minvar")
    w_max = optimizar(mu, cov, lo, hi, "maxret")
    r_min, r_max = w_min @ mu, w_max @ mu
    pts = []
    for r in np.linspace(r_min, r_max, n_puntos):
        w = optimizar(mu, cov, lo, hi, "target", ret_obj=r)
        pts.append((*ret_vol(w, mu, cov)[::-1], w))   # (vol, ret, pesos)
    return w_min, w_max, pts

# ---------------------------------------------------------------- datos
st.sidebar.header("Datos")
archivo = st.sidebar.file_uploader(
    "CSV de retornos anuales (una columna por activo, una fila por año; decimales con punto, ej. 0.15 = 15%)",
    type=["csv"])
if archivo is not None:
    R = pd.read_csv(archivo)
    R = R.select_dtypes("number")
else:
    st.sidebar.info("Sin archivo: se usan datos de ejemplo ficticios.")
    rng = np.random.default_rng(1)
    R = pd.DataFrame(rng.multivariate_normal(
        [0.10, 0.07, 0.12, 0.05], 0.02 * np.eye(4) + 0.01, size=10),
        columns=["Activo A", "Activo B", "Activo C", "Activo D"])

st.sidebar.header("Parámetros")
rf = st.sidebar.number_input("Tasa libre de riesgo anual (%)", value=4.0, step=0.25) / 100
lo = st.sidebar.number_input("Peso mínimo por activo (%)", value=0.0, min_value=0.0, max_value=100.0, step=1.0) / 100
hi = st.sidebar.number_input("Peso máximo por activo (%)", value=100.0, min_value=0.0, max_value=100.0, step=1.0) / 100
n_puntos = st.sidebar.slider("Puntos de la frontera", 5, 50, 25)

if R.shape[1] < 2:
    st.error("Se necesitan al menos 2 activos."); st.stop()
if lo * R.shape[1] > 1 or hi * R.shape[1] < 1:
    st.error("Los topes de peso son incompatibles con la cantidad de activos (deben permitir que la suma sea 100%)."); st.stop()

with st.expander("Retornos anuales utilizados", expanded=False):
    st.dataframe(R.style.format("{:.2%}"))

# ---------------------------------------------------------------- cálculo
if True:
    mu, cov = estadisticos(R)
    activos = list(R.columns)
    w_min, w_max, pts = frontera(mu, cov, lo, hi, n_puntos)
    w_sh = optimizar(mu, cov, lo, hi, "sharpe", rf=rf)

    r_min, v_min = ret_vol(w_min, mu, cov)
    r_max, v_max = ret_vol(w_max, mu, cov)
    r_sh, v_sh = ret_vol(w_sh, mu, cov)
    sharpe = (r_sh - rf) / v_sh

    # ------------------------------------------------------------ gráfico
    vols = [p[0] for p in pts]; rets = [p[1] for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=rets, mode="lines", name="Frontera eficiente"))
    fig.add_trace(go.Scatter(x=np.sqrt(np.diag(cov)), y=mu, mode="markers+text", text=activos,
                             textposition="top center", name="Activos", marker=dict(size=9, symbol="diamond")))
    x_cml = np.linspace(0, max(vols + [v_sh]) * 1.2, 50)
    fig.add_trace(go.Scatter(x=x_cml, y=rf + sharpe * x_cml, mode="lines", name="CML",
                             line=dict(dash="dash")))
    for nombre, v, r, simb in [("Mínima varianza", v_min, r_min, "circle"),
                               ("Máximo retorno", v_max, r_max, "square"),
                               ("Máximo Sharpe", v_sh, r_sh, "star")]:
        fig.add_trace(go.Scatter(x=[v], y=[r], mode="markers", name=nombre,
                                 marker=dict(size=15, symbol=simb, line=dict(width=1, color="black"))))
    fig.add_trace(go.Scatter(x=[0], y=[rf], mode="markers", name="Activo libre de riesgo",
                             marker=dict(size=10, symbol="x")))
    fig.update_layout(xaxis_title="Riesgo (desvío estándar anual)", yaxis_title="Retorno esperado anual",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=600,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------ tabla de los 3 portafolios
    tabla = pd.DataFrame({
        "Mínima varianza": list(w_min) + [r_min, v_min, (r_min - rf) / v_min],
        "Máximo retorno": list(w_max) + [r_max, v_max, (r_max - rf) / v_max],
        "Máximo Sharpe": list(w_sh) + [r_sh, v_sh, sharpe],
    }, index=activos + ["Retorno esperado", "Riesgo (desvío)", "Ratio de Sharpe"])
    fmt = {c: "{:.2%}" for c in tabla.columns}
    st.subheader("Los tres portafolios")
    st.dataframe(tabla.style.format(lambda x: f"{x:.2%}").format(
        "{:.3f}", subset=pd.IndexSlice[["Ratio de Sharpe"], :]))

    st.subheader("Puntos de la frontera")
    tf = pd.DataFrame([[p[1], p[0], *p[2]] for p in pts],
                      columns=["Retorno", "Riesgo"] + activos)
    st.dataframe(tf.style.format("{:.2%}"))
    st.download_button("Descargar frontera (CSV)", tf.to_csv(index=False).encode("utf-8"),
                       "frontera.csv", "text/csv")

    st.caption("Supuestos: retornos anuales históricos, media aritmética y covarianza muestral; "
               "sin ventas en corto; la CML parte de la tasa libre de riesgo y es tangente a la frontera en el portafolio de máximo Sharpe.")
