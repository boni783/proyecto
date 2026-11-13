"""
Tablero: riesgo de impago de la primera cuota en créditos vehiculares.

Archivos necesarios en la misma carpeta (los genera el notebook, sección 7):
    predicciones_dash.csv, tasas_canal.csv, config_dash.json

Para correrlo en tu computador:
    pip install -r requirements.txt
    python app.py
y abre http://127.0.0.1:8050 en el navegador.

Uso de IA: este código se generó con apoyo de IA; declararlo en el reporte.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ------------------------------------------------------------------
# Colores (los mismos en todo el tablero)
# ------------------------------------------------------------------
ROJO, AZUL, VERDE, AMARILLO, GRIS = "#C44E52", "#4C72B0", "#55A868", "#E1A33B", "#8A8A8A"
FONDO, TARJETA, TEXTO = "#F4F6F9", "#FFFFFF", "#2B2B2B"

# ------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------
pred = pd.read_csv("predicciones_dash.csv", dtype={"estado": str, "sucursal": str, "fabricante": str})
canal = pd.read_csv("tasas_canal.csv", dtype={"codigo": str})
with open("config_dash.json", encoding="utf-8") as f:
    cfg = json.load(f)

TASA = pred["y_true"].mean()
TASA_GLOBAL = cfg.get("tasa_global", TASA)

# Clientes ordenados del más al menos riesgoso, para el punto de corte
orden = np.argsort(-pred["proba"].to_numpy())
y_ord = pred["y_true"].to_numpy()[orden]
N = len(y_ord)
morosos_acum = np.cumsum(y_ord) / y_ord.sum()
buenos_acum = np.cumsum(1 - y_ord) / (1 - y_ord).sum()


def miles(n):
    """46631 -> '46.631'"""
    return f"{n:,}".replace(",", ".")


def pct(v, dec=1):
    """0.182 -> '18,2 %'"""
    return f"{v * 100:.{dec}f}".replace(".", ",") + " %"


def resultado_corte(porcentaje):
    """Qué pasa si se rechaza al `porcentaje` % de solicitantes más riesgosos."""
    k = int(N * porcentaje / 100)
    return {
        "mora_con_modelo": y_ord[k:].mean(),
        "morosos_detectados": morosos_acum[k - 1] if k > 0 else 0.0,
        "buenos_rechazados": buenos_acum[k - 1] if k > 0 else 0.0,
    }


def estilo_figura(fig, titulo, alto=380):
    fig.update_layout(title={"text": titulo, "x": 0.02, "font": {"size": 16}},
                      template="simple_white", height=alto,
                      margin={"l": 60, "r": 30, "t": 60, "b": 50},
                      font={"family": "Segoe UI, Arial", "color": TEXTO},
                      legend={"orientation": "h", "y": -0.2})
    return fig


# ------------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------------
def fig_sin_con_modelo(porcentaje):
    r = resultado_corte(porcentaje)
    fig = go.Figure(go.Bar(
        x=["Sin modelo", "Con modelo"], y=[TASA, r["mora_con_modelo"]],
        marker_color=[ROJO, VERDE], text=[pct(TASA), pct(r["mora_con_modelo"])],
        textposition="outside", textfont={"size": 18}))
    fig.update_yaxes(tickformat=".0%", range=[0, TASA * 1.3], title="Incumplimiento de la 1.ª cuota")
    return estilo_figura(fig, f"Rechazando al {porcentaje} % más riesgoso, la mora baja de "
                              f"{pct(TASA)} a {pct(r['mora_con_modelo'])}")


def fig_curva(porcentaje):
    paso = max(N // 400, 1)
    x = (np.arange(1, N + 1) / N)[::paso]
    limite = x <= 0.6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[limite], y=morosos_acum[::paso][limite], name="Morosos detectados",
                             line={"color": ROJO, "width": 3}))
    fig.add_trace(go.Scatter(x=x[limite], y=buenos_acum[::paso][limite], name="Buenos clientes rechazados",
                             line={"color": AZUL, "width": 3}))
    fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], name="Rechazar al azar",
                             line={"color": GRIS, "dash": "dash"}))
    r = resultado_corte(porcentaje)
    fig.add_vline(x=porcentaje / 100, line={"color": TEXTO, "dash": "dot"})
    fig.add_trace(go.Scatter(x=[porcentaje / 100] * 2, y=[r["morosos_detectados"], r["buenos_rechazados"]],
                             mode="markers+text", marker={"size": 11, "color": [ROJO, AZUL]},
                             text=[pct(r["morosos_detectados"], 0), pct(r["buenos_rechazados"], 0)],
                             textposition="middle right", showlegend=False))
    fig.update_xaxes(tickformat=".0%", title="Solicitantes rechazados (los de mayor riesgo)")
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="% de cada grupo")
    return estilo_figura(fig, "Cada moroso que se frena cuesta algunos buenos clientes")


def fig_ltv_historial():
    t = pred.groupby(["tramo_ltv", "historial"])["y_true"].mean().reset_index()
    orden_ltv = ["≤60", "60–70", "70–80", ">80"]
    fig = go.Figure()
    for hist, color in [("Con historial", AZUL), ("Sin historial", ROJO)]:
        s = t[t["historial"] == hist].set_index("tramo_ltv").reindex(orden_ltv)
        fig.add_trace(go.Bar(x=orden_ltv, y=s["y_true"], name=hist, marker_color=color,
                             text=[pct(v) for v in s["y_true"]], textposition="outside"))
    fig.add_hline(y=TASA, line={"color": GRIS, "dash": "dash"})
    fig.update_yaxes(tickformat=".0%", range=[0, t["y_true"].max() * 1.3], title="Tasa de impago")
    fig.update_xaxes(title="LTV (% del vehículo financiado)")
    fig.update_layout(barmode="group")
    return estilo_figura(fig, "A más financiamiento, más impago, sobre todo sin historial")


def fig_sin_historial(porcentaje):
    sub = pred[pred["historial"] == "Sin historial"].sort_values("proba", ascending=False)
    k = int(len(sub) * porcentaje / 100)
    valores = [sub["y_true"].mean(), sub["y_true"].iloc[:k].mean(), sub["y_true"].iloc[k:].mean()]
    etiquetas = ["Todo el grupo", f"{porcentaje} % más riesgoso", "Resto (aprobados)"]
    fig = go.Figure(go.Bar(x=etiquetas, y=valores, marker_color=[GRIS, ROJO, VERDE],
                           text=[pct(v) for v in valores], textposition="outside", textfont={"size": 16}))
    fig.update_yaxes(tickformat=".0%", range=[0, max(valores) * 1.3], title="Tasa de impago")
    return estilo_figura(fig, "Sin historial: el modelo separa a los más riesgosos aunque no haya score")


def fig_canal(tipo):
    t = canal[canal["canal"] == tipo].sort_values("tasa_impago")
    if tipo == "Sucursal":                       # solo los extremos, para que se lea
        t = pd.concat([t.head(10), t.tail(10)])
    etiquetas = [f"{tipo} {c}" for c in t["codigo"]]
    colores = [VERDE if v < TASA_GLOBAL else ROJO for v in t["tasa_impago"]]
    fig = go.Figure(go.Bar(y=etiquetas, x=t["tasa_impago"], orientation="h", marker_color=colores,
                           text=[f"{pct(v)}  (n={miles(n)})" for v, n in zip(t["tasa_impago"], t["creditos"])],
                           textposition="outside"))
    fig.add_vline(x=TASA_GLOBAL, line={"color": GRIS, "dash": "dash"})
    fig.update_xaxes(tickformat=".0%", range=[0, t["tasa_impago"].max() * 1.35], title="Tasa de impago")
    titulo = (f"{tipo}: verde = bajo el promedio ({pct(TASA_GLOBAL)}), rojo = sobre el promedio"
              + (" · 10 mejores y 10 peores" if tipo == "Sucursal" else ""))
    return estilo_figura(fig, titulo, alto=max(380, 26 * len(t) + 120))


# ------------------------------------------------------------------
# Piezas de la página
# ------------------------------------------------------------------
CAJA = {"backgroundColor": TARJETA, "borderRadius": "12px", "padding": "18px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "marginBottom": "18px"}


def tarjeta(titulo, id_valor, color):
    return html.Div([
        html.Div(titulo, style={"fontSize": "14px", "color": GRIS}),
        html.Div(id=id_valor, style={"fontSize": "34px", "fontWeight": "700", "color": color}),
    ], style={**CAJA, "flex": "1", "minWidth": "200px", "borderTop": f"5px solid {color}"})


def seccion(titulo, hijos):
    return html.Div([html.H2(titulo, style={"fontSize": "22px", "marginTop": "8px"}), *hijos], style=CAJA)


def texto_calidad():
    """Línea pequeña con el AUC; si config_dash.json no trae esos datos, no se muestra."""
    partes = []
    if "auc_final" in cfg:
        partes.append(f"Calidad del modelo ({cfg.get('modelo', 'modelo final')}): ROC_AUC {cfg['auc_final']:.3f}")
    if "auc_variables_iniciales" in cfg:
        partes.append(f"con las variables iniciales: {cfg['auc_variables_iniciales']:.3f}")
    return "  ·  ".join(partes).replace(".", ",")


app = Dash(__name__, title="Riesgo de crédito vehicular")
server = app.server   # para desplegar con gunicorn

app.layout = html.Div(style={"backgroundColor": FONDO, "fontFamily": "Segoe UI, Arial", "color": TEXTO,
                             "padding": "24px", "maxWidth": "1250px", "margin": "0 auto"}, children=[

    html.H1("Riesgo de impago en créditos vehiculares", style={"marginBottom": "4px"}),
    html.Div(f"Gerencias de riesgo y de ventas · {miles(N)} créditos de validación · tasa real de impago {pct(TASA)}",
             style={"color": GRIS, "marginBottom": "18px"}),

    # Deslizador del punto de corte
    html.Div([
        html.Div("¿Qué porcentaje de solicitantes, los de mayor riesgo, se rechaza?",
                 style={"fontWeight": "600", "marginBottom": "8px"}),
        dcc.Slider(id="corte", min=5, max=40, step=5, value=20,
                   marks={v: f"{v} %" for v in range(5, 45, 5)}),
    ], style=CAJA),

    # Tarjetas
    html.Div([
        tarjeta("Incumplimiento sin modelo", "k_sin", ROJO),
        tarjeta("Incumplimiento con modelo", "k_con", VERDE),
        tarjeta("Morosos detectados", "k_det", AZUL),
        tarjeta("Buenos clientes rechazados", "k_bue", AMARILLO),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

    seccion("1. Criterio de aprobación", [
        html.Div([
            html.Div(dcc.Graph(id="g_barras", config={"displayModeBar": False}), style={"flex": "1", "minWidth": "380px"}),
            html.Div(dcc.Graph(id="g_curva", config={"displayModeBar": False}), style={"flex": "1.4", "minWidth": "420px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ]),

    seccion("2. Clientes sin historial en la central de riesgos", [
        html.Div([
            html.Div(dcc.Graph(figure=fig_ltv_historial(), config={"displayModeBar": False}),
                     style={"flex": "1", "minWidth": "420px"}),
            html.Div(dcc.Graph(id="g_sin_hist", config={"displayModeBar": False}),
                     style={"flex": "1", "minWidth": "380px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ]),

    seccion("3. Canales de venta", [
        dcc.RadioItems(id="tipo_canal", value="Estado", inline=True,
                       options=["Estado", "Fabricante", "Sucursal"],
                       inputStyle={"marginRight": "6px", "marginLeft": "14px"}),
        dcc.Graph(id="g_canal", config={"displayModeBar": False}),
    ]),

    html.Div(texto_calidad(), style={"color": GRIS, "fontSize": "13px", "textAlign": "right"}),
])


# ------------------------------------------------------------------
# Interacción
# ------------------------------------------------------------------
@app.callback(
    Output("k_sin", "children"), Output("k_con", "children"), Output("k_det", "children"),
    Output("k_bue", "children"), Output("g_barras", "figure"), Output("g_curva", "figure"),
    Output("g_sin_hist", "figure"),
    Input("corte", "value"))
def actualizar_corte(porcentaje):
    r = resultado_corte(porcentaje)
    return (pct(TASA), pct(r["mora_con_modelo"]), pct(r["morosos_detectados"], 0),
            pct(r["buenos_rechazados"], 0), fig_sin_con_modelo(porcentaje), fig_curva(porcentaje),
            fig_sin_historial(porcentaje))


@app.callback(Output("g_canal", "figure"), Input("tipo_canal", "value"))
def actualizar_canal(tipo):
    return fig_canal(tipo)


def puerto_libre(inicio=8050):
    """Devuelve 8050 o, si está ocupado, el siguiente puerto libre."""
    import socket
    for puerto in range(inicio, inicio + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", puerto)) != 0:
                return puerto
    return inicio


if __name__ == "__main__":
    import threading
    import webbrowser
    PUERTO = puerto_libre()
    URL = f"http://127.0.0.1:{PUERTO}"
    print(f"\n  Tablero listo en: {URL}   (Ctrl + C para apagarlo)\n")
    threading.Timer(1.5, lambda: webbrowser.open(URL)).start()   # abre el navegador solo
    app.run(debug=False, port=PUERTO)
