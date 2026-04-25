import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import pickle

# ─────────────────────────────────────────────
# CITY COORDINATES
# ─────────────────────────────────────────────
city_coords = {
    "Quito":         (-0.1807, -78.4678),
    "Guayaquil":     (-2.1700, -79.9224),
    "Cuenca":        (-2.9006, -79.0045),
    "Santo Domingo": (-0.2530, -79.1710),
    "Ambato":        (-1.2491, -78.6168),
    "Manta":         (-0.9677, -80.7089),
    "Machala":       (-3.2581, -79.9553),
    "Esmeraldas":    ( 0.9592, -79.6530),
    "Loja":          (-3.9931, -79.2042),
    "Ibarra":        ( 0.3392, -78.1223),
    "Latacunga":     (-0.9352, -78.6155),
    "Riobamba":      (-1.6700, -78.6470),
    "Salinas":       (-2.2145, -80.9584),
    "Cayambe":       ( 0.0292, -78.1453),
    "El Carmen":     (-0.2500, -79.4500),
    "Libertad":      (-2.2330, -80.9000),
    "Quevedo":       (-1.0333, -79.4500),
    "Guaranda":      (-1.6000, -79.0000),
    "Playas":        (-2.6333, -80.3833),
    "Babahoyo":      (-1.8167, -79.5333),
    "Daule":         (-1.8667, -79.9833),
    "Puyo":          (-1.4900, -77.9900),
}

# ─────────────────────────────────────────────
# LOAD & PREP DATA
# ─────────────────────────────────────────────
train_df        = pd.read_csv("train.csv",            parse_dates=["date"])
stores_df       = pd.read_csv("stores.csv")           # keep original 'type' col
holidays_df_raw = pd.read_csv("holidays_events.csv",  parse_dates=["date"])
oil_df          = pd.read_csv("oil.csv",               parse_dates=["date"])
transactions_df = pd.read_csv("transactions.csv",      parse_dates=["date"])

train_df["year"]  = train_df["date"].dt.year
train_df["month"] = train_df["date"].dt.month

# ── Build merged exactly as the model notebook did ──────────────────────────
# Deduplicate holidays: some dates have 2-3 entries → keep first to avoid
# row explosion that breaks the time-series prediction alignment
_hols_dedup = (holidays_df_raw
               .sort_values("date")
               .drop_duplicates("date", keep="first"))

_df = train_df.merge(stores_df, on="store_nbr", how="left")
_hols = _hols_dedup[["date","type","locale","locale_name","description","transferred"]]
_df = _df.merge(_hols, on="date", how="left", suffixes=("","_holiday"))
_df = _df.rename(columns={"type_holiday": "holiday_type"})
_df = _df.merge(oil_df[["date","dcoilwtico"]], on="date", how="left")
for _col in ["holiday_type","locale","locale_name","description"]:
    _df[_col] = _df[_col].fillna("No Holiday")
_df["transferred"] = _df["transferred"].fillna(False)
_df["dcoilwtico"]  = _df["dcoilwtico"].bfill().ffill()

# Build category maps (same order as notebook's cat.codes)
MODEL_FEATURE_COLS = ['store_nbr','family','onpromotion','type','city','state',
                      'cluster','holiday_type','locale','locale_name','description',
                      'transferred','dcoilwtico']
_X_ref = _df[MODEL_FEATURE_COLS].copy()
CAT_MAPS = {}  # col -> sorted list of category strings (index = encoded int)
for _c in ['family','type','city','state','holiday_type','locale','locale_name','description','transferred']:
    CAT_MAPS[_c] = _X_ref[_c].astype('category').cat.categories.tolist()

# ── Build merged for dashboard charts (store_type renamed for clarity) ───────
stores_df = stores_df.rename(columns={"type": "store_type"})
holidays_df = holidays_df_raw.rename(columns={"type": "holiday_type"})
merged = train_df.merge(stores_df, on="store_nbr", how="left")

# Pre-compute for speed
sales_per_day = train_df.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
merged_trans  = transactions_df.merge(sales_per_day, on=["date", "store_nbr"], how="left")
merged_trans["basket_size"] = merged_trans["sales"] / merged_trans["transactions"]

# Monthly store line data
monthly_store = merged.groupby(["store_nbr", "year", "month"])["sales"].sum().reset_index()

# Holiday + sales
holiday_sales = merged.merge(
    holidays_df[["date", "holiday_type", "locale", "locale_name"]],
    on="date", how="left"
)
holiday_sales["is_holiday"] = holiday_sales["holiday_type"].notna()

# Transactions by store (total)
trans_store = transactions_df.groupby("store_nbr")["transactions"].sum().reset_index()

# Transactions by store per day (for avg daily per store)
daily_store_sales = merged.groupby(["store_nbr", "date"])["sales"].sum().reset_index()

# City-level bubble data for map
city_sales_map = merged.groupby("city")["sales"].sum().reset_index()
city_sales_map["lat"] = city_sales_map["city"].map(lambda c: city_coords.get(c, (0, 0))[0])
city_sales_map["lon"] = city_sales_map["city"].map(lambda c: city_coords.get(c, (0, 0))[1])

# Global KPI values
total_sales_global  = merged["sales"].sum()
total_stores_global = stores_df["store_nbr"].nunique()
total_families_glob = merged["family"].nunique()
total_trans_global  = transactions_df["transactions"].sum()
avg_daily_global    = merged.groupby("date")["sales"].sum().mean()

# ── ML model ──────────────────────────────────────────────────────────────
# NOTE: The training notebook (store_forcasting.ipynb) contains a bug on its
# last cell — it saves `model` (the unfitted base XGBRegressor) instead of
# `best_model` (the GridSearchCV-fitted estimator).
#
# TO FIX IN YOUR NOTEBOOK: change the last cell to:
#     import joblib
#     joblib.dump(best_model, "model.pkl")   # ← use best_model, not model
#
# The dashboard loads with pickle (robust) and validates the model is fitted.
MODEL_LOADED = False
model = None
MODEL_LOAD_ERROR = ""

try:
    # Load with pickle as recommended
    with open("model.pkl", "rb") as _f:
        model = pickle.load(_f)
    # Verify it is actually fitted (not a bare unfitted estimator)
    try:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(model)
        MODEL_LOADED = True
    except Exception:
        MODEL_LOAD_ERROR = (
            "model.pkl loaded but the model is NOT fitted. "
            "In your notebook's last cell, change: joblib.dump(model, 'model.pkl') "
            "→ joblib.dump(best_model, 'model.pkl'), re-run, and replace model.pkl."
        )
except FileNotFoundError:
    MODEL_LOAD_ERROR = (
        "model.pkl not found. "
        "Run store_forcasting.ipynb, fix the last cell to save best_model "
        "(not model), then place model.pkl next to this script."
    )
except Exception as _load_ex:
    MODEL_LOAD_ERROR = f"Could not load model.pkl: {_load_ex}"


def encode_and_predict(feat_dict):
    """
    Encode exactly as the notebook did (pd.Categorical label encoding),
    then predict with the XGBoost model.
    feat_dict keys: store_nbr, family, onpromotion, type, city, state,
                    cluster, holiday_type, locale, locale_name,
                    description, transferred, dcoilwtico
    """
    if not MODEL_LOADED:
        raise ValueError(MODEL_LOAD_ERROR)
    row = {}
    # Numeric cols — pass through
    row["store_nbr"]   = int(feat_dict["store_nbr"])
    row["onpromotion"] = int(feat_dict.get("onpromotion", 0))
    row["cluster"]     = int(feat_dict["cluster"])
    row["dcoilwtico"]  = float(feat_dict.get("dcoilwtico", 50.0))
    # Categorical cols — encode to integer code via CAT_MAPS
    for col in ['family','type','city','state','holiday_type',
                'locale','locale_name','description','transferred']:
        val  = feat_dict[col]
        cats = CAT_MAPS[col]
        # transferred stored as bool or string
        if col == 'transferred':
            val = bool(val) if not isinstance(val, bool) else val
        code = cats.index(val) if val in cats else -1
        row[col] = code
    X = pd.DataFrame([row])[MODEL_FEATURE_COLS]
    return float(model.predict(X)[0])

# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
BG_PAGE  = "#0f1b2d"
BG_CARD  = "#162032"
BG_CARD2 = "#1a2840"
NAV_BG   = "#0d1825"
BORDER   = "#1e3050"

C_ORANGE = "#f5a623"
C_RED    = "#e05c5c"
C_GREEN  = "#4caf80"
C_BLUE   = "#4a9eff"
C_PINK   = "#e040fb"
C_CYAN   = "#26c6da"
C_YELLOW = "#ffd740"
C_WHITE  = "#e8eaf0"
C_MUTED  = "#7b93b3"

PIE_COLORS = [C_ORANGE, C_RED, C_GREEN, C_BLUE, C_PINK, C_CYAN, C_YELLOW]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C_WHITE, family="Segoe UI, Arial, sans-serif", size=11),
    margin=dict(l=10, r=10, t=38, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_MUTED, size=10),
                orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    xaxis=dict(showgrid=False, color=C_MUTED, tickfont=dict(size=10),
               linecolor=BORDER, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
               tickfont=dict(size=10), zeroline=False),
)

FF = "Segoe UI, Arial, sans-serif"
EXT_CSS = ["https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def card(children, width="100%", extra=None):
    style = {
        "backgroundColor": BG_CARD, "borderRadius": "10px",
        "border": f"1px solid {BORDER}", "padding": "16px 18px",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.5)",
        "width": width, "boxSizing": "border-box",
    }
    if extra:
        style.update(extra)
    return html.Div(children, style=style)


def lbl(text):
    return html.P(text, style={
        "color": C_MUTED, "fontSize": "10px", "letterSpacing": "2.5px",
        "textTransform": "uppercase", "margin": "0 0 10px 0",
        "fontFamily": FF, "fontWeight": "600"
    })


def dd(id_, opts, val, width="180px"):
    s = {"backgroundColor": BG_CARD2, "color": C_WHITE,
         "border": f"1px solid {BORDER}", "borderRadius": "6px",
         "fontSize": "12px", "marginBottom": "10px"}
    return html.Div(
        dcc.Dropdown(id=id_, options=opts, value=val, clearable=False, style=s),
        style={"width": width}
    )


def gr(gid, h="320px"):
    return dcc.Graph(id=gid, config={"displayModeBar": False}, style={"height": h})


def nav(active):
    def lnk(label, href, pg):
        s = {"color": C_ORANGE, "borderBottom": f"2px solid {C_ORANGE}", "paddingBottom": "4px"} \
            if active == pg else {"color": C_MUTED}
        s.update({"textDecoration": "none", "marginLeft": "28px",
                  "fontSize": "13px", "fontWeight": "600", "fontFamily": FF})
        return html.A(label, href=href, style=s)

    return html.Div([
        html.Div([
            html.Span("🛒", style={"fontSize": "20px", "marginRight": "10px"}),
            html.Span("RetailIQ", style={"fontSize": "20px", "fontWeight": "700",
                                          "color": C_WHITE, "letterSpacing": "2px", "fontFamily": FF}),
            html.Span(" · Ecuador", style={"fontSize": "12px", "color": C_MUTED, "marginLeft": "6px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([lnk("📊  Analytics", "/", 1),
                  lnk("🔮  Predictions", "/predictions", 2),
                  lnk("🏪  Store Insights", "/insights", 3)],
                 style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "backgroundColor": NAV_BG, "padding": "14px 32px",
              "borderBottom": f"1px solid {BORDER}",
              "position": "sticky", "top": "0", "zIndex": "999"})


def kpi_box(label, val_id, color):
    return card([
        html.P(label, style={"color": C_MUTED, "fontSize": "10px", "letterSpacing": "2px",
                              "textTransform": "uppercase", "margin": "0 0 6px 0", "fontFamily": FF}),
        html.Div(id=val_id, style={"color": color, "fontSize": "28px",
                                    "fontWeight": "700", "fontFamily": FF}),
    ], width="19%")

# ─────────────────────────────────────────────
# PAGE 1 LAYOUT
# ─────────────────────────────────────────────
store_opts = [{"label": f"Store {s}", "value": s} for s in sorted(stores_df["store_nbr"].unique())]
city_opts  = [{"label": c, "value": c} for c in sorted(merged["city"].unique())]

page1 = html.Div(style={"backgroundColor": BG_PAGE, "minHeight": "100vh", "fontFamily": FF,
                          "color": C_WHITE}, children=[
    nav(1),
    html.Div(style={"padding": "20px 28px"}, children=[

        # ── KPI Row with store selector ────────────────────────────────────
        html.Div([
            # Store selector drives the dynamic KPIs
            card([
                lbl("Filter KPIs by store"),
                dd("dd-kpi-store",
                   [{"label": "All stores", "value": "ALL"}] + store_opts,
                   "ALL", width="100%"),
            ], width="18%", extra={"display": "flex", "flexDirection": "column",
                                    "justifyContent": "center"}),
            kpi_box("Total Sales",        "kpi-total-sales",  C_ORANGE),
            kpi_box("Total Transactions", "kpi-total-trans",  C_RED),
            kpi_box("Avg Daily Sales",    "kpi-avg-daily",    C_CYAN),
            card([
                html.P("TOTAL STORES", style={"color": C_MUTED, "fontSize": "10px",
                                               "letterSpacing": "2px", "textTransform": "uppercase",
                                               "margin": "0 0 6px 0", "fontFamily": FF}),
                html.Div(str(total_stores_global),
                         style={"color": C_BLUE, "fontSize": "28px",
                                "fontWeight": "700", "fontFamily": FF}),
            ], width="19%"),
            card([
                html.P("PRODUCT FAMILIES", style={"color": C_MUTED, "fontSize": "10px",
                                                    "letterSpacing": "2px", "textTransform": "uppercase",
                                                    "margin": "0 0 6px 0", "fontFamily": FF}),
                html.Div(str(total_families_glob),
                         style={"color": C_GREEN, "fontSize": "28px",
                                "fontWeight": "700", "fontFamily": FF}),
            ], width="19%"),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "18px",
                  "alignItems": "stretch"}),

        # ── ROW A: Store bar  +  Top-5 donut with city filter ─────────────
        html.Div([

            # A1 – Sales by product per store (red horizontal bar)
            card([
                lbl("Sales by product family per store"),
                dd("dd-store-bar", store_opts, 1),
                gr("store-hbar", "400px"),
            ], width="49%"),

            # A2 – Top-5 donut with city dropdown + side text panel
            card([
                lbl("Top 5 product families – share of total sales"),
                dd("dd-city-donut",
                   [{"label": "All cities", "value": "ALL"}] + city_opts, "ALL"),
                # donut + legend text side-by-side
                html.Div([
                    # Donut pushed to right half
                    html.Div(gr("top5-donut", "350px"),
                             style={"width": "55%", "flexShrink": "0"}),
                    # Text panel on the right
                    html.Div(id="top5-text-panel",
                             style={"width": "42%", "display": "flex",
                                    "flexDirection": "column", "justifyContent": "center",
                                    "paddingLeft": "10px"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ], width="49%"),

        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px",
                  "alignItems": "flex-start"}),

        # ── ROW B: Yearly donut (store filter)  +  Holiday bar (blue) ─────
        html.Div([

            # B1 – Yearly donut with store dropdown
            card([
                lbl("Sales distribution by year (% of total)"),
                dd("dd-store-yearly", [{"label": "All stores", "value": "ALL"}] + store_opts, "ALL"),
                gr("yearly-pie", "320px"),
            ], width="36%"),

            # B2 – Holiday impact bar (blue)
            card([
                lbl("Average sales impact by holiday type"),
                gr("holiday-bar", "320px"),
            ], width="62%"),

        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px",
                  "alignItems": "flex-start"}),

        # ── ROW C: City location bar (text-only)  +  Transactions bar ─────
        html.Div([

            # C1 – City horizontal bar – plain text box "All cities", no dropdown
            card([
                lbl("Sales by location – share of total"),
                html.Div([
                    html.Div("All cities", style={
                        "backgroundColor": BG_CARD2, "color": C_MUTED,
                        "border": f"1px solid {BORDER}", "borderRadius": "6px",
                        "fontSize": "13px", "padding": "8px 14px",
                        "marginBottom": "10px", "width": "fit-content",
                        "fontFamily": FF, "letterSpacing": "0.5px",
                        "userSelect": "none",
                    }),
                ]),
                gr("city-hbar", "380px"),
            ], width="55%"),

            # C2 – Transactions by store
            card([
                lbl("Total transactions by store number"),
                gr("trans-store-bar", "420px"),
            ], width="43%"),

        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px",
                  "alignItems": "flex-start"}),

        # ── ROW D: Geographic bubble map ───────────────────────────────────
        card([
            lbl("Geographic sales distribution – bubble map"),
            gr("geo-map", "420px"),
        ], extra={"marginBottom": "18px"}),

    ])
])

# ─────────────────────────────────────────────
# PREDICTION FORM HELPERS
# ─────────────────────────────────────────────
def _dds():
    return {"backgroundColor": BG_CARD2, "color": C_WHITE,
            "border": f"1px solid {BORDER}", "borderRadius": "6px",
            "fontSize": "12px", "marginBottom": "6px"}

def _inp():
    return {"backgroundColor": BG_CARD2, "color": C_WHITE,
            "border": f"1px solid {BORDER}", "borderRadius": "6px",
            "padding": "8px 10px", "fontSize": "13px"}

def _pred_field(label_text, component):
    return html.Div([
        html.Label(label_text, style={"color": C_MUTED, "fontSize": "10px",
                                       "letterSpacing": "1px", "textTransform": "uppercase",
                                       "display": "block", "marginBottom": "4px",
                                       "fontFamily": FF}),
        component,
    ], style={"flex": "1"})

# ─────────────────────────────────────────────
# PAGE 2 LAYOUT
# ─────────────────────────────────────────────
page2 = html.Div(style={"backgroundColor": BG_PAGE, "minHeight": "100vh", "fontFamily": FF,
                          "color": C_WHITE}, children=[
    nav(2),
    html.Div(style={"padding": "20px 28px"}, children=[

        # Monthly store line chart
        card([
            lbl("Monthly sales per store across years"),
            dd("dd-store-line", store_opts, 1),
            gr("store-monthly-line", "360px"),
        ], extra={"marginBottom": "18px"}),

        # Prediction + bubble map
        html.Div([
            card([
                lbl("ML sales forecast – all model features"),
                # Row 1: store | family | onpromotion | type | oil price
                html.Div([
                    _pred_field("store_nbr", dcc.Dropdown(
                        id="pred-store", options=store_opts, value=1, clearable=False,
                        style=_dds())),
                    _pred_field("family", dcc.Dropdown(
                        id="pred-family",
                        options=[{"label": f, "value": f} for f in sorted(train_df["family"].unique())],
                        value="GROCERY I", clearable=False, style=_dds())),
                    _pred_field("onpromotion", dcc.Input(
                        id="pred-promo", type="number", value=0, min=0, max=741,
                        style={**_inp(), "width": "100%"})),
                    _pred_field("type (store)", dcc.Dropdown(
                        id="pred-type",
                        options=[{"label": t, "value": t} for t in ["A","B","C","D","E"]],
                        value="D", clearable=False, style=_dds())),
                    _pred_field("dcoilwtico (oil)", dcc.Input(
                        id="pred-oil", type="number", value=50.0, min=26.0, max=111.0, step=0.1,
                        style={**_inp(), "width": "100%"})),
                ], style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),
                # Row 2: city | state | cluster | holiday_type | locale
                html.Div([
                    _pred_field("city", dcc.Dropdown(
                        id="pred-city",
                        options=[{"label": c, "value": c} for c in sorted(stores_df["city"].unique())],
                        value="Quito", clearable=False, style=_dds())),
                    _pred_field("state", dcc.Dropdown(
                        id="pred-state",
                        options=[{"label": s, "value": s} for s in sorted(stores_df["state"].unique())],
                        value="Pichincha", clearable=False, style=_dds())),
                    _pred_field("cluster", dcc.Dropdown(
                        id="pred-cluster",
                        options=[{"label": str(c), "value": c} for c in sorted(stores_df["cluster"].unique())],
                        value=13, clearable=False, style=_dds())),
                    _pred_field("holiday_type", dcc.Dropdown(
                        id="pred-htype",
                        options=[{"label": h, "value": h}
                                 for h in ["No Holiday","Holiday","Additional","Bridge","Event","Transfer","Work Day"]],
                        value="No Holiday", clearable=False, style=_dds())),
                    _pred_field("locale", dcc.Dropdown(
                        id="pred-locale",
                        options=[{"label": l, "value": l} for l in ["No Holiday","Local","National","Regional"]],
                        value="No Holiday", clearable=False, style=_dds())),
                ], style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),
                # Row 3: locale_name | description | transferred
                html.Div([
                    _pred_field("locale_name", dcc.Dropdown(
                        id="pred-locale-name",
                        options=[{"label": ln, "value": ln}
                                 for ln in sorted(holidays_df["locale_name"].dropna().unique())],
                        value="Ecuador", clearable=False, style=_dds())),
                    _pred_field("description", dcc.Dropdown(
                        id="pred-desc",
                        options=[{"label": d, "value": d}
                                 for d in CAT_MAPS["description"]],
                        value="No Holiday", clearable=False, style=_dds())),
                    _pred_field("transferred", dcc.Dropdown(
                        id="pred-transferred",
                        options=[{"label": "False", "value": False},
                                 {"label": "True",  "value": True}],
                        value=False, clearable=False, style=_dds())),
                    html.Div(style={"flex": "2"}),  # spacer
                ], style={"display": "flex", "gap": "10px", "marginBottom": "16px"}),
                html.Div(id="pred-output"),
                html.Div(style={"height":"12px"}),
                html.P("ACTUAL VS PREDICTED – MONTHLY TIME SERIES", style={
                    "color": C_MUTED, "fontSize": "10px", "letterSpacing": "2.5px",
                    "textTransform": "uppercase", "margin": "0 0 6px 0", "fontFamily": FF
                }),
                gr("pred-ts-graph", "300px"),
            ], width="100%"),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px"}),

        # Bubble map + trans vs store  (now in their own row)
        html.Div([
            card([
                lbl("Store locations – bubble = total transactions"),
                gr("store-bubble-map", "320px"),
            ], width="49%"),
            card([
                lbl("Transactions vs store number – all stores"),
                gr("trans-vs-store2", "320px"),
            ], width="49%"),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px"}),



    ])
])

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=EXT_CSS,
                suppress_callback_exceptions=True)
app.title = "RetailIQ Dashboard"

# Fix dropdown text color: Dash renders option text in its own div;
# override via index_string CSS injection
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
/* ══ Dash 4.x Dropdown class names (from async-dropdown.js) ══ */

/* Selected value text & placeholder */
.dash-dropdown-value,
.dash-dropdown-value-item,
.dash-dropdown-placeholder {
    color: #111111 !important;
}

/* The trigger button (collapsed state) */
.dash-dropdown-trigger {
    background-color: #1a2840 !important;
    border-color: #1e3050 !important;
    color: #111111 !important;
}

/* Dropdown options list panel */
.dash-dropdown-content {
    background-color: #ffffff !important;
    border: 1px solid #cccccc !important;
    z-index: 9999 !important;
}

/* Each option row */
.dash-dropdown-option {
    color: #111111 !important;
    background-color: #ffffff !important;
    padding: 8px 12px !important;
}
.dash-dropdown-option:hover,
.dash-dropdown-option[aria-selected="true"] {
    background-color: #dce8ff !important;
    color: #000000 !important;
}

/* Search box inside open dropdown */
.dash-dropdown-search input,
.dash-dropdown-search {
    color: #111111 !important;
    background-color: #f5f5f5 !important;
}

/* Wrapper */
.dash-dropdown-wrapper {
    background-color: #1a2840 !important;
}

/* Date picker inputs */
.DateInput_input, .DateInput input {
    color: #111111 !important;
    background-color: #ffffff !important;
    font-size: 13px !important;
}

/* Number input boxes stay light-on-dark */
input[type="number"] {
    color: #e8eaf0 !important;
    background-color: #1a2840 !important;
}
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
])

# ── Routing ──────────────────────────────────
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/predictions":
        return page2
    if pathname == "/insights":
        return page3
    return page1

# ─────────────────────────────────────────────
# PAGE 1 CALLBACKS
# ─────────────────────────────────────────────

# KPI cards – driven by store dropdown
@app.callback(
    [Output("kpi-total-sales", "children"),
     Output("kpi-total-trans", "children"),
     Output("kpi-avg-daily",   "children")],
    Input("dd-kpi-store", "value")
)
def cb_kpi(store):
    if store == "ALL":
        ts = total_sales_global
        tt = total_trans_global
        ad = avg_daily_global
    else:
        s  = int(store)
        ts = merged[merged["store_nbr"] == s]["sales"].sum()
        tt = int(transactions_df[transactions_df["store_nbr"] == s]["transactions"].sum())
        ad = daily_store_sales[daily_store_sales["store_nbr"] == s]["sales"].mean()

    def fmt(v):
        if v >= 1e6:
            return f"{v/1e6:.1f}M"
        if v >= 1e3:
            return f"{v/1e3:.1f}K"
        return f"{v:,.0f}"

    return fmt(ts), fmt(tt), f"{ad:,.0f}"


# A1 – Store horizontal bar (red, sorted descending, x-label = "Amount Of Sales")
@app.callback(Output("store-hbar", "figure"), Input("dd-store-bar", "value"))
def cb_store_hbar(store):
    store = int(store)
    df = merged[merged["store_nbr"] == store]
    grp = df.groupby("family")["sales"].sum().sort_values(ascending=True).reset_index()
    grp["pct"] = (grp["sales"] / grp["sales"].sum() * 100).round(1)

    # graduated red shades for bars
    n = len(grp)
    red_scale = [f"rgba(224, 92, 92, {0.35 + 0.65 * i / max(n-1,1):.2f})" for i in range(n)]

    fig = go.Figure(go.Bar(
        x=grp["sales"], y=grp["family"],
        orientation="h",
        marker=dict(color=red_scale),
        text=[f"{p:.1f}%" for p in grp["pct"]],
        textposition="outside",
        textfont=dict(color=C_MUTED, size=9),
        hovertemplate="<b>%{y}</b><br>Amount Of Sales: %{x:,.0f}<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["margin"] = dict(l=10, r=55, t=10, b=40)
    lay["xaxis"]  = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                         tickfont=dict(size=10), zeroline=False,
                         title=dict(text="Amount Of Sales", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"]  = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=9), zeroline=False)
    fig.update_layout(**lay)
    return fig


# A2 – Top-5 donut + side text (city filter → % of city total)
@app.callback(
    [Output("top5-donut",     "figure"),
     Output("top5-text-panel","children")],
    Input("dd-city-donut", "value")
)
def cb_top5_donut(city):
    if city == "ALL":
        df = merged
    else:
        city_stores = stores_df[stores_df["city"] == city]["store_nbr"]
        df = merged[merged["store_nbr"].isin(city_stores)]

    city_total = df["sales"].sum()
    top5 = df.groupby("family")["sales"].sum().nlargest(5).reset_index()
    top5["pct"] = (top5["sales"] / city_total * 100).round(1)

    fig = go.Figure(go.Pie(
        labels=top5["family"],
        values=top5["sales"],
        hole=0.52,
        marker=dict(colors=PIE_COLORS[:5], line=dict(color=BG_PAGE, width=2)),
        textinfo="percent",
        textfont=dict(size=10, color=C_WHITE),
        hovertemplate="<b>%{label}</b><br>Sales: %{value:,.0f}<br>%{percent} of city<extra></extra>",
        direction="clockwise",
        sort=True,
        showlegend=False,
    ))
    lay = {**BASE_LAYOUT}
    lay.pop("xaxis", None); lay.pop("yaxis", None)
    lay["margin"] = dict(l=0, r=0, t=10, b=10)
    lay["annotations"] = [dict(text="<b>Top 5</b>", x=0.5, y=0.5,
                               font=dict(size=14, color=C_ORANGE), showarrow=False)]
    fig.update_layout(**lay)

    # Side text: rank + family + pct
    text_items = []
    for i, row in top5.iterrows():
        col = PIE_COLORS[i % len(PIE_COLORS)]
        text_items.append(html.Div([
            html.Div(style={
                "width": "10px", "height": "10px", "borderRadius": "50%",
                "backgroundColor": col, "flexShrink": "0", "marginTop": "3px"
            }),
            html.Div([
                html.Span(row["family"], style={"color": C_WHITE, "fontSize": "11px",
                                                "fontWeight": "600", "display": "block"}),
                html.Span(f"{row['pct']:.1f}% of city sales",
                          style={"color": C_MUTED, "fontSize": "10px"}),
            ], style={"marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "flex-start",
                  "marginBottom": "12px"}))

    return fig, text_items


# B1 – Yearly donut (store filter)
@app.callback(Output("yearly-pie", "figure"), Input("dd-store-yearly", "value"))
def cb_yearly_pie(store):
    if store == "ALL":
        df = merged
    else:
        df = merged[merged["store_nbr"] == int(store)]

    yt = df.groupby("year")["sales"].sum().reset_index()
    yt["pct"] = (yt["sales"] / yt["sales"].sum() * 100).round(1)

    colors = [C_RED, C_GREEN, C_BLUE, C_CYAN, C_ORANGE]
    fig = go.Figure(go.Pie(
        labels=[str(y) for y in yt["year"]],
        values=yt["sales"],
        hole=0.50,
        marker=dict(colors=colors[:len(yt)], line=dict(color=BG_PAGE, width=2)),
        textinfo="label+percent",
        textfont=dict(size=10, color=C_WHITE),
        hovertemplate="<b>%{label}</b><br>Sales: %{value:,.0f}<br>%{percent}<extra></extra>",
        direction="clockwise",
    ))
    lay = {**BASE_LAYOUT}
    lay.pop("xaxis", None); lay.pop("yaxis", None)
    lay["margin"] = dict(l=10, r=10, t=10, b=60)
    lay["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_MUTED, size=10),
                         orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5)
    lbl_store = f"Store {store}" if store != "ALL" else "All stores"
    lay["annotations"] = [dict(text=f"<b>by year</b>", x=0.5, y=0.5,
                               font=dict(size=13, color=C_BLUE), showarrow=False)]
    fig.update_layout(**lay)
    return fig


# B2 – Holiday bar (BLUE bars, labelled axes)
@app.callback(Output("holiday-bar", "figure"), Input("url", "pathname"))
def cb_holiday_bar(_):
    grp = (holiday_sales.groupby("holiday_type")["sales"]
           .mean().dropna().sort_values(ascending=False).reset_index())
    baseline = holiday_sales[~holiday_sales["is_holiday"]]["sales"].mean()
    baseline_row = pd.DataFrame([{"holiday_type": "Non-Holiday", "sales": baseline}])
    grp = pd.concat([grp, baseline_row], ignore_index=True)

    colors = [C_BLUE if h != "Non-Holiday" else C_MUTED for h in grp["holiday_type"]]

    fig = go.Figure(go.Bar(
        x=grp["holiday_type"], y=grp["sales"],
        marker=dict(color=colors),
        text=[f"{v:,.0f}" for v in grp["sales"]],
        textposition="outside",
        textfont=dict(color=C_WHITE, size=10),
        hovertemplate="<b>%{x}</b><br>Avg Sales: %{y:,.0f}<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["margin"] = dict(l=55, r=10, t=30, b=55)
    lay["xaxis"] = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=10),
                        zeroline=False, linecolor=BORDER,
                        title=dict(text="Holiday Type", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"] = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                        tickfont=dict(size=10), zeroline=False,
                        title=dict(text="Average Sales", font=dict(color=C_MUTED, size=11)))
    fig.update_layout(**lay)
    return fig


# C1 – City horizontal bar (static – no dropdown, always all cities)
@app.callback(Output("city-hbar", "figure"), Input("url", "pathname"))
def cb_city_hbar(_):
    grp = (merged.groupby("city")["sales"].sum()
           .sort_values(ascending=True).reset_index())
    total = grp["sales"].sum()
    grp["pct"] = (grp["sales"] / total * 100).round(1)

    fig = go.Figure(go.Bar(
        x=grp["sales"], y=grp["city"],
        orientation="h",
        marker=dict(color=grp["sales"],
                    colorscale=[[0, BG_CARD2], [1, C_BLUE]], showscale=False),
        text=[f"{p:.1f}%" for p in grp["pct"]],
        textposition="outside",
        textfont=dict(color=C_MUTED, size=9),
        hovertemplate="<b>%{y}</b><br>Sales: %{x:,.0f}<br>%{text} of total<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["margin"] = dict(l=10, r=60, t=10, b=10)
    lay["yaxis"] = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=9), zeroline=False)
    lay["xaxis"] = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                        tickfont=dict(size=10), zeroline=False)
    fig.update_layout(**lay)
    return fig


# C2 – Transactions by store (vertical, sorted desc, green gradient)
@app.callback(Output("trans-store-bar", "figure"), Input("url", "pathname"))
def cb_trans_store(_):
    df = trans_store.sort_values("transactions", ascending=False)
    fig = go.Figure(go.Bar(
        x=df["store_nbr"].astype(str),
        y=df["transactions"],
        marker=dict(color=df["transactions"],
                    colorscale=[[0, BG_CARD2], [1, C_GREEN]], showscale=False),
        hovertemplate="Store <b>%{x}</b><br>Transactions: %{y:,.0f}<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["margin"] = dict(l=10, r=10, t=10, b=40)
    lay["xaxis"] = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=8),
                        zeroline=False, linecolor=BORDER,
                        title=dict(text="Store Number", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"] = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                        tickfont=dict(size=10), zeroline=False,
                        title=dict(text="Total Transactions", font=dict(color=C_MUTED, size=11)))
    fig.update_layout(**lay)
    return fig


# D – Geographic bubble map
@app.callback(Output("geo-map", "figure"), Input("url", "pathname"))
def cb_geo_map(_):
    df = city_sales_map[city_sales_map["lat"] != 0].copy()
    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", size="sales", size_max=60,
        hover_name="city",
        hover_data={"sales": ":,.0f", "lat": False, "lon": False},
        color="sales",
        color_continuous_scale=[[0, C_BLUE], [0.5, C_ORANGE], [1, C_RED]],
        zoom=5, center={"lat": -1.8, "lon": -78.5},
        mapbox_style="open-street-map",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────
# PAGE 2 CALLBACKS
# ─────────────────────────────────────────────

# Monthly store line chart
@app.callback(Output("store-monthly-line", "figure"), Input("dd-store-line", "value"))
def cb_store_monthly_line(store):
    store = int(store)
    df = monthly_store[monthly_store["store_nbr"] == store].copy()
    colors_yr = {2013: C_BLUE, 2014: C_GREEN, 2015: C_ORANGE, 2016: C_RED, 2017: C_CYAN}

    fig = go.Figure()
    for yr in sorted(df["year"].unique()):
        grp = df[df["year"] == yr].sort_values("month")
        fig.add_trace(go.Scatter(
            x=grp["month"], y=grp["sales"], mode="lines+markers",
            name=str(yr), line=dict(color=colors_yr.get(int(yr), C_WHITE), width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{yr}</b> · Month %{{x}}<br>Sales: %{{y:,.0f}}<extra></extra>",
        ))
    lay = {**BASE_LAYOUT}
    lay["xaxis"] = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=10),
                        zeroline=False, linecolor=BORDER,
                        tickmode="array",
                        tickvals=list(range(1, 13)),
                        ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                                  "Jul","Aug","Sep","Oct","Nov","Dec"],
                        title=dict(text="Month", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"] = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                        tickfont=dict(size=10), zeroline=False,
                        title=dict(text="Total Sales", font=dict(color=C_MUTED, size=11)))
    lay["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_MUTED, size=10),
                         orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
    fig.update_layout(**lay)
    return fig


# Store bubble map
@app.callback(Output("store-bubble-map", "figure"), Input("url", "pathname"))
def cb_store_bubble_map(_):
    st = transactions_df.groupby("store_nbr")["transactions"].sum().reset_index()
    info = stores_df.merge(st, on="store_nbr", how="left")
    info["lat"] = info["city"].map(lambda c: city_coords.get(c, (0, 0))[0])
    info["lon"] = info["city"].map(lambda c: city_coords.get(c, (0, 0))[1])
    info = info[info["lat"] != 0]

    fig = px.scatter_mapbox(
        info, lat="lat", lon="lon", size="transactions", size_max=45,
        hover_name="store_nbr",
        hover_data={"city": True, "transactions": ":,.0f", "lat": False, "lon": False},
        color="transactions",
        color_continuous_scale=[[0, C_BLUE], [0.5, C_GREEN], [1, C_ORANGE]],
        zoom=5, center={"lat": -1.8, "lon": -78.5},
        mapbox_style="open-street-map",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_showscale=False)
    return fig


# Transactions vs store bar + rolling avg (page 2 version)
@app.callback(Output("trans-vs-store2", "figure"), Input("url", "pathname"))
def cb_trans_vs_store2(_):
    df = trans_store.sort_values("store_nbr")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["store_nbr"].astype(str), y=df["transactions"],
        name="Transactions",
        marker=dict(color=C_ORANGE, opacity=0.85),
        hovertemplate="Store <b>%{x}</b><br>Transactions: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["store_nbr"].astype(str),
        y=df["transactions"].rolling(5, center=True).mean(),
        mode="lines", name="Rolling avg (5)",
        line=dict(color=C_PINK, width=2, dash="dot"),
        hovertemplate="Rolling avg: %{y:,.0f}<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["xaxis"] = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=9),
                        zeroline=False, linecolor=BORDER,
                        title=dict(text="Store Number", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"] = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                        tickfont=dict(size=10), zeroline=False,
                        title=dict(text="Total Transactions", font=dict(color=C_MUTED, size=11)))
    lay["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_MUTED, size=10),
                         orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
    fig.update_layout(**lay)
    return fig


# Prediction output + time series actual vs predicted
@app.callback(
    [Output("pred-output",   "children"),
     Output("pred-ts-graph", "figure")],
    [Input("pred-store",       "value"),
     Input("pred-family",      "value"),
     Input("pred-promo",       "value"),
     Input("pred-type",        "value"),
     Input("pred-city",        "value"),
     Input("pred-state",       "value"),
     Input("pred-cluster",     "value"),
     Input("pred-htype",       "value"),
     Input("pred-locale",      "value"),
     Input("pred-locale-name", "value"),
     Input("pred-desc",        "value"),
     Input("pred-transferred",  "value"),
     Input("pred-oil",         "value")]
)
def cb_prediction(store, family, promo, stype, city, state, cluster,
                  htype, locale, locale_name, desc, transferred, oil):

    # ── Time series: actual vs model predictions (rolling monthly) ─────────
    store_int = int(store) if store else 1
    fam       = family if family else "GROCERY I"
    ts_df = _df[(_df["store_nbr"] == store_int) & (_df["family"] == fam)].copy()
    ts_df = ts_df.sort_values("date")

    # Actual monthly aggregation (agg("first") already returns datetime64)
    ts_monthly = ts_df.groupby(ts_df["date"].dt.to_period("M")).agg(
        date=("date", "first"), actual=("sales", "sum")
    ).reset_index(drop=True)
    ts_monthly["date"] = pd.to_datetime(ts_monthly["date"])

    # Predict each month using model (batch encode)
    pred_vals = []
    pred_dates = []
    if MODEL_LOADED:
        try:
            # Filter from _df and reset index so raw_preds aligns row-by-row
            _mask = (_df["store_nbr"] == store_int) & (_df["family"] == fam)
            _Xb = _df[_mask][MODEL_FEATURE_COLS].copy().reset_index(drop=True)
            # Encode categoricals to integer codes
            for _c in ['family','type','city','state','holiday_type',
                       'locale','locale_name','description','transferred']:
                _cats = CAT_MAPS[_c]
                _Xb[_c] = _Xb[_c].apply(lambda v: _cats.index(v) if v in _cats else -1)
            raw_preds = model.predict(_Xb)
            # Attach predictions to the filtered ts_df (same row count guaranteed by dedup)
            ts_df2 = ts_df.reset_index(drop=True).copy()
            ts_df2["pred"] = raw_preds
            pred_monthly = ts_df2.groupby(ts_df2["date"].dt.to_period("M")).agg(
                date=("date", "first"), predicted=("pred", "sum")
            ).reset_index(drop=True)
            pred_monthly["date"] = pd.to_datetime(pred_monthly["date"])
            pred_vals  = pred_monthly["predicted"].tolist()
            pred_dates = pred_monthly["date"].tolist()
        except Exception as _e:
            pred_vals  = []
            pred_dates = []

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts_monthly["date"], y=ts_monthly["actual"],
        mode="lines", name="Actual Sales",
        line=dict(color=C_ORANGE, width=2),
        hovertemplate="<b>Actual</b><br>%{x|%b %Y}: %{y:,.0f}<extra></extra>",
    ))
    if MODEL_LOADED and len(pred_vals):
        fig_ts.add_trace(go.Scatter(
            x=pred_dates, y=pred_vals,
            mode="lines", name="Predicted Sales",
            line=dict(color=C_CYAN, width=2, dash="dot"),
            hovertemplate="<b>Predicted</b><br>%{x|%b %Y}: %{y:,.0f}<extra></extra>",
        ))
    ts_lay = {**BASE_LAYOUT}
    ts_lay["margin"] = dict(l=55, r=10, t=10, b=40)
    ts_lay["xaxis"]  = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=10),
                             zeroline=False, linecolor=BORDER,
                             title=dict(text="Date", font=dict(color=C_MUTED, size=11)))
    ts_lay["yaxis"]  = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                             tickfont=dict(size=10), zeroline=False,
                             title=dict(text="Monthly Sales", font=dict(color=C_MUTED, size=11)))
    ts_lay["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_MUTED, size=10),
                             orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
    fig_ts.update_layout(**ts_lay)

    # ── Point prediction card ──────────────────────────────────────────────
    if not all([store, family]):
        return (html.P("Select store and family to get a forecast.",
                       style={"color": C_MUTED, "fontSize": "13px"}), fig_ts)
    try:
        feat = {
            "store_nbr":   int(store),
            "family":      family,
            "onpromotion": int(promo)    if promo      is not None else 0,
            "type":        stype         if stype      is not None else "D",
            "city":        city          if city       is not None else "Quito",
            "state":       state         if state      is not None else "Pichincha",
            "cluster":     int(cluster)  if cluster    is not None else 13,
            "holiday_type":htype         if htype      is not None else "No Holiday",
            "locale":      locale        if locale     is not None else "No Holiday",
            "locale_name": locale_name   if locale_name is not None else "Ecuador",
            "description": desc          if desc       is not None else "No Holiday",
            "transferred": bool(transferred) if transferred is not None else False,
            "dcoilwtico":  float(oil)   if oil        is not None else 50.0,
        }
        forecast = encode_and_predict(feat)
        card = html.Div([
            html.Div([
                html.P("FORECASTED SALES", style={"color": C_MUTED, "fontSize": "10px",
                                                   "letterSpacing": "2px", "margin": "0"}),
                html.P(f"{max(forecast, 0):,.0f}",
                       style={"color": C_ORANGE, "fontSize": "44px",
                              "fontWeight": "700", "margin": "4px 0"}),
            ], style={"flex": "1"}),
            html.Div([
                html.Div([html.Span("Store: ",    style={"color": C_MUTED}),
                          html.Span(f"#{feat['store_nbr']} · {feat['city']}",
                                    style={"color": C_WHITE, "fontWeight": "600"})]),
                html.Div([html.Span("Family: ",   style={"color": C_MUTED}),
                          html.Span(family,        style={"color": C_WHITE, "fontWeight": "600"})]),
                html.Div([html.Span("Type: ",     style={"color": C_MUTED}),
                          html.Span(feat["type"],  style={"color": C_WHITE, "fontWeight": "600"})]),
                html.Div([html.Span("Cluster: ",  style={"color": C_MUTED}),
                          html.Span(str(feat["cluster"]), style={"color": C_WHITE, "fontWeight": "600"})]),
                html.Div([html.Span("Oil price: ",style={"color": C_MUTED}),
                          html.Span(f"${feat['dcoilwtico']:.1f}",
                                    style={"color": C_WHITE, "fontWeight": "600"})]),
                html.Div([html.Span("Promo: ",    style={"color": C_MUTED}),
                          html.Span(str(feat["onpromotion"]),
                                    style={"color": C_WHITE, "fontWeight": "600"})]),
            ], style={"fontSize": "13px", "lineHeight": "1.9", "flex": "1"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "24px",
                  "backgroundColor": BG_CARD2, "borderRadius": "8px",
                  "padding": "18px 24px", "border": f"1px solid {C_ORANGE}44"})
        return card, fig_ts
    except Exception as e:
        return (html.P(f"⚠ {str(e)}", style={"color": C_RED, "fontSize": "13px"}),
                fig_ts)



# ─────────────────────────────────────────────
# PAGE 3 LAYOUT – Store Insights & Promotion
# ─────────────────────────────────────────────
# Pre-compute page3 data
stores_per_city = stores_df.groupby("city")["store_nbr"].count().reset_index()
stores_per_city.columns = ["city", "num_stores"]
stores_per_city = stores_per_city.sort_values("num_stores", ascending=False)

# Promotion vs no-promotion yearly stacked data
merged_p3 = merged.copy()
merged_p3["year"] = merged_p3["date"].dt.year
merged_p3["promoted"] = (merged_p3["onpromotion"] > 0).map({True: "Promoted", False: "Not Promoted"})
promo_yearly = merged_p3.groupby(["year", "promoted"])["sales"].sum().reset_index()

# Promo lift per family (avg sales with promo vs without)
family_promo_0   = (merged_p3[merged_p3["promoted"]=="Not Promoted"]
                    .groupby("family")["sales"].mean())
family_promo_yes = (merged_p3[merged_p3["promoted"]=="Promoted"]
                    .groupby("family")["sales"].mean())
promo_lift = ((family_promo_yes - family_promo_0) / family_promo_0 * 100).dropna()
promo_lift_df = promo_lift.reset_index()
promo_lift_df.columns = ["family", "lift_pct"]
promo_lift_df = promo_lift_df.sort_values("lift_pct", ascending=False)

# Sales bucket: avg per day by promo units buckets
BUCKET_LABELS = ["0 (none)", "1-5", "6-15", "16-30", "31-60", "61-200", "200+"]
merged_p3["promo_bucket"] = pd.cut(
    merged_p3["onpromotion"],
    bins=[-1, 0, 5, 15, 30, 60, 200, 742],
    labels=BUCKET_LABELS
).astype(str)   # convert to plain string immediately – avoids Categorical callback issues
promo_bucket_sales = (merged_p3.groupby("promo_bucket", sort=False)["sales"]
                      .mean().reset_index()
                      .rename(columns={"sales": "avg_sales"}))
# Restore correct order
promo_bucket_sales["promo_bucket"] = pd.Categorical(
    promo_bucket_sales["promo_bucket"], categories=BUCKET_LABELS, ordered=True
)
promo_bucket_sales = promo_bucket_sales.sort_values("promo_bucket").reset_index(drop=True)
promo_bucket_sales["promo_bucket"] = promo_bucket_sales["promo_bucket"].astype(str)

page3 = html.Div(style={"backgroundColor": BG_PAGE, "minHeight": "100vh",
                          "fontFamily": FF, "color": C_WHITE}, children=[
    nav(3),
    html.Div(style={"padding": "20px 28px"}, children=[

        # ── ROW 1: Store count per city (bar) + Store type donut ──────────
        html.Div([
            card([
                lbl("Number of stores per city"),
                gr("stores-per-city", "380px"),
            ], width="55%"),
            card([
                lbl("Store type distribution"),
                gr("store-type-donut", "380px"),
            ], width="43%"),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px",
                  "alignItems": "flex-start"}),

        # ── ROW 2: Promo vs No-Promo stacked bar per year ─────────────────
        card([
            lbl("Sales: promoted vs non-promoted items — by year"),
            gr("promo-yearly-bar", "320px"),
        ], extra={"marginBottom": "18px"}),

        # ── ROW 3: Promo lift per family + Promo bucket line ──────────────
        html.Div([
            card([
                lbl("Promotion sales lift by product family (% increase vs no-promo)"),
                gr("promo-lift-bar", "360px"),
            ], width="55%"),
            card([
                lbl("Average sales vs number of items on promotion"),
                gr("promo-bucket-line", "360px"),
            ], width="43%"),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px",
                  "alignItems": "flex-start"}),

        # ── ROW 4: Promo family deep-dive (dropdown) ──────────────────────
        card([
            lbl("Sales comparison: promoted vs non-promoted — by family"),
            dd("dd-p3-family",
               [{"label": f, "value": f} for f in sorted(merged_p3["family"].unique())],
               "GROCERY I"),
            gr("promo-family-detail", "300px"),
        ], extra={"marginBottom": "18px"}),

    ])
])

# ─────────────────────────────────────────────
# PAGE 3 CALLBACKS
# ─────────────────────────────────────────────

# Stores per city – horizontal bar sorted desc
@app.callback(Output("stores-per-city", "figure"), Input("url", "pathname"))
def cb_stores_per_city(_):
    df = stores_per_city.sort_values("num_stores", ascending=True)
    n  = len(df)
    blue_scale = [f"rgba(74,158,255,{0.35 + 0.65*i/max(n-1,1):.2f})" for i in range(n)]
    fig = go.Figure(go.Bar(
        x=df["num_stores"], y=df["city"],
        orientation="h",
        marker=dict(color=blue_scale),
        text=df["num_stores"],
        textposition="outside",
        textfont=dict(color=C_WHITE, size=11),
        hovertemplate="<b>%{y}</b><br>Stores: %{x}<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["margin"] = dict(l=10, r=40, t=10, b=40)
    lay["xaxis"]  = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                         tickfont=dict(size=10), zeroline=False,
                         title=dict(text="Number of Stores", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"]  = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=10), zeroline=False)
    fig.update_layout(**lay)
    return fig

# Store type donut
@app.callback(Output("store-type-donut", "figure"), Input("url", "pathname"))
def cb_store_type_donut(_):
    df = stores_df.groupby("store_type")["store_nbr"].count().reset_index()
    df.columns = ["store_type", "count"]
    fig = go.Figure(go.Pie(
        labels=df["store_type"], values=df["count"],
        hole=0.50,
        marker=dict(colors=PIE_COLORS[:len(df)], line=dict(color=BG_PAGE, width=2)),
        textinfo="label+percent+value",
        textfont=dict(size=11, color=C_WHITE),
        hovertemplate="<b>Type %{label}</b><br>Stores: %{value}<br>%{percent}<extra></extra>",
        direction="clockwise",
    ))
    lay = {**BASE_LAYOUT}
    lay.pop("xaxis", None); lay.pop("yaxis", None)
    lay["margin"]      = dict(l=10, r=10, t=10, b=60)
    lay["legend"]      = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_MUTED, size=10),
                               orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    lay["annotations"] = [dict(text="<b>Types</b>", x=0.5, y=0.5,
                               font=dict(size=14, color=C_CYAN), showarrow=False)]
    fig.update_layout(**lay)
    return fig

# Promo vs no-promo stacked bar per year
@app.callback(Output("promo-yearly-bar", "figure"), Input("url", "pathname"))
def cb_promo_yearly(_):
    promo   = promo_yearly[promo_yearly["promoted"] == "Promoted"]
    nopromo = promo_yearly[promo_yearly["promoted"] == "Not Promoted"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=promo["year"].astype(str), y=promo["sales"],
        name="Promoted",
        marker_color=C_ORANGE,
        hovertemplate="<b>%{x}</b><br>Promoted: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=nopromo["year"].astype(str), y=nopromo["sales"],
        name="Not Promoted",
        marker_color=C_BLUE,
        opacity=0.75,
        hovertemplate="<b>%{x}</b><br>Not Promoted: %{y:,.0f}<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["barmode"] = "group"
    lay["margin"]  = dict(l=55, r=10, t=10, b=40)
    lay["xaxis"]   = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=11),
                          zeroline=False, linecolor=BORDER,
                          title=dict(text="Year", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"]   = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                          tickfont=dict(size=10), zeroline=False,
                          title=dict(text="Total Sales", font=dict(color=C_MUTED, size=11)))
    lay["legend"]  = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_MUTED, size=10),
                          orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
    fig.update_layout(**lay)
    return fig

# Promo lift per family (horizontal bar)
@app.callback(Output("promo-lift-bar", "figure"), Input("url", "pathname"))
def cb_promo_lift(_):
    df = promo_lift_df.sort_values("lift_pct", ascending=True)
    colors = [C_GREEN if v >= 0 else C_RED for v in df["lift_pct"]]
    fig = go.Figure(go.Bar(
        x=df["lift_pct"], y=df["family"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.0f}%" for v in df["lift_pct"]],
        textposition="outside",
        textfont=dict(color=C_MUTED, size=9),
        hovertemplate="<b>%{y}</b><br>Lift: %{x:.1f}%<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["margin"] = dict(l=10, r=55, t=10, b=40)
    lay["xaxis"]  = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                         tickfont=dict(size=10), zeroline=True, zerolinecolor=BORDER,
                         title=dict(text="% Sales Lift vs No-Promo", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"]  = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=9), zeroline=False)
    fig.update_layout(**lay)
    return fig

# Promo bucket: avg sales vs promo units bracket
# Uses Bar+line overlay so fill works on categorical x-axis
@app.callback(Output("promo-bucket-line", "figure"), Input("url", "pathname"))
def cb_promo_bucket(_):
    df = promo_bucket_sales.copy()
    # Use integer positions so fill="tozeroy" works reliably
    x_pos = list(range(len(df)))
    labels = df["promo_bucket"].tolist()
    y_vals = df["avg_sales"].tolist()

    fig = go.Figure()
    # Bar in background
    fig.add_trace(go.Bar(
        x=x_pos, y=y_vals,
        marker=dict(color=[f"rgba(38,198,218,{0.3 + 0.7*i/max(len(x_pos)-1,1):.2f})"
                           for i in range(len(x_pos))]),
        hovertemplate="<b>%{customdata}</b><br>Avg Sales: %{y:,.0f}<extra></extra>",
        customdata=labels,
        showlegend=False,
    ))
    # Line overlay
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_vals,
        mode="lines+markers",
        line=dict(color=C_CYAN, width=2.5),
        marker=dict(size=8, color=C_WHITE, line=dict(color=C_CYAN, width=2)),
        hoverinfo="skip",
        showlegend=False,
    ))
    lay = {**BASE_LAYOUT}
    lay["margin"] = dict(l=55, r=10, t=10, b=55)
    lay["xaxis"]  = dict(
        showgrid=False, color=C_MUTED, tickfont=dict(size=9),
        zeroline=False, linecolor=BORDER,
        tickmode="array",
        tickvals=x_pos,
        ticktext=labels,
        title=dict(text="Items on Promotion (bracket)", font=dict(color=C_MUTED, size=11))
    )
    lay["yaxis"]  = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                         tickfont=dict(size=10), zeroline=False,
                         title=dict(text="Avg Sales", font=dict(color=C_MUTED, size=11)))
    fig.update_layout(**lay)
    return fig

# Promo family detail – side-by-side avg sales with vs without promo
@app.callback(Output("promo-family-detail", "figure"), Input("dd-p3-family", "value"))
def cb_promo_family(family):
    df = merged_p3[merged_p3["family"] == family]
    grp = df.groupby(["year", "promoted"])["sales"].mean().reset_index()
    yes = grp[grp["promoted"] == "Promoted"]
    no  = grp[grp["promoted"] == "Not Promoted"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yes["year"].astype(str), y=yes["sales"],
        name="Promoted", marker_color=C_ORANGE,
        hovertemplate="<b>%{x}</b><br>Avg Promoted Sales: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=no["year"].astype(str), y=no["sales"],
        name="Not Promoted", marker_color=C_BLUE, opacity=0.75,
        hovertemplate="<b>%{x}</b><br>Avg Non-Promo Sales: %{y:,.0f}<extra></extra>",
    ))
    lay = {**BASE_LAYOUT}
    lay["barmode"] = "group"
    lay["margin"]  = dict(l=55, r=10, t=10, b=40)
    lay["xaxis"]   = dict(showgrid=False, color=C_MUTED, tickfont=dict(size=10),
                          zeroline=False, linecolor=BORDER,
                          title=dict(text="Year", font=dict(color=C_MUTED, size=11)))
    lay["yaxis"]   = dict(showgrid=True, gridcolor=BORDER, color=C_MUTED,
                          tickfont=dict(size=10), zeroline=False,
                          title=dict(text="Avg Sales", font=dict(color=C_MUTED, size=11)))
    lay["legend"]  = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_MUTED, size=10),
                          orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    fig.update_layout(**lay)
    return fig

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

