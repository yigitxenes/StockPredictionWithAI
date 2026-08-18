import os
import sys
import markdown as md_lib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from dotenv import load_dotenv


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from agent.write_report import generate_report, load_performance_summary
from agent.live_data_fetch import get_live_technical_data
from ticker_panel import render_ticker_panel, render_market_board, inject_ticker_css

load_dotenv()

st.set_page_config(page_title="Borsa Asistanı | Hisse Analiz Raporu", page_icon="◐", layout="wide")

# ============================================================
# TASARIM SISTEMI
# ============================================================
COLORS = {
    "bg": "#0D1117",
    "bg_card": "#161B22",
    "bg_card_alt": "#1C2333",
    "border": "#2A3142",
    "text": "#E6E8EB",
    "text_muted": "#8B93A1",
    "teal": "#5EC9A8",     # yukselis
    "rose": "#E2725B",     # dusus
    "amber": "#E8A33D",    # belirsizlik -- imza rengi
    "blue": "#6C8EEF",
}


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {COLORS["bg"]};
        color: {COLORS["text"]};
    }}

    .stApp {{
        background-color: {COLORS["bg"]};
    }}

    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }}

    .hero-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }}
    .hero-sub {{
        color: {COLORS["text_muted"]};
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }}

    .reliability-banner {{
        background: linear-gradient(90deg, rgba(232,163,61,0.14), rgba(232,163,61,0.04));
        border: 1px solid rgba(232,163,61,0.4);
        border-left: 4px solid {COLORS["amber"]};
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 1.4rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }}
    .reliability-banner b {{ color: {COLORS["amber"]}; }}

    .metric-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 16px 18px;
        height: 100%;
    }}
    .metric-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {COLORS["text_muted"]};
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem;
        font-weight: 600;
    }}
    .metric-caption {{
        font-size: 0.78rem;
        color: {COLORS["text_muted"]};
        margin-top: 4px;
    }}

    .report-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 22px 26px;
        line-height: 1.65;
        font-size: 0.95rem;
    }}
    .report-card p {{ margin-bottom: 0.9em; }}
    .report-card strong {{ color: {COLORS["amber"]}; font-weight: 600; }}
    .report-card h1, .report-card h2, .report-card h3 {{
        font-family: 'Space Grotesk', sans-serif;
        margin-top: 0.6em; margin-bottom: 0.4em;
        color: {COLORS["text"]};
    }}

    .section-label {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {COLORS["text_muted"]};
        margin: 1.6rem 0 0.6rem 0;
        border-bottom: 1px solid {COLORS["border"]};
        padding-bottom: 6px;
    }}

    div[data-testid="stSelectbox"] label, div[data-testid="stButton"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stButton button {{
        background-color: {COLORS["amber"]};
        color: #1A1206;
        font-weight: 600;
        border: none;
        border-radius: 6px;
    }}
    .stButton button:hover {{
        background-color: #F0B255;
        color: #1A1206;
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# GRAFIKLER
# ============================================================

def render_price_chart(df, ticker):
    """Mum grafik + hacim + RSI alt paneli, koyu tema."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03,
        subplot_titles=(None, None, None),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=COLORS["teal"], decreasing_line_color=COLORS["rose"],
        name="Fiyat",
    ), row=1, col=1)

    if "BBU_20_2.0_2.0" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BBU_20_2.0_2.0"], line=dict(color=COLORS["border"], width=1),
                                  name="BB Ust", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BBL_20_2.0_2.0"], line=dict(color=COLORS["border"], width=1),
                                  fill="tonexty", fillcolor="rgba(108,142,239,0.06)",
                                  name="BB Alt", showlegend=False), row=1, col=1)

    colors_vol = [COLORS["teal"] if c >= o else COLORS["rose"] for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors_vol, name="Hacim",
                          showlegend=False), row=2, col=1)

    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], line=dict(color=COLORS["amber"], width=1.5),
                                  name="RSI", showlegend=False), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=COLORS["rose"], line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=COLORS["teal"], line_width=1, row=3, col=1)

    fig.update_layout(
        height=520, plot_bgcolor=COLORS["bg_card"], paper_bgcolor=COLORS["bg_card"],
        font=dict(family="Inter", color=COLORS["text_muted"], size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=COLORS["border"], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS["border"], showgrid=True)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    return fig


def render_confidence_gauge(auc):
    """IMZA OGE: AUC'yi 0.45-0.60 arasina YAKINLASTIRILMIS bir ibre olarak
    gosterir, ortada 0.48-0.52 'sis bandi' ile -- ibrenin bu sisin disina
    neredeyse hic cikamadigini GORSEL olarak vurgulamak icin bilincli bir
    tasarim karari. Bu, projenin ana bulgusunu (sinyal sans seviyesinden
    ayirt edilemiyor) literal olarak temsil ediyor."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=auc,
        number={"font": {"family": "JetBrains Mono", "size": 34, "color": COLORS["text"]}, "valueformat": ".4f"},
        gauge={
            "axis": {"range": [0.45, 0.60], "tickfont": {"color": COLORS["text_muted"], "size": 10}},
            "bar": {"color": COLORS["amber"], "thickness": 0.25},
            "bgcolor": COLORS["bg_card"],
            "borderwidth": 0,
            "steps": [
                {"range": [0.45, 0.48], "color": "rgba(139,147,161,0.15)"},
                {"range": [0.48, 0.52], "color": "rgba(232,163,61,0.22)"},  # sis bandi
                {"range": [0.52, 0.60], "color": "rgba(139,147,161,0.15)"},
            ],
            "threshold": {
                "line": {"color": COLORS["text_muted"], "width": 2},
                "thickness": 0.9, "value": 0.50,
            },
        },
    ))
    fig.update_layout(
        height=210, paper_bgcolor=COLORS["bg_card"],
        margin=dict(l=20, r=20, t=10, b=10),
        font=dict(color=COLORS["text_muted"]),
    )
    return fig


def render_sentiment_bar(score, confidence):
    fig = go.Figure(go.Bar(
        x=[score], y=["Sentiment"], orientation="h",
        marker_color=COLORS["teal"] if score >= 0 else COLORS["rose"],
        width=0.5,
    ))
    fig.add_vline(x=0, line_color=COLORS["border"], line_width=1)
    fig.update_layout(
        height=90, xaxis=dict(range=[-1, 1], gridcolor=COLORS["border"], zeroline=False,
                               tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_muted"])),
        yaxis=dict(visible=False),
        plot_bgcolor=COLORS["bg_card"], paper_bgcolor=COLORS["bg_card"],
        margin=dict(l=10, r=10, t=10, b=25),
        font=dict(color=COLORS["text_muted"]),
    )
    return fig


# ============================================================
# BILESENLER
# ============================================================

def metric_card(label, value, caption="", color=None):
    color = color or COLORS["text"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


def render_reliability_banner(perf):
    auc = perf["direction"]["value"]
    st.markdown(f"""
    <div class="reliability-banner">
        ⚠️ <b>Model Güvenilirlik Uyarısı</b> — Yön tahmini modelinin tarihsel AUC değeri
        <b>{auc:.4f}</b>, şans seviyesine (0.50) çok yakındır. Aşağıdaki tahminler
        <b>yatırım tavsiyesi değildir</b> ve güvenilir bir sinyal olarak yorumlanmamalıdır.
        Bu sayfadaki gösterge, bu belirsizliği bilinçli olarak görünür kılacak şekilde tasarlanmıştır.
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


@st.cache_resource
def get_config():
    return load_config()


def main():
    inject_css()
    inject_ticker_css()
    cfg = get_config()
    perf = load_performance_summary()

    st.markdown('<div class="hero-title">◐ Borsa Asistanı</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Şeffaf hisse analiz aracı — kesin tahmin değil, ölçülmüş belirsizlikle raporlama.</div>',
        unsafe_allow_html=True,
    )
    render_ticker_panel(cfg)
    render_reliability_banner(perf)

    tickers = cfg["data"]["tickers"]
    company_names = cfg["data"]["company_names"]
    ticker_labels = [f"{t} — {company_names.get(t, t)}" for t in tickers]

    col_select, col_button = st.columns([3, 1])
    with col_select:
        selected_label = st.selectbox("Hisse seçin", ticker_labels, label_visibility="collapsed")
    with col_button:
        generate = st.button("Rapor Oluştur", use_container_width=True)

    selected_ticker = tickers[ticker_labels.index(selected_label)]

    if generate:
        with st.spinner(f"{selected_ticker} için veri ve rapor hazırlanıyor..."):
            try:
                client = get_gemini_client()
                price_df = get_live_technical_data(selected_ticker, cfg)
                result = generate_report(selected_ticker, cfg, client)
            except Exception as e:
                st.error(f"Hata: {type(e).__name__}: {e}")
                return

        st.markdown(f'<div class="section-label">{selected_ticker} — {result["as_of_date"]} itibarıyla (gerçek zamanlı değil, son kapanış)</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            up_prob = result["direction_probability_up"]
            metric_card("Yükseliş Olasılığı", f"{up_prob:.1%}",
                        "Şans seviyesine yakın model", COLORS["amber"])
        with m2:
            metric_card("5G Volatilite Tahmini", f"{result['volatility_forecast']:.4f}",
                        f"Naive: {perf['volatility']['naive_baseline_rmse']:.4f}")
        with m3:
            sentiment = result["sentiment"]
            s_color = COLORS["teal"] if sentiment["overall_score"] >= 0 else COLORS["rose"]
            metric_card("Sentiment Skoru", f"{sentiment['overall_score']:+.2f}",
                        f"{sentiment.get('news_count', 0)} haber değerlendirildi", s_color)

        st.markdown('<div class="section-label">Fiyat & Teknik Görünüm</div>', unsafe_allow_html=True)
        st.plotly_chart(render_price_chart(price_df, selected_ticker), use_container_width=True)

        col_gauge, col_sent = st.columns([1, 1])
        with col_gauge:
            st.markdown('<div class="section-label">Model Güven Göstergesi (AUC)</div>', unsafe_allow_html=True)
            st.plotly_chart(render_confidence_gauge(perf["direction"]["value"]), use_container_width=True)
            st.caption("Amber bant = istatistiksel olarak şanstan ayırt edilemeyen bölge")
        with col_sent:
            st.markdown('<div class="section-label">Sentiment Yönü</div>', unsafe_allow_html=True)
            st.plotly_chart(render_sentiment_bar(sentiment["overall_score"], sentiment["overall_confidence"]),
                             use_container_width=True)
            st.caption(f"Güven: {sentiment['overall_confidence']:.2f}")

        st.markdown('<div class="section-label">Agent Raporu</div>', unsafe_allow_html=True)
        report_html = md_lib.markdown(result["report_text"])
        st.markdown(f'<div class="report-card">{report_html}</div>', unsafe_allow_html=True)

        with st.expander("Sentiment gerekçesi (ham)"):
            st.write(sentiment["reasoning"])
    else:
        render_market_board(cfg)


if __name__ == "__main__":
    main()