"""
Ticker paneli — Sinyal Sisi
Kullanim: app/app.py icine

    from app.ticker_panel import render_ticker_panel, render_market_board

    inject_css()          # mevcut CSS
    inject_ticker_css()   # bu dosyadan
    ...
    render_ticker_panel(cfg)     # hero'dan hemen sonra, banner'dan once
    ...
    else:
        render_market_board(cfg) # st.info yerine (bos ekrani dolduran kisim)
"""

import streamlit as st
import yfinance as yf

COLORS = {
    "bg": "#0D1117",
    "bg_card": "#161B22",
    "bg_card_alt": "#1C2333",
    "border": "#2A3142",
    "corner": "#3A4356",
    "text": "#E6E8EB",
    "text_muted": "#8B93A1",
    "teal": "#5EC9A8",
    "rose": "#E2725B",
    "amber": "#E8A33D",
    "blue": "#6C8EEF",
}


# ============================================================
# VERI
# ============================================================

@st.cache_data(ttl=900)
def get_daily_changes(tickers, company_names):
    """Her ticker icin son kapanis + onceki kapanisa gore degisim.
    Tek yf.download cagrisi; 7 gunluk pencere tatil/haftasonu icin yeterli."""
    df = yf.download(list(tickers), period="7d", interval="1d",
                     auto_adjust=True, progress=False, group_by="column")

    close = df["Close"] if "Close" in df.columns else df
    rows, as_of = [], None

    for t in tickers:
        try:
            s = close[t].dropna() if hasattr(close, "columns") else close.dropna()
        except KeyError:
            continue
        if len(s) < 2:
            continue

        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        as_of = s.index[-1]
        pct = (last - prev) / prev * 100.0

        rows.append({
            "ticker": t,
            "name": company_names.get(t, t),
            "close": last,
            "prev": prev,
            "abs": last - prev,
            "pct": pct,
            "up": pct >= 0,
        })

    return rows, as_of


def _decorate(rows):
    """Renk, ok ve bar genisligi — sunum alanlari."""
    max_mag = max((abs(r["pct"]) for r in rows), default=1.0) or 1.0
    for r in rows:
        r["color"] = COLORS["teal"] if r["up"] else COLORS["rose"]
        r["tint"] = "rgba(94,201,168,0.12)" if r["up"] else "rgba(226,114,91,0.12)"
        r["arrow"] = "▲" if r["up"] else "▼"
        r["pct_str"] = f"{r['pct']:+.2f}%"
        r["abs_str"] = f"{r['abs']:+.2f}"
        r["bar"] = max(6.0, abs(r["pct"]) / max_mag * 100.0)
    return rows


# ============================================================
# CSS  (mevcut inject_css'e ek — kose isaretleri ve hover)
# ============================================================

def inject_ticker_css():
    st.markdown(f"""
    <style>
    .bp {{ position: relative; border: 1px solid {COLORS["border"]}; border-radius: 0; }}
    .bp > i.corner {{
        position: absolute; width: 11px; height: 11px; color: {COLORS["corner"]};
    }}
    .bp > i.corner::before, .bp > i.corner::after {{
        content: ""; position: absolute; background: currentColor;
    }}
    .bp > i.corner::before {{ left: 5px; top: 0; width: 1px; height: 100%; }}
    .bp > i.corner::after  {{ top: 5px; left: 0; width: 100%; height: 1px; }}
    .bp > i.tl {{ top: -6px; left: -6px; }}
    .bp > i.tr {{ top: -6px; right: -6px; }}
    .bp > i.bl {{ bottom: -6px; left: -6px; }}
    .bp > i.br {{ bottom: -6px; right: -6px; }}

    .ticker-strip {{ background: {COLORS["bg_card"]}; margin-bottom: 6px; }}
    .ticker-strip-head {{
        display: flex; align-items: center; gap: 10px;
        padding: 9px 14px; border-bottom: 1px solid {COLORS["border"]};
    }}
    .ticker-strip-head .label {{
        font-family: 'Space Grotesk', sans-serif; font-weight: 500;
        font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase;
        color: {COLORS["text_muted"]};
    }}
    .ticker-strip-head .tally {{
        margin-left: auto; display: flex; gap: 14px;
        font-family: 'JetBrains Mono', monospace; font-size: 10px;
        color: {COLORS["text_muted"]};
    }}
    .ticker-strip-head .tally span {{ display: flex; align-items: center; gap: 5px; }}
    .ticker-strip-head .tally i {{ width: 7px; height: 7px; display: block; font-style: normal; }}

    .ticker-row {{ display: grid; overflow-x: auto; }}
    .ticker-cell {{
        padding: 11px 12px 12px; border-right: 1px solid {COLORS["border"]};
        display: flex; flex-direction: column; gap: 3px; min-width: 0;
        transition: background 120ms;
    }}
    .ticker-cell:last-child {{ border-right: none; }}
    .ticker-cell:hover {{ background: {COLORS["bg_card_alt"]}; }}
    .ticker-cell .sym {{
        font-family: 'Space Grotesk', sans-serif; font-weight: 500;
        font-size: 12px; letter-spacing: 0.04em; color: {COLORS["text_muted"]};
    }}
    .ticker-cell .px {{
        font-family: 'JetBrains Mono', monospace; font-weight: 600;
        font-size: 14px; color: {COLORS["text"]};
    }}
    .ticker-cell .chg {{ font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }}

    .board {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .board-cell {{
        background: {COLORS["bg_card"]}; padding: 14px 16px 15px;
        display: flex; flex-direction: column; gap: 9px; transition: background 120ms;
    }}
    .board-cell:hover {{ background: {COLORS["bg_card_alt"]}; }}
    .board-cell .head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }}
    .board-cell .sym {{
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 16px; letter-spacing: 0.02em; color: {COLORS["text"]};
    }}
    .board-cell .co {{
        font-size: 11px; color: {COLORS["text_muted"]}; text-align: right;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .board-cell .mid {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 10px; }}
    .board-cell .px {{
        font-family: 'JetBrains Mono', monospace; font-weight: 600;
        font-size: 1.55rem; line-height: 1.05; color: {COLORS["text"]};
    }}
    .board-cell .badge {{
        font-family: 'JetBrains Mono', monospace; font-size: 12.5px; padding: 2px 7px;
    }}
    .board-cell .track {{ height: 3px; background: {COLORS["border"]}; position: relative; }}
    .board-cell .track div {{ position: absolute; top: 0; bottom: 0; left: 0; }}
    .board-cell .foot {{
        display: flex; justify-content: space-between;
        font-family: 'JetBrains Mono', monospace; font-size: 10px;
        color: {COLORS["text_muted"]};
    }}
    .board-note {{
        border: 1px dashed {COLORS["border"]}; padding: 14px 16px;
        display: flex; flex-direction: column; justify-content: center; gap: 6px;
    }}
    .board-note .label {{
        font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 11px;
        letter-spacing: 0.1em; text-transform: uppercase; color: {COLORS["text_muted"]};
    }}
    .board-note span {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; }}

    .board-info {{
        margin-top: 1.6rem; background: rgba(108,142,239,0.10);
        border: 1px solid rgba(108,142,239,0.35); border-radius: 6px;
        padding: 13px 18px; font-size: 0.9rem; color: #C7CDD6;
    }}
    </style>
    """, unsafe_allow_html=True)


_CORNERS = ('<i class="corner tl"></i><i class="corner tr"></i>'
            '<i class="corner bl"></i><i class="corner br"></i>')


# ============================================================
# PANELLER
# ============================================================

def render_ticker_panel(cfg):
    """Hero'nun altindaki tek satirlik takip listesi seridi."""
    tickers = cfg["data"]["tickers"]
    names = cfg["data"]["company_names"]

    try:
        rows, as_of = get_daily_changes(tickers, names)
    except Exception as e:
        st.caption(f"Fiyat verisi alinamadi ({type(e).__name__})")
        return

    if not rows:
        return

    rows = _decorate(rows)
    up = sum(1 for r in rows if r["up"])
    down = len(rows) - up

    cells = "".join(
        f'<div class="ticker-cell">'
        f'<span class="sym">{r["ticker"]}</span>'
        f'<span class="px">${r["close"]:.2f}</span>'
        f'<span class="chg" style="color:{r["color"]}">{r["pct_str"]}</span>'
        f'</div>'
        for r in rows
    )

    st.markdown(f"""
    <div class="bp ticker-strip">{_CORNERS}
        <div class="ticker-strip-head">
            <span class="label">Takip listesi — önceki kapanışa göre</span>
            <span class="tally">
                <span><i style="background:{COLORS["teal"]}"></i>{up} yükselen</span>
                <span><i style="background:{COLORS["rose"]}"></i>{down} düşen</span>
            </span>
        </div>
        <div class="ticker-row" style="grid-template-columns:repeat({len(rows)},minmax(0,1fr))">
            {cells}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_market_board(cfg):
    """Rapor olusturulmadan onceki bos ekrani dolduran piyasa gorunumu."""
    tickers = cfg["data"]["tickers"]
    names = cfg["data"]["company_names"]

    try:
        rows, as_of = get_daily_changes(tickers, names)
    except Exception as e:
        st.info("Bir hisse seçip **Rapor Oluştur**'a basarak analizi başlatabilirsin.")
        st.caption(f"Piyasa görünümü yüklenemedi ({type(e).__name__})")
        return

    if not rows:
        st.info("Bir hisse seçip **Rapor Oluştur**'a basarak analizi başlatabilirsin.")
        return

    rows = _decorate(rows)
    by_pct = sorted(rows, key=lambda r: r["pct"], reverse=True)
    avg = sum(r["pct"] for r in rows) / len(rows)
    date_str = as_of.strftime("%d.%m.%Y") if as_of is not None else "son kapanış"

    st.markdown(f"""
    <div class="section-label" style="display:flex;justify-content:space-between;gap:16px">
        <span>Piyasa Görünümü — {date_str} kapanışı</span>
        <span style="font-family:'JetBrains Mono',monospace;text-transform:none;letter-spacing:0">
            Ort. {avg:+.2f}%
        </span>
    </div>
    """, unsafe_allow_html=True)

    cards = "".join(
        f'<div class="bp board-cell">{_CORNERS}'
        f'  <div class="head"><span class="sym">{r["ticker"]}</span>'
        f'       <span class="co">{r["name"]}</span></div>'
        f'  <div class="mid"><span class="px">${r["close"]:.2f}</span>'
        f'       <span class="badge" style="color:{r["color"]};background:{r["tint"]}">'
        f'{r["arrow"]} {r["pct_str"]}</span></div>'
        f'  <div style="display:flex;flex-direction:column;gap:5px">'
        f'    <div class="track"><div style="width:{r["bar"]:.0f}%;background:{r["color"]}"></div></div>'
        f'    <div class="foot"><span>önceki ${r["prev"]:.2f}</span>'
        f'         <span>{r["abs_str"]} $</span></div>'
        f'  </div>'
        f'</div>'
        for r in rows
    )

    note = (
        f'<div class="bp board-note">'
        f'  <span class="label">Günün ucu</span>'
        f'  <span style="color:{COLORS["teal"]}">↑ {by_pct[0]["ticker"]} {by_pct[0]["pct_str"]}</span>'
        f'  <span style="color:{COLORS["rose"]}">↓ {by_pct[-1]["ticker"]} {by_pct[-1]["pct_str"]}</span>'
        f'</div>'
    )

    st.markdown(f'<div class="board">{cards}{note}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="board-info">Bir hisse seçip <b style="color:#E6E8EB">Rapor Oluştur</b>\'a '
        'basarak analizi başlatabilirsin. Yukarıdaki fiyatlar gerçek zamanlı değildir — '
        'son kapanış verisidir.</div>',
        unsafe_allow_html=True,
    )
