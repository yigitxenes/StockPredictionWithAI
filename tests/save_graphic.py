# save_price_chart.py -- proje kok dizininde calistir
import sys
sys.path.append(".")
from config.loader import load_config
from agent.live_data_fetch import get_live_technical_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots

cfg = load_config()
ticker = "NVDA"  # istedigin ticker'i sec
df = get_live_technical_data(ticker, cfg)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                              low=df["Low"], close=df["Close"],
                              increasing_line_color="#2E9E83", decreasing_line_color="#C0524A"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], line=dict(color="#E8A33D")), row=2, col=1)
fig.update_layout(height=600, width=1000, template="plotly_white",
                   title=f"{ticker} — Fiyat ve RSI", showlegend=False, xaxis_rangeslider_visible=False)
fig.write_image("price_chart.png", scale=2)
print("Kaydedildi: price_chart.png")