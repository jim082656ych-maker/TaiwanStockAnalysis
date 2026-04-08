import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import requests
import re

# --- 頁面配置 (超寬螢幕模式) ---
st.set_page_config(page_title="專業級 AI 台股分析儀表板", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 40px; }
    .stTabs [data-baseweb="tab"] { height: 70px; font-weight: 900; font-size: 20px; color: #1a237e; border-bottom: 4px solid transparent; }
    .stTabs [aria-selected="true"] { border-bottom: 4px solid #1a237e !important; }
    .strategy-box { background-color: #ffffff; padding: 30px; border-radius: 15px; border-left: 12px solid #d32f2f; box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom: 35px; line-height: 1.8; }
    .strategy-box h3 { color: #d32f2f; margin-top: 0; font-size: 30px; border-bottom: 3px solid #f0f2f6; padding-bottom: 15px; margin-bottom: 20px; }
    .strategy-box b { color: #1a237e; font-size: 1.15em; }
    .strategy-box i { color: #388e3c; font-style: normal; font-weight: bold; }
    .metric-container { background-color: #ffffff; padding: 25px; border-radius: 12px; border: 2px solid #1a237e; margin-top: 25px; }
    .stock-title-area { background-color: #1a237e; color: white; padding: 25px; border-radius: 10px; margin-bottom: 25px; text-align: center; }
    .stock-title-main { font-size: 56px; font-weight: 900; margin: 0; letter-spacing: 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 側邊欄 ---
st.sidebar.header("⚙️ 核心設定")
stock_id = st.sidebar.text_input("輸入台股代號", value="2330")
symbol = f"{stock_id}.TW" if "." not in stock_id else stock_id

# --- 抓取中文名稱 ---
def get_chinese_name(sid):
    common_stocks = {"2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電", "2303": "聯電", "2881": "富邦金", "2882": "國泰金", "2412": "中華電", "2382": "廣達"}
    pure_id = sid.split('.')[0]
    if pure_id in common_stocks: return common_stocks[pure_id]
    try:
        url = f"https://tw.stock.yahoo.com/quote/{pure_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=3)
        if resp.status_code == 200:
            match = re.search(r'<title>(.*?)</title>', resp.text)
            if match:
                name = match.group(1).split('-')[0].replace(pure_id, "").strip()
                if name: return name
    except: pass
    return "台股個股"

# --- 核心指標計算 ---
def get_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['std20'] * 2)
    df['Lower'] = df['MA20'] - (df['std20'] * 2)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    l9 = df['Low'].rolling(window=9).min()
    h9 = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - l9) / (h9 - l9 + 1e-10) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA'])
    df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
    df['BIAS60'] = (df['Close'] - df['MA60']) / df['MA60'] * 100
    return df

# --- 核心數據抓取配置 (偽裝成真人在使用 Chrome) ---
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})

@st.cache_data
def load_data(symbol):
    s = yf.Ticker(symbol, session=session)
    df = s.history(period="3y")
    return get_indicators(df), s.info

def show_latest_metrics(latest, prev, stock_full_name):
    with st.container():
        st.markdown(f'<div class="metric-container"><h3>📊 {stock_full_name} 即時報價</h3>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("最新收盤", f"{latest['Close']:.2f}", f"{latest['Close']-prev['Close']:.2f}")
        m2.metric("今日開盤", f"{latest['Open']:.2f}")
        m3.metric("今日最高", f"{latest['High']:.2f}")
        m4.metric("今日最低", f"{latest['Low']:.2f}")
        m5.metric("今日成交量", f"{int(latest['Volume']):,}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- 主程式執行 ---
try:
    with st.spinner('載入操盤大數據中...'):
        df, info = load_data(symbol)
        
    if df.empty:
        st.error("代號錯誤，請檢查。")
    else:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        stock_name = get_chinese_name(symbol)
        stock_full_name = f"{stock_id} {stock_name}"
        
        st.markdown(f'<div class="stock-title-area"><h1 class="stock-title-main">🏛️ {stock_full_name}</h1></div>', unsafe_allow_html=True)
        
        t1, t2, t3, t4, t5 = st.tabs([
            "📊 1. MA均線與布林突破", 
            "📈 2. KD與MACD 轉折決策", 
            "📉 3. RSI/BIAS 逆勢攻略", 
            "🤖 4. AI 策略預測", 
            "📋 5. 數據統計中心"
        ])

        default_range = [df.index[-30], df.index[-1]]
        def setup_chart(fig, height=1500):
            fig.update_xaxes(rangeslider_visible=False, rangeselector=dict(buttons=list([dict(count=1, label="1m", step="month", stepmode="backward"), dict(count=6, label="6m", step="month", stepmode="backward"), dict(count=1, label="1y", step="year", stepmode="backward"), dict(step="all", label="All")]), font=dict(size=14, weight="bold"), y=1.02), range=default_range)
            fig.update_layout(height=height, template="plotly_white", hovermode="x unified")

        def add_k_vol(fig, display_df, row_k=1, row_v=2):
            fig.add_trace(go.Candlestick(x=display_df.index, open=display_df['Open'], high=display_df['High'], low=display_df['Low'], close=display_df['Close'], name='K線', increasing_line_color='#FF0000', decreasing_line_color='#006400'), row=row_k, col=1)
            colors = ['#FF0000' if display_df['Close'].iloc[i] >= display_df['Open'].iloc[i] else '#006400' for i in range(len(display_df))]
            fig.add_trace(go.Bar(x=display_df.index, y=display_df['Volume'], name='實體成交量', marker_color=colors, opacity=1.0), row=row_v, col=1)

        # --- 分頁 1: MA 均線與布林通道 (詳細說明) ---
        with t1:
            st.markdown(f"""
            <div class="strategy-box">
                <h3>📖 {stock_full_name} MA 均線與布林通道決策指南</h3>
                <ul>
                    <li><b>均線交叉策略</b>：
                        <ul>
                            <li><b>🚀 黃金交叉</b>：5MA 向上穿過 20MA，代表短期動能轉強，為<b>買進點</b>。</li>
                            <li><b>📉 死亡交叉</b>：5MA 向下跌破 20MA，趨勢轉空，為<b>減碼停損點</b>。</li>
                        </ul>
                    </li>
                    <li><b>布林通道決策</b>：
                        <ul>
                            <li><b>⚡ 擠壓噴發</b>：通道寬度縮小後，股價<b>帶量突破上軌</b>，通常是波段噴出的開端。</li>
                            <li><b>🛡️ 下軌支撐</b>：股價觸及布林下軌且與月線負乖離過大，常具備超跌反彈契機。</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.65, 0.35])
            add_k_vol(fig1, df); fig1.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='5MA', line=dict(color='#FF69B4', width=4)), row=1, col=1); fig1.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20MA', line=dict(color='#FFA500', width=4)), row=1, col=1); fig1.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='60MA', line=dict(color='#0000FF', width=4)), row=1, col=1)
            # 布林線 trace
            fig1.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='布林上', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='布林下', line=dict(color='gray', dash='dash'), fill='tonexty'), row=1, col=1)
            setup_chart(fig1); st.plotly_chart(fig1, width="stretch"); show_latest_metrics(latest, prev, stock_full_name)

        # --- 分頁 2: KD 與 MACD (詳細交叉說明) ---
        with t2:
            st.markdown(f"""
            <div class="strategy-box">
                <h3>📖 {stock_full_name} KD 與 MACD 轉折決策詳解</h3>
                <ul>
                    <li><b>KD 指標決策</b>：
                        <ul>
                            <li><b>🚀 黃金交叉</b>：當 K 線(快) 向上穿過 D 線(慢)，且發生在 <b>20 以下</b>，是極高勝率的<b>抄底起漲點</b>。</li>
                            <li><b>📉 死亡交叉</b>：當 K 線(快) 向下穿過 D 線(慢)，且發生在 <b>80 以上</b>，代表過熱警訊，應防範見頂。</li>
                        </ul>
                    </li>
                    <li><b>MACD 趨勢確認</b>：
                        <ul>
                            <li><b>紅柱增長</b>：代表多頭動能轉強。</li>
                            <li><b>零軸金叉</b>：DIF 由下往上穿過 DEA 且位於零軸之上，是強勢股主升段的<b>進場訊號</b>。</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.4, 0.2, 0.2, 0.2])
            add_k_vol(fig2, df, row_k=1, row_v=1)
            fig2.add_trace(go.Scatter(x=df.index, y=df['K'], name='KD-K', line=dict(color='blue', width=4)), row=2, col=1); fig2.add_trace(go.Scatter(x=df.index, y=df['D'], name='KD-D', line=dict(color='orange', width=4)), row=2, col=1)
            hist_colors = ['#FF0000' if val >= 0 else '#006400' for val in df['MACD_Hist']]
            fig2.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD柱', marker_color=hist_colors), row=3, col=1)
            fig2.add_trace(go.Scatter(x=df.index, y=df['DIF'], name='MACD DIF', line=dict(color='black', width=3)), row=4, col=1); fig2.add_trace(go.Scatter(x=df.index, y=df['DEA'], name='MACD DEA', line=dict(color='orange', width=3)), row=4, col=1)
            setup_chart(fig2); st.plotly_chart(fig2, width="stretch"); show_latest_metrics(latest, prev, stock_full_name)

        # --- 分頁 3 ---
        with t3:
            st.markdown(f"""<div class="strategy-box"><h3>📖 {stock_full_name} RSI 指標決策與 BIAS 交叉意義</h3><ul><li><b>RSI 底部背離</b>：抄底的最強先行指標。</li><li><b>BIAS 5/60 交叉</b>：快線(5)向上穿過慢線(60)代表短線反彈動能確立。</li></ul></div>""", unsafe_allow_html=True)
            fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.25, 0.25])
            add_k_vol(fig3, df, row_k=1, row_v=1)
            fig3.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=4)), row=2, col=1)
            fig3.add_trace(go.Scatter(x=df.index, y=df['BIAS5'], name='BIAS 5', line=dict(color='red', width=2)), row=3, col=1)
            fig3.add_trace(go.Scatter(x=df.index, y=df['BIAS60'], name='BIAS 60', line=dict(color='blue', width=2)), row=3, col=1)
            setup_chart(fig3); st.plotly_chart(fig3, width="stretch"); show_latest_metrics(latest, prev, stock_full_name)
            st.write("---"); c1, c2, c3, c4 = st.columns(4); c1.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}"); c2.metric("EPS", f"{info.get('trailingEps', 'N/A')}"); c3.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%"); c4.metric("殖利率", f"{info.get('dividendYield', 0)*100:.2f}%")

        # --- Tab 4 & 5 ---
        with t4:
            st.subheader(f"🤖 {stock_full_name} AI 預測"); score = sum([1 if latest['K']>latest['D'] else 0, 1 if latest['Close']>latest['MA20'] else 0])
            st.success(f"### AI 綜合評價：{'🚀 積極買入' if score==2 else '⚖️ 中性觀望'} Supreme"); years = 3; days = 252 * years; last_p = latest['Close']; total_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
            ann_ret = (1 + total_ret) ** (1/3) - 1; ann_vol = df['Close'].pct_change().std() * np.sqrt(252); future_idx = [df.index[-1] + timedelta(days=i) for i in range(1, days + 1)]; path = last_p * np.exp((ann_ret - (ann_vol**2)/2) * (np.arange(1, days+1)/252))
            fig_p = go.Figure(); fig_p.add_trace(go.Scatter(x=df.index[-200:], y=df['Close'][-200:], name='歷史記錄')); fig_p.add_trace(go.Scatter(x=future_idx, y=path, name='AI 預測路徑', line=dict(dash='dash', color='red'))); fig_p.update_layout(height=800, title=f"{stock_full_name} 未來預測", template="plotly_white"); st.plotly_chart(fig_p, width="stretch")

        with t5:
            st.subheader(f"🕒 {stock_full_name} 數據中心"); st.table(pd.DataFrame({"指標": ["最新收盤", "最高", "最低", "成交量"], "數值": [f"{latest['Close']:.2f}", f"{latest['High']:.2f}", f"{latest['Low']:.2f}", f"{int(latest['Volume']):,}"]}))
            st.divider(); st.dataframe(df.sort_index(ascending=False), width="stretch")

except Exception as e:
    st.error(f"系統發生問題: {e}")
