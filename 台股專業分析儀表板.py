import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# --- 頁面配置 ---
st.set_page_config(page_title="終極操盤手專業分析儀表板", layout="wide")

# 取得桌面路徑
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 40px; }
    .stTabs [data-baseweb="tab"] { height: 70px; font-weight: 900; font-size: 20px; color: #1a237e; border-bottom: 4px solid transparent; }
    .stTabs [aria-selected="true"] { border-bottom: 4px solid #1a237e !important; }
    .strategy-box { background-color: #ffffff; padding: 30px; border-radius: 15px; border-left: 15px solid #d32f2f; box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom: 35px; line-height: 1.8; }
    .strategy-box h3 { color: #d32f2f; margin-top: 0; font-size: 30px; border-bottom: 3px solid #f0f2f6; padding-bottom: 15px; margin-bottom: 20px; }
    .strategy-box b { color: #1a237e; font-size: 1.15em; }
    .strategy-box li { margin-bottom: 10px; font-size: 18px; }
    .metric-container { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 2px solid #e0e0e0; margin-top: 25px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ 終極操盤手 AI 台股全方位分析儀表板")

# --- 側邊欄 ---
st.sidebar.header("⚙️ 核心設定")
stock_id = st.sidebar.text_input("輸入台股代號", value="2330")
symbol = f"{stock_id}.TW" if "." not in stock_id else stock_id

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
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
    df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
    df['BIAS60'] = (df['Close'] - df['MA60']) / df['MA60'] * 100
    return df

@st.cache_data
def load_data(symbol):
    s = yf.Ticker(symbol)
    df = s.history(period="3y")
    return get_indicators(df), s.info

def show_latest_metrics(latest, prev):
    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("最新收盤", f"{latest['Close']:.2f}", f"{latest['Close']-prev['Close']:.2f}")
        m2.metric("今日開盤", f"{latest['Open']:.2f}")
        m3.metric("今日最高", f"{latest['High']:.2f}")
        m4.metric("今日最低", f"{latest['Low']:.2f}")
        m5.metric("今日成交量", f"{int(latest['Volume']):,}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- 主程式執行 ---
try:
    with st.spinner('同步全球市場數據中...'):
        df, info = load_data(symbol)
        
    if df.empty:
        st.error("代號錯誤，請檢查。")
    else:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        t1, t2, t3, t4, t5 = st.tabs([
            "📊 1. 均線/布林/實體量 (突破策略)", 
            "📉 2. RSI/乖離率 (逆勢操盤)", 
            "📈 3. KD/MACD/基本面 (勝率攻略)", 
            "🤖 4. AI 策略預測", 
            "📋 5. 數據中心"
        ])

        default_range = [df.index[-30], df.index[-1]]
        def setup_chart(fig, height=1500):
            fig.update_xaxes(rangeslider_visible=False, rangeselector=dict(buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")]), font=dict(size=14, weight="bold"), y=1.02), range=default_range)
            fig.update_layout(height=height, template="plotly_white", hovermode="x unified")

        def add_k_vol(fig, display_df, row_k=1, row_v=2):
            fig.add_trace(go.Candlestick(x=display_df.index, open=display_df['Open'], high=display_df['High'], low=display_df['Low'], close=display_df['Close'], name='K線', increasing_line_color='#FF0000', decreasing_line_color='#006400'), row=row_k, col=1)
            colors = ['#FF0000' if display_df['Close'][i] >= display_df['Open'][i] else '#006400' for i in range(len(display_df))]
            fig.add_trace(go.Bar(x=display_df.index, y=display_df['Volume'], name='實體成交量', marker_color=colors, opacity=1.0), row=row_v, col=1)

        with t1:
            st.markdown("""<div class="strategy-box"><h3>📖 選項 1：MA 均線多頭排列與布林突破攻略</h3><ul><li><b>均線多頭排列</b>：當 MA5 > MA20 > MA60 時，股價處於主升段。</li><li><b>布林通道擠壓突破</b>：隨後突破上軌並伴隨實體大紅量柱，即是噴發訊號。</li></ul></div>""", unsafe_allow_html=True)
            fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.65, 0.35])
            add_k_vol(fig1, df); fig1.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='5MA', line=dict(color='#FF69B4', width=4)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20MA', line=dict(color='#FFA500', width=4)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='60MA', line=dict(color='#0000FF', width=4)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='布林上', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='布林下', line=dict(color='gray', dash='dash'), fill='tonexty'), row=1, col=1)
            setup_chart(fig1); st.plotly_chart(fig1, use_container_width=True); show_latest_metrics(latest, prev)

        with t2:
            st.markdown("""<div class="strategy-box"><h3>📖 選項 2：RSI 底部背離與 BIAS 均線回歸逆勢布局</h3><ul><li><b>RSI 底部背離</b>：股價新低但 RSI 低點不再破底，是精準抄底訊號。</li><li><b>負乖離操作</b>：當 60MA 乖離率低於 -10% 時，引發強勁回歸動能。</li></ul></div>""", unsafe_allow_html=True)
            fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.25, 0.25])
            add_k_vol(fig2, df, row_k=1, row_v=1)
            fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=4)), row=2, col=1)
            fig2.add_trace(go.Scatter(x=df.index, y=df['BIAS5'], name='BIAS 5', line=dict(color='red', width=3)), row=3, col=1)
            fig2.add_trace(go.Scatter(x=df.index, y=df['BIAS60'], name='BIAS 60', line=dict(color='blue', width=3)), row=3, col=1)
            setup_chart(fig2); st.plotly_chart(fig2, use_container_width=True); show_latest_metrics(latest, prev)

        with t3:
            st.markdown("""<div class="strategy-box"><h3>📖 選項 3：KD 高檔鈍化、MACD 趨勢確認與基本面篩選</h3><ul><li><b>KD 高檔鈍化</b>：K 值連三天 > 80，應「續抱」直到脫離鈍化區。</li><li><b>基本面安全邊際</b>：挑選 ROE > 15% 且 殖利率 > 4% 的標的。</li></ul></div>""", unsafe_allow_html=True)
            fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.25, 0.25])
            add_k_vol(fig3, df, row_k=1, row_v=1)
            fig3.add_trace(go.Scatter(x=df.index, y=df['K'], name='K線', line=dict(color='blue', width=4)), row=2, col=1)
            fig3.add_trace(go.Scatter(x=df.index, y=df['D'], name='D線', line=dict(color='orange', width=4)), row=2, col=1)
            fig3.add_trace(go.Scatter(x=df.index, y=df['DIF'], name='MACD DIF', line=dict(color='black', width=3)), row=3, col=1)
            fig3.add_trace(go.Scatter(x=df.index, y=df['DEA'], name='MACD DEA', line=dict(color='orange', width=3)), row=3, col=1)
            setup_chart(fig3); st.plotly_chart(fig3, use_container_width=True); show_latest_metrics(latest, prev)
            st.write("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            c2.metric("EPS", f"{info.get('trailingEps', 'N/A')}")
            c3.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
            c4.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

        with t4:
            st.subheader("🤖 AI 投資建議與未來預測")
            score = sum([1 if latest['K']>latest['D'] else 0, 1 if latest['Close']>latest['MA20'] else 0])
            st.success(f"### AI 綜合評價：{'🚀 積極買入' if score==2 else '⚖️ 中性觀望'}")
            years = 3; days = 252 * years; last_p = latest['Close']; total_ret = (df['Close'][-1] / df['Close'][0]) - 1
            ann_ret = (1 + total_ret) ** (1/3) - 1; ann_vol = df['Close'].pct_change().std() * np.sqrt(252)
            future_idx = [df.index[-1] + timedelta(days=i) for i in range(1, days + 1)]; path = last_p * np.exp((ann_ret - (ann_vol**2)/2) * (np.arange(1, days+1)/252))
            fig_p = go.Figure(); fig_p.add_trace(go.Scatter(x=df.index[-200:], y=df['Close'][-200:], name='歷史記錄'))
            fig_p.add_trace(go.Scatter(x=future_idx, y=path, name='AI 預測路徑', line=dict(dash='dash', color='red')))
            fig_p.update_layout(height=800, title="未來三年 AI 趨勢路徑預測", template="plotly_white"); st.plotly_chart(fig_p, use_container_width=True)

        with t5:
            st.subheader("🕒 數據中心")
            st.table(pd.DataFrame({"指標": ["收盤", "最高", "最低", "開盤", "成交量"], "數值": [f"{latest['Close']:.2f}", f"{latest['High']:.2f}", f"{latest['Low']:.2f}", f"{latest['Open']:.2f}", f"{int(latest['Volume']):,}"]}))
            
            # --- 一鍵存檔功能 ---
            st.divider()
            csv_filename = f"{stock_id}_3y_history.csv"
            save_path = os.path.join(desktop_path, csv_filename)
            if st.button(f"🚀 將此個股 3 年資料存到桌面 ({csv_filename})"):
                df.to_csv(save_path)
                st.success(f"✅ 存檔成功！檔案已放在您的桌面：{save_path}")
            
            st.dataframe(df.sort_index(ascending=False), use_container_width=True)

except Exception as e:
    st.error(f"系統發生問題: {e}")
