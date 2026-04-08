import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from datetime import datetime

# --- 專業技術指標計算引擎 ---
def calculate_pro_indicators(df):
    # 1. 均線系統
    for m in [5, 10, 20, 60]:
        df[f'MA{m}'] = df['Close'].rolling(window=m).mean()
    
    # 2. 布林通道 (Bollinger Bands)
    df['BB_Mid'] = df['MA20']
    std = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['BB_Mid'] + (std * 2)
    df['BB_Low'] = df['BB_Mid'] - (std * 2)
    
    # 3. KD 指標 (Stochastic Oscillator)
    low_9 = df['Low'].rolling(window=9).min()
    high_9 = df['High'].rolling(window=9).max()
    df['RSV'] = ((df['Close'] - low_9) / (high_9 - low_9) * 100).fillna(50)
    k_values = [50.0]; d_values = [50.0]
    for i in range(1, len(df)):
        new_k = (2/3) * k_values[-1] + (1/3) * df['RSV'].iloc[i]
        new_d = (2/3) * d_values[-1] + (1/3) * new_k
        k_values.append(new_k); d_values.append(new_d)
    df['K'] = k_values; df['D'] = d_values
    
    # 4. MACD 指標
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean() # DIF
    df['Signal'] = df['MACD'].ewm(span=9).mean() # DEA
    df['Hist'] = (df['MACD'] - df['Signal']) * 2 # 柱狀圖
    
    # 5. RSI 指標
    def rsi(series, n):
        d = series.diff(); g = d.where(d > 0, 0); l = -d.where(d < 0, 0)
        return 100 - (100 / (1 + g.rolling(n).mean() / l.rolling(n).mean()))
    df['RSI12'] = rsi(df['Close'], 12)
    
    # 6. DMI 指標
    up = df['High'].diff(); down = -df['Low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['+DI'] = pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr * 100
    df['-DI'] = pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr * 100
    
    # 7. 基礎資訊
    df['漲跌'] = df['Close'].diff()
    df['成交張數'] = (df['Volume'] / 1000).astype(int)
    df['Golden_Cross'] = (df['MA20'] > df['MA60']) & (df['MA20'].shift(1) <= df['MA60'].shift(1))
    df['Death_Cross'] = (df['MA20'] < df['MA60']) & (df['MA20'].shift(1) >= df['MA60'].shift(1))
    
    return df

# --- 網頁佈局 ---
st.set_page_config(page_title="AI 台股全方位分析 Pro", layout="wide")

# 側邊欄
st.sidebar.header("🔍 股票搜尋")
user_input = st.sidebar.text_input("輸入代碼 (數字如 2330)", value="2330")
stock_id = f"{user_input}.TW" if user_input.isdigit() else user_input.upper()

st.sidebar.markdown("""
---
### 🖱️ 專業操作提示
- **K線對齊**：週末空白已移除。
- **投資決策**：說明直接呈現在圖表下方。
- **紅買綠賣**：箭頭標註關鍵訊號。
""")

import time
import requests

# --- 核心數據抓取配置 (偽裝成真人在使用 Chrome) ---
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_history(stock_id):
    """專門抓取 K 線歷史，失敗會重試一次"""
    ticker = yf.Ticker(stock_id, session=session)
    try:
        df = ticker.history(period="3y", interval="1d")
        if df.empty: # 嘗試第二次
            time.sleep(2)
            df = ticker.history(period="3y", interval="1d")
        if not df.empty:
            df.index = pd.to_datetime(df.index).date
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def get_cached_info(stock_id):
    """基本面資訊改為快取 24 小時，因為這最容易被鎖"""
    ticker = yf.Ticker(stock_id, session=session)
    try:
        return ticker.info
    except:
        return {}

def get_live_data(stock_id):
    ticker = yf.Ticker(stock_id, session=session)
    try:
        return ticker.fast_info
    except:
        return {"last_price": 0}

if stock_id:
    with st.spinner('連線至全球金融資料庫...'):
        df_all = get_cached_history(stock_id)
        stock_info = get_cached_info(stock_id)
        fast = get_live_data(stock_id)
        
        if df_all.empty:
            st.error("⚠️ 偵測到流量異常：Yahoo Finance 暫時限制了您的訪問。請等待 10 分鐘後再重新整理網頁。")
            st.info("💡 提示：建議不要在短時間內快速切換大量不同的股票代碼。")
            st.stop()
            
        live_price = fast.get('last_price', df_all['Close'].iloc[-1])
        live_time = datetime.now().strftime('%H:%M:%S')

    if not df_all.empty:
        st.markdown(f"# 📈 {stock_info.get('longName', stock_id)} ({stock_id})")
        st.caption(f"🚀 **即時行情與專業決策模式** | 最後更新：{live_time}")
        
        df_all = calculate_pro_indicators(df_all)
        last = df_all.iloc[-1]; prev = df_all.iloc[-2]

        # 即時看板
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("最新成交價", f"{live_price:.2f}", f"{live_price - prev['Close']:.2f}")
        m2.metric("今日開盤", f"{last['Open']:.2f}")
        m3.metric("今日最高", f"{last['High']:.2f}")
        m4.metric("今日最低", f"{last['Low']:.2f}")
        m5.metric("今日成交", f"{int(last['成交張數']):,} 張")
        st.divider()

        tabs = st.tabs(["🏠 均線與布林決策", "⚡ KD 與 RSI 決策", "📉 MACD 趨勢決策", "🏢 基本面分析", "📑 價量明細查詢", "🤖 AI 智能診斷"])
        df_plot = df_all.tail(120)
        
        PLOT_HEIGHT = 1000
        RANGEBREAKS = [dict(bounds=["sat", "mon"])]
        COMMON_X_AXIS = dict(type='date', showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat='%Y-%m-%d', tickangle=-45, rangebreaks=RANGEBREAKS, spikemode='across', showspikes=True, spikedash='dash')
        COMMON_Y_AXIS = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', side='right', showspikes=True, spikedash='dash')
        COMMON_LAYOUT = dict(height=PLOT_HEIGHT, template="plotly_dark", margin=dict(l=50, r=80, t=50, b=50), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode='x unified', bargap=0.15)
        CHART_CONFIG = {'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False}

        with tabs[0]:
            st.subheader("均線交叉與布林通道決策圖 (MA 5/20/60 + Bollinger)")
            fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig1.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='K線', increasing_line_width=3, decreasing_line_width=3), 1, 1)
            fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA5'], line=dict(color='#00FF00', width=1.5), name='MA5 (綠)'), 1, 1)
            fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], line=dict(color='orange', width=2), name='MA20 (橘)'), 1, 1)
            fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA60'], line=dict(color='cyan', width=2), name='MA60 (青)'), 1, 1)
            fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Up'], line=dict(color='gray', width=1, dash='dash'), name='布林上軌'), 1, 1)
            fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Low'], line=dict(color='gray', width=1, dash='dash'), name='布林下軌'), 1, 1)
            
            # --- 買進/賣出提醒箭頭 ---
            gold_pts = df_plot[df_plot['Golden_Cross']]
            fig1.add_trace(go.Scatter(x=gold_pts.index, y=gold_pts['MA20']*0.97, mode='markers+text', name='買進進場', marker=dict(symbol='triangle-up', size=25, color='red'), text="<b>買入</b>", textposition="bottom center"), 1, 1)
            death_pts = df_plot[df_plot['Death_Cross']]
            fig1.add_trace(go.Scatter(x=death_pts.index, y=death_pts['MA20']*1.03, mode='markers+text', name='賣出退場', marker=dict(symbol='triangle-down', size=25, color='#00FF00'), text="<b>賣出</b>", textposition="top center"), 1, 1)
            
            vol_colors = ['#FF0000' if c >= o else '#00FF00' for c, o in zip(df_plot['Close'], df_plot['Open'])]
            fig1.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=vol_colors, name='成交量', opacity=0.9), 2, 1)
            fig1.update_layout(**COMMON_LAYOUT, xaxis_rangeslider_visible=False)
            fig1.update_xaxes(**COMMON_X_AXIS); fig1.update_yaxes(**COMMON_Y_AXIS)
            st.plotly_chart(fig1, width="stretch", config=CHART_CONFIG)
            
            st.success("""
            **🏠 均線與布林投資決策標準：**
            1. **🌟 黃金交叉 (紅色向上箭頭)**：當 **MA20 向上突破 MA60** 時，趨勢由空轉多，為強烈買進訊號。
            2. **💀 死亡交叉 (綠色向下箭頭)**：當 **MA20 向下墜破 MA60** 時，趨勢正式轉空，建議果斷賣出避險。
            3. **🛡️ 布林通道決策**：股價觸及上軌代表短線過熱；觸及下軌代表超跌醞釀反彈。
            """)

        with tabs[1]:
            st.subheader("KD 與 RSI 決策分析 (含 K/D 指標說明)")
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
            fig2.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='K線', increasing_line_width=2, decreasing_line_width=2), 1, 1)
            # KD
            fig2.add_trace(go.Scatter(x=df_plot.index, y=df_plot['K'], name='K線 (藍-快隨機)', line=dict(color='#00FFFF', width=2.5)), 2, 1)
            fig2.add_trace(go.Scatter(x=df_plot.index, y=df_plot['D'], name='D線 (橘-慢隨機)', line=dict(color='#FF4500', width=2.5)), 2, 1)
            fig2.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI12'], name='RSI12', line=dict(color='#FFD700', width=2.5)), 2, 1)
            fig2.update_layout(**COMMON_LAYOUT, xaxis_rangeslider_visible=False)
            fig2.update_xaxes(**COMMON_X_AXIS); fig2.update_yaxes(**COMMON_Y_AXIS)
            st.plotly_chart(fig2, width="stretch", config=CHART_CONFIG)
            
            st.info("""
            **⚡ KD & RSI 投資決策詳細說明：**
            1. **線條定義**：藍色為 **K 線 (快隨機指標)**，反應靈敏；橘色為 **D 線 (慢隨機指標)**，由 K 值平均而來，趨勢較穩。
            2. **🌟 KD 黃金交叉**：當 **K 線由下往上突破 D 線** 且位於 20 附近的低檔區，是極佳的短線買進點。
            3. **💀 KD 死亡交叉**：當 **K 線由上往下跌破 D 線** 且位於 80 附近的高檔區，代表漲勢動能耗盡，應賣出。
            """)

        with tabs[2]:
            st.subheader("MACD 趨勢決策分析 (含 DIF/DEA 說明)")
            fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
            fig3.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='K線', increasing_line_width=2, decreasing_line_width=2), 1, 1)
            # MACD
            macd_colors = ['#FF0000' if v >= 0 else '#00FF00' for v in df_plot['Hist']]
            fig3.add_trace(go.Bar(x=df_plot.index, y=df_plot['Hist'], name='MACD 柱狀圖', marker_color=macd_colors), 2, 1)
            fig3.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], name='DIF快線 (粉)', line=dict(color='#FF00FF', width=3)), 2, 1)
            fig3.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Signal'], name='DEA慢線 (黃)', line=dict(color='yellow', width=3)), 2, 1)
            fig3.update_layout(**COMMON_LAYOUT, xaxis_rangeslider_visible=False)
            fig3.update_xaxes(**COMMON_X_AXIS); fig3.update_yaxes(**COMMON_Y_AXIS)
            st.plotly_chart(fig3, width="stretch", config=CHART_CONFIG)
            
            st.warning("""
            **📉 MACD 投資決策詳細說明：**
            1. **線條定義**：粉紅色為 **DIF (快線)**，代表短期動能與乖離；黃色為 **DEA (慢線)**，代表中期波段趨勢。
            2. **🌟 買進訊號**：當 **DIF (粉紅) 向上突破 DEA (黃)**，代表多頭動能轉強，波段上攻。
            3. **💀 賣出訊號**：當 **DIF (粉紅) 向下穿過 DEA (黃)**，代表動能衰退，應防範拉回。
            """)

        with tabs[3]:
            st.subheader("🏢 基本面財務數據分析")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("本益比 (PE)", f"{stock_info.get('trailingPE', 'N/A')}")
            col2.metric("現金殖利率", f"{stock_info.get('dividendYield', 0)*100:.2f}%")
            col3.metric("ROE (獲利效率)", f"{stock_info.get('returnOnEquity', 0)*100:.2f}%")
            col4.metric("PB比 (價值)", f"{stock_info.get('priceToBook', 'N/A')}")

        with tabs[4]:
            st.subheader("📑 價量明細表 (過去三年歷史)")
            hist_df = df_all[['Open', 'High', 'Low', 'Close', 'Volume', '成交張數']].copy()
            hist_df.columns = ['開盤價', '最高價', '最低價', '收盤價', '成交量', '成交張數(張)']
            st.dataframe(hist_df.sort_index(ascending=False), width="stretch")

        with tabs[5]:
            st.subheader("🤖 AI 智能診斷與長線展望")
            tech_is_bull = last['+DI'] > last['-DI']
            roe_val = stock_info.get('returnOnEquity', 0)
            if tech_is_bull and roe_val > 0.15: st.success("🌟 **診斷：極力推薦**")
            elif tech_is_bull: st.success("🚀 **診斷：偏多操作**")
            elif roe_val > 0.15: st.warning("⚖️ **診斷：價值佈局**")
            else: st.error("💀 **診斷：保守觀望**")
            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("### 📅 三年看法")
                if roe_val > 0.15: st.success("🟢 看好成長")
                else: st.info("🟡 平穩發展")
            with c2:
                st.write("### 📅 五年看法")
                if roe_val > 0.20: st.success("💎 產業龍頭")
                else: st.info("📈 穩健領息")
            with c3:
                st.write("### 📅 十年看法")
                if stock_info.get('revenueGrowth', 0) > 0.1: st.success("🌍 規模翻倍")
                else: st.info("🏦 價值典範")

st.caption("🚀 專業股市全方位分析 Pro | 技術決策與布林整合終極版")
