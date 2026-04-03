@echo off
title 台股分析儀表板啟動器
echo 正在嘗試啟動台股分析儀表板，請稍候...
cd /d "C:\Users\jim97\Desktop\TaiwanStockAnalysis"

:: 檢查 Python 是否存在並啟動
python -m streamlit run stock_analysis_app.py
if %ERRORLEVEL% NEQ 0 (
    echo [錯誤] 嘗試使用 'python' 指令失敗，切換為 'py' 指令...
    py -m streamlit run stock_analysis_app.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo [啟動失敗] 系統找不到 Python 或 Streamlit。
    echo 請確保您已安裝 Python 並且執行過 'pip install streamlit'。
    echo ========================================================
    pause
)
