@echo off
chcp 65001 >nul
echo ==========================================
echo 超高分辨率图像生成系统 - 启动脚本
echo ==========================================

:: 设置环境变量
set ARK_API_KEY=sk-xIr6z0QlYiu498lwe406xbeuxXeIE6Mp6neFxkhABigECvQ9
set ARK_MODEL=ep-20260228221135-66v8k

echo.
echo [配置信息]
echo ARK_API_KEY: %ARK_API_KEY:~0,20%...
echo ARK_MODEL: %ARK_MODEL%
echo.

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 启动 Streamlit
echo [启动 Web 界面...]
streamlit run app.py

pause
