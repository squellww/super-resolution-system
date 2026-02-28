# 超高分辨率图像生成系统 - PowerShell 启动脚本

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "超高分辨率图像生成系统 - 启动脚本" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 设置环境变量
$env:ARK_API_KEY = "sk-xIr6z0QlYiu498lwe406xbeuxXeIE6Mp6neFxkhABigECvQ9"
$env:ARK_MODEL = "ep-20260228221135-66v8k"

Write-Host ""
Write-Host "[配置信息]" -ForegroundColor Yellow
Write-Host "ARK_API_KEY: $($env:ARK_API_KEY.Substring(0,20))..." -ForegroundColor Gray
Write-Host "ARK_MODEL: $($env:ARK_MODEL)" -ForegroundColor Gray
Write-Host ""

# 激活虚拟环境
& .\venv\Scripts\Activate.ps1

# 启动 Streamlit
Write-Host "[启动 Web 界面...]" -ForegroundColor Green
streamlit run app.py
