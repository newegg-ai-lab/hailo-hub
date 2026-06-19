# Hailo Hub Launcher
# Stops any running Streamlit first, then starts the unified hub.

Write-Host "Stopping any running Streamlit processes..." -ForegroundColor Yellow
taskkill /im streamlit.exe /f 2>$null
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "Starting Hailo Hub..." -ForegroundColor Green
Write-Host "Open: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

& "C:\Users\Test\hailo_ai\hailo_venv\Scripts\streamlit.exe" run "C:\Users\Test\hailo_ai\hailo_hub.py"
